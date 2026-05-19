# MLX dots.mocr command

This document describes the local Apple Silicon / MLX command added to this repo for parsing PDFs and images with an `mlx-community` dots.mocr model.

The command produces:

- structured Markdown: `<output-name>.md`
- cropped source-document assets: `<output-name>_imgs/*.png`
- structured layout JSON: `<output-name>.json`
- raw model output per page: `<output-name>_page_###_raw.txt`
- page preview images: `<output-name>_pages/page_###.png`
- model-input page images: `<output-name>_mlx_input/page_###.png`
- layout-overlay preview images: `<output-name>_layout/page_###.jpg`
- a static side-by-side review UI: `<output-name>_viewer.html`

The console script is:

```bash
dots-mocr-mlx
```

It defaults to:

```text
mlx-community/dots.mocr-4bit
```

You can point it at other MLX quantizations, for example:

```text
mlx-community/dots.mocr-bf16
mlx-community/dots.mocr-mxfp4
mlx-community/dots.mocr-nvfp4
```

## 1. Setup on another Mac

### Hardware and OS assumptions

Recommended:

- Apple Silicon Mac.
- macOS with working Xcode command line tools.
- Enough disk space for the Python environment and model cache.
  - The 4-bit model is roughly 3.5 GB on first download.
  - The bf16 model is larger, roughly 6 GB of model weights.
  - Keep extra space available for Hugging Face cache, Python wheels, and generated page images.

### Install system prerequisites

If the Mac does not already have the command line tools:

```bash
xcode-select --install
```

Install Python, Git, and uv with Homebrew if needed:

```bash
brew install python git uv
```

Check the basics:

```bash
python3 --version
uv --version
git --version
```

### Clone or enter the repo

```bash
git clone https://github.com/rednote-hilab/dots.mocr.git
cd dots.mocr
```

If these MLX changes are on a branch or local patch, check out or apply that version before installing.

### Create a virtual environment

This repo was validated with Python 3.12 in a uv-managed `.venv`:

```bash
uv venv .venv --python 3.12
```

If Python 3.12 is not available, install it with Homebrew or let uv locate a compatible Python >= 3.10.

### Install the package and dependencies

From the repo root:

```bash
uv pip install --python .venv/bin/python -e .
```

That installs the console script into:

```bash
.venv/bin/dots-mocr-mlx
```

Verify the command is available:

```bash
.venv/bin/dots-mocr-mlx --help
```

Expected help should include options such as:

```text
--backend
--output-dir / -o
--output-name
--model
--api-base-url
--api-key
--api-model
--prompt
--dpi
--fitz-preprocess
--no-page-header-footer
--infer-heading-hierarchy
--max-tokens
--temperature
--top-p
--repetition-penalty
--num-thread
--parallel-mode
--performance-preset
--prefill-step-size
--view
--host
--port
--open-browser
```

### Hugging Face access

The public `mlx-community` models can be downloaded without a token, but unauthenticated downloads may be slower or rate-limited.

Optional:

```bash
export HF_TOKEN="..."
```

Do not commit tokens or put them in this repo.

## 2. Basic usage

### Parse an image or PDF without launching the viewer

```bash
.venv/bin/dots-mocr-mlx /path/to/input.pdf \
  --output-dir output_mlx \
  --output-name parsed \
  --model mlx-community/dots.mocr-4bit \
  --prefill-step-size 512 \
  --temperature 0 \
  --max-tokens 24000
```

For an image input:

```bash
.venv/bin/dots-mocr-mlx /path/to/page.png \
  --output-dir output_mlx \
  --output-name page \
  --model mlx-community/dots.mocr-4bit \
  --prefill-step-size 512 \
  --temperature 0 \
  --max-tokens 24000
```

### Parse and launch the comparison UI

Add `--view`:

```bash
.venv/bin/dots-mocr-mlx /path/to/input.pdf \
  --output-dir output_mlx \
  --output-name parsed \
  --model mlx-community/dots.mocr-4bit \
  --prefill-step-size 512 \
  --temperature 0 \
  --max-tokens 24000 \
  --view
```

The command prints a local URL like:

```text
Viewer: http://127.0.0.1:7860/parsed_viewer.html
Serving comparison viewer. Press Ctrl-C to stop.
```

Use `Ctrl-C` in that terminal to stop the viewer.

Open the browser automatically:

```bash
.venv/bin/dots-mocr-mlx /path/to/input.pdf \
  --output-dir output_mlx \
  --output-name parsed \
  --view \
  --open-browser
```

If port 7860 is already in use, let the OS choose an available port:

```bash
.venv/bin/dots-mocr-mlx /path/to/input.pdf \
  --output-dir output_mlx \
  --output-name parsed \
  --view \
  --port 0
```

### Use a local oMLX API backend

The newer `dots-mocr-mlx` command can call an OpenAI-compatible local oMLX server instead of loading the MLX model in-process:

```bash
.venv/bin/dots-mocr-mlx /path/to/input.pdf \
  --backend omlx-api \
  --api-base-url http://127.0.0.1:8000/v1 \
  --api-key 1220 \
  --api-model dots.mocr-bf16 \
  --output-dir output_mlx \
  --output-name parsed \
  --temperature 0 \
  --top-p 1.0 \
  --max-tokens 24000 \
  --frequency-penalty 0 \
  --presence-penalty 0 \
  --seed 0 \
  --repetition-penalty 1.05
```

Those oMLX values are also the command defaults for API mode. You usually do not need to set them as backend-global defaults; keeping them explicit at the command layer makes runs easier to reproduce and avoids changing other oMLX models.

The oMLX API path sends the image and prompt directly. It intentionally does not prepend the vLLM-only `<|img|><|imgpad|><|endofimg|>` sentinel, because local oMLX image requests work without it.

For a single local oMLX server, `--parallel-mode server-safe` is the default because concurrent generation requests can be slower than sequential processing. Use `--parallel-mode fixed --num-thread N` only after benchmarking your oMLX server:

```bash
.venv/bin/dots-mocr-mlx /path/to/input.pdf \
  --backend omlx-api \
  --parallel-mode fixed \
  --num-thread 2
```

The oMLX path also exposes image-cost presets:

```bash
--performance-preset quality   # existing default image size behavior
--performance-preset balanced  # uses --max-pixels 1600000 unless overridden
--performance-preset fast      # uses --max-pixels 1000000 unless overridden
```

An explicit `--max-pixels` value always wins over the preset.

Measured on the first 8 pages of `assets/mlx/book-chapter-2.pdf` with local `dots.mocr-bf16`:

| Run | Threads | Max pixels | Total seconds | Output structure |
| --- | ---: | ---: | ---: | --- |
| Baseline | 1 | default | 356.11 | cells `[9,87,5,6,6,6,8,9]` |
| Balanced | 1 | 1,600,000 | 222.39 | cells `[9,87,5,6,6,6,8,9]` |
| Fast | 1 | 1,000,000 | 186.26 | cells `[10,88,5,6,6,6,8,9]` |

On that measured 8-page book chapter sample, `balanced` reduced wall time by about 37% with identical cell/category counts. `fast` reduced wall time by about 48% but changed two detected cells, so review quality before using it as a default. `--num-thread 2` was slower on the same local oMLX server and was interrupted after long waits, so benchmark before forcing parallel requests.

### Demonstrated Russian showcase command

This exact command was run successfully in this working tree:

```bash
.venv/bin/dots-mocr-mlx assets/showcase/origin/russian.png \
  --output-dir output_mlx_demo \
  --output-name russian \
  --model mlx-community/dots.mocr-4bit \
  --prefill-step-size 512 \
  --temperature 0 \
  --max-tokens 24000
```

It produced:

```text
output_mlx_demo/russian.md
output_mlx_demo/russian.json
output_mlx_demo/russian_viewer.html
output_mlx_demo/russian_page_001_raw.txt
output_mlx_demo/russian_imgs/page_001_img_007.png
output_mlx_demo/russian_pages/page_001.png
output_mlx_demo/russian_mlx_input/page_001.png
output_mlx_demo/russian_layout/page_001.jpg
```

The viewer was then served at:

```text
http://127.0.0.1:7860/russian_viewer.html
```

The browser UI showed:

- left panel: source/layout preview
- right panel: result display
- tabs: Markdown Render Preview, Formatted JSON, Raw Markdown
- clicking a layout box in the left preview scrolls the Markdown preview to the matching output block and briefly highlights it

### Expected output layout

For this command:

```bash
.venv/bin/dots-mocr-mlx input.pdf --output-dir out --output-name report
```

Expected output is:

```text
out/
  report.md
  report.json
  report_viewer.html
  report_page_001_raw.txt
  report_page_002_raw.txt
  ...
  report_imgs/
    page_001_img_###.png
    page_002_img_###.png
    ...
  report_pages/
    page_001.png
    page_002.png
    ...
  report_mlx_input/
    page_001.png
    page_002.png
    ...
  report_layout/
    page_001.jpg
    page_002.jpg
    ...
```

For multi-page PDFs, the Markdown combines pages with page comments and separators.

## 3. Useful options and notes

### Model choice

Default:

```bash
--model mlx-community/dots.mocr-4bit
```

Higher precision, larger download:

```bash
--model mlx-community/dots.mocr-bf16
```

Other MLX quantizations may also work if published by `mlx-community`:

```bash
--model mlx-community/dots.mocr-mxfp4
--model mlx-community/dots.mocr-nvfp4
```

You can also set the default model by environment variable:

```bash
export DOTS_MOCR_MLX_MODEL=mlx-community/dots.mocr-bf16
```

### Memory tuning

For Apple Silicon memory pressure during MLX prefill, use:

```bash
--prefill-step-size 512
```

Try smaller values if needed.

### PDF rendering quality

PDFs are rendered to images with PyMuPDF before MLX inference.

Default:

```bash
--dpi 200
```

For small text or dense tables, try:

```bash
--dpi 250
```

or:

```bash
--dpi 300
```

Higher DPI increases memory use and runtime.

### Image preprocessing

For image inputs, the command normally uses the image directly.

To route an image through the same PyMuPDF-style preprocessing path first:

```bash
--fitz-preprocess
```

### Markdown header formatting

Detected heading categories are now emitted as real Markdown headings:

```text
Title -> # Title
Section-header -> ## Section heading
Subsection-header -> ### Subsection heading
Subsubsection-header -> #### Subsubsection heading
```

If the model output itself includes a Markdown heading prefix in the cell text, that also wins over the category fallback. For example, some checkpoints emit a broad `Section-header` category with text like `### 2016–2019: From order to delivery`; the converter preserves that as an H3 instead of flattening it to H2.

Caption cells are now marked as captions in Markdown while staying portable:

```text
Caption: Bow -> _Caption: Bow_
Figure 1. System diagram. -> _Figure 1. System diagram._
Table 1. Example data. -> _Table 1. Example data._
```

If a future model output includes an explicit level field, that wins:

```text
markdown_level
heading_level
header_level
level
```

Supported values include integers `1` through `6` and strings like `h2` / `h3`.

Page headers and footers are not automatically promoted to Markdown headings because they are often running headers, issue titles, dates, or page numbers rather than document structure.

### Optional heading hierarchy inference

For screenshots or PDFs where the model emits layout correctly but underspecifies the semantic outline, enable:

```bash
--infer-heading-hierarchy
```

This mode is opt-in. It keeps the default converter conservative, then adds context-aware fixes:

- promotes the first article/page title to H1 when no H1 is already present
- promotes a first-page book/chapter title emitted as `Section-header` to H1 when it is title-like and not a common section label such as `Abstract` or `Introduction`
- infers numbered section depth such as `1.` -> H2, `1.1.` -> H3, and `6.2.1.` -> H4 when the model did not already emit Markdown prefixes
- infers visually smaller same-page section headers as H3 when the model labeled all of them as `Section-header`
- demotes model-emitted H1 section headings on later PDF pages to H2 so a multi-page document keeps one document H1

For web/article screenshots, this pairs well with page chrome filtering:

```bash
.venv/bin/dots-mocr-mlx assets/mlx/wikipedia.png \
  --output-dir output_mlx_wikipedia \
  --output-name wikipedia_bf16_hierarchy \
  --model mlx-community/dots.mocr-bf16 \
  --prefill-step-size 512 \
  --temperature 0 \
  --max-tokens 24000 \
  --no-page-header-footer \
  --infer-heading-hierarchy
```

On `assets/mlx/wikipedia.png`, this produced:

```text
# MV *Hondius*
## Description
## History
### 2016–2019: From order to delivery
```

The bf16 model is recommended for this workflow. The 4-bit model is faster and smaller, but in testing it did not reliably distinguish the Wikipedia H3 subsection. A raw Markdown prompt can produce a good outline, but it loses layout JSON and cropped assets, so the layout prompt plus `--infer-heading-hierarchy` is the safer export path.

### Markdown filtering

To omit detected page headers and footers from Markdown:

```bash
--no-page-header-footer
```

### Prompts

Default prompt mode:

```text
prompt_layout_all_en
```

Available prompt modes are whatever is exposed by `dots_mocr.utils.prompts.dict_promptmode_to_prompt`; the CLI help lists them.

Example:

```bash
.venv/bin/dots-mocr-mlx input.png \
  --prompt prompt_ocr \
  --output-dir output_mlx \
  --output-name ocr_only
```

You can also pass a full prompt string:

```bash
.venv/bin/dots-mocr-mlx input.png \
  --custom-prompt "Parse this document into structured markdown." \
  --output-dir output_mlx \
  --output-name custom
```

## 4. Troubleshooting

### First run is slow

The first run downloads the MLX model from Hugging Face and compiles/initializes MLX kernels. Later runs should start faster because the model is cached.

### Hugging Face warning about unauthenticated requests

This warning is normal without `HF_TOKEN`:

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

Set `HF_TOKEN` if higher rate limits are needed.

### `Qwen2VLVideoProcessor requires the Torchvision library`

Install or reinstall with the updated requirements:

```bash
uv pip install --python .venv/bin/python -e .
```

The MLX path declares `torchvision` because the Transformers processor stack can require it even for still-image OCR.

### Dependency resolver conflict around Transformers

The old repo pinned:

```text
transformers==4.57.6
```

`mlx-vlm` requires newer Transformers, so this MLX revision changed the dependency to:

```text
transformers>=5.5.0
```

If another part of the repo depends on the old Transformers pin, run its tests before relying on the combined environment.

### Native cairo / cairosvg errors

The MLX command is designed to avoid importing the heavy parser/SVG/cairo path at startup. That is why `dots_mocr/__init__.py` now lazy-loads `DotsMOCRParser`.

If you use the older parser/demo path and see cairo errors on macOS, install cairo with Homebrew:

```bash
brew install cairo
```

If needed for direct `cairosvg` imports:

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix cairo)/lib:/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH"
```

### Viewer port already in use

Use a different port:

```bash
--port 7861
```

or let the OS pick one:

```bash
--port 0
```

## 5. Validation commands

Run focused tests for the MLX command:

```bash
.venv/bin/python -m pytest tests/test_mlx_cli.py -q
```

Run a compile check:

```bash
.venv/bin/python -m compileall -q dots_mocr tests
```

Check CLI wiring:

```bash
.venv/bin/dots-mocr-mlx --help
```

Run the oMLX benchmark harness:

```bash
.venv/bin/python tools/benchmark_omlx_cli.py \
  --input assets/mlx/book-chapter-2.pdf \
  --pages 8 \
  --output-dir /tmp/dots_mocr_omlx_bench \
  --threads 1 \
  --max-pixels default 1600000 1000000 \
  --parallel-mode server-safe
```

The validation run for this revision passed:

```text
13 passed, 5 warnings
```

The compile check also passed with no errors.

## 6. Detailed account of changes from the last commit

Last commit observed while writing this document:

```text
23f3e56 update
```

No commit was created for these changes at the time this file was written.

### Intentional source changes

#### `dots_mocr/mlx_cli.py`

New module implementing the MLX command path.

Main responsibilities:

- defines the default model as `DOTS_MOCR_MLX_MODEL` or `mlx-community/dots.mocr-4bit`
- loads MLX models through `mlx_vlm.load`
- runs generation through `mlx_vlm.generate`
- accepts PDF and image inputs
- renders PDF pages through PyMuPDF
- optionally preprocesses image inputs through the PyMuPDF helper path
- post-processes dots.mocr layout output through existing layout utilities
- crops detected picture/figure regions into `<output-name>_imgs/`
- emits Markdown with relative crop links
- emits Markdown headings from detected heading layout categories
- emits structured JSON with page metadata, cell categories, bounding boxes, and text
- writes raw model output per page
- writes source page images, MLX input images, and layout overlay images
- builds a static side-by-side HTML viewer
- serves the viewer with Python `http.server`
- exposes the command-line parser and `main()` entry point

Top-level functions/classes added include:

```text
InferenceRunner
PageInput
PageOutput
PipelineResult
MlxDotsRunner
layout_cells_to_markdown_with_crops
load_input_pages
process_page
render_markdown
build_viewer_html
run_pipeline
serve_viewer
parse_args
main
```

This was intentionally kept separate from `dots_mocr/parser.py` so the MLX path does not require the heavier vLLM/HF parser stack.

#### `dots_mocr/__init__.py`

Changed from eager parser import:

```python
from .parser import DotsMOCRParser
```

to a lazy `__getattr__` export.

Reason:

- importing `dots_mocr.mlx_cli` should not immediately import the old parser path
- the old parser path can transitively import optional SVG/cairo/vLLM-related dependencies
- lazy loading keeps the lightweight MLX CLI importable on a normal Mac setup

The public `DotsMOCRParser` export is preserved when explicitly requested.

#### `setup.py`

Added console-script entry point:

```python
entry_points={
    'console_scripts': [
        'dots-mocr-mlx=dots_mocr.mlx_cli:main',
    ],
}
```

This makes the single command available after editable install:

```bash
.venv/bin/dots-mocr-mlx
```

#### `requirements.txt`

Changed:

```text
transformers==4.57.6
```

to:

```text
transformers>=5.5.0
```

Added:

```text
mlx-vlm>=0.4.1
torchvision
markdown
```

Reasons:

- `mlx-vlm` is required for MLX model loading/generation.
- `mlx-vlm` requires newer Transformers than the previous pin.
- `torchvision` is required by the Transformers Qwen2VL processor stack used while loading the model processor.
- `markdown` is used to render Markdown into HTML for the comparison viewer.

#### `tests/test_mlx_cli.py`

New focused test file.

Coverage added:

- Markdown generation from layout cells, including crop export and relative image links.
- Static viewer HTML generation, including rendered Markdown, JSON tab, and raw Markdown tab.
- `run_pipeline` behavior using a fake runner, so tests do not download/load the real MLX model.

#### `MLX.md`

This documentation file.

### Intentional generated/demo artifacts

The Russian showcase verification produced local output under:

```text
output_mlx_demo/
```

Observed files:

```text
output_mlx_demo/russian.md
output_mlx_demo/russian.json
output_mlx_demo/russian_viewer.html
output_mlx_demo/russian_page_001_raw.txt
output_mlx_demo/russian_imgs/page_001_img_007.png
output_mlx_demo/russian_pages/page_001.png
output_mlx_demo/russian_mlx_input/page_001.png
output_mlx_demo/russian_layout/page_001.jpg
```

Observed sizes from verification:

```text
russian.md: 23,944 bytes
russian.json: 29,671 bytes
russian_viewer.html: 83,141 bytes
russian_page_001_raw.txt: 26,266 bytes
page_001_img_007.png: 467x279 crop, 74,821 bytes
```

These are generated outputs, not source code.

### Current working-tree items that are not part of the MLX implementation

The working tree also showed these deviations from the last commit:

```text
.gitignore
.DS_Store
assets/.DS_Store
assets/showcase/.DS_Store
```

Notes:

- `.gitignore` contained additional `.codex/...` ignore/allow patterns. This appeared before the MLX implementation work and is not required by the MLX command.
- `.DS_Store` files are macOS Finder artifacts and are not required by the MLX command.

### Behavioral deviations from the previous repo state

Before this revision, the repo had the original parser/demo flow oriented around the existing parser stack.

This revision adds a separate MLX-oriented path with these intentional differences:

- uses `mlx-vlm` instead of vLLM for local Apple Silicon inference
- defaults to an `mlx-community` quantized model
- exposes one console command instead of requiring a Python script invocation
- exports Markdown plus cropped image links by default
- promotes detected `Title`, `Section-header`, and subsection header cells into Markdown `#` / `##` / `###` headings
- adds opt-in heading hierarchy inference via `--infer-heading-hierarchy`
- writes JSON and raw output for debugging/review
- creates source/layout page images for inspection
- uses a static HTML comparison viewer instead of requiring Gradio for the review UI
- links layout boxes in the viewer to corresponding rendered Markdown blocks for click-to-scroll review
- lazy-loads the old parser export to avoid optional heavy dependencies during MLX CLI startup

### Verification performed for this revision

Focused tests:

```bash
.venv/bin/python -m pytest tests/test_mlx_cli.py -q
```

Result:

```text
13 passed, 5 warnings
```

Compile check:

```bash
.venv/bin/python -m compileall -q dots_mocr tests
```

Result: passed with no output.

CLI help:

```bash
.venv/bin/dots-mocr-mlx --help
```

Result: printed expected CLI usage and options.

Heading hierarchy smoke tests were also run with `mlx-community/dots.mocr-bf16` and:

```bash
--prefill-step-size 512 --temperature 0 --max-tokens 24000 --no-page-header-footer --infer-heading-hierarchy
```

Validated files:

```text
assets/mlx/wikipedia.png
assets/mlx/arxiv.pdf
assets/mlx/book-chapter.pdf
assets/mlx/rho-class.pdf
```

Generated review artifacts:

```text
output_mlx_wikipedia/wikipedia_bf16_hierarchy.md
output_mlx_wikipedia/wikipedia_bf16_hierarchy.json
output_mlx_wikipedia/wikipedia_bf16_hierarchy_viewer.html
output_mlx_assets/arxiv_bf16_hierarchy.md
output_mlx_assets/arxiv_bf16_hierarchy.json
output_mlx_assets/arxiv_bf16_hierarchy_viewer.html
output_mlx_assets/book_chapter_bf16_hierarchy.md
output_mlx_assets/book_chapter_bf16_hierarchy.json
output_mlx_assets/book_chapter_bf16_hierarchy_viewer.html
output_mlx_assets/rho_class_bf16_hierarchy.md
output_mlx_assets/rho_class_bf16_hierarchy.json
output_mlx_assets/rho_class_bf16_hierarchy_viewer.html
```

Real fixture run:

```bash
.venv/bin/dots-mocr-mlx assets/showcase/origin/russian.png \
  --output-dir output_mlx_demo \
  --output-name russian \
  --model mlx-community/dots.mocr-4bit \
  --prefill-step-size 512 \
  --temperature 0 \
  --max-tokens 24000
```

Result: generated Markdown, JSON, raw output, crop image, source page image, MLX input image, layout overlay image, and viewer HTML.

Viewer check:

```text
http://127.0.0.1:7860/russian_viewer.html
```

Result:

```text
HTTP 200
content-type: text/html
File Preview present
Result Display present
Markdown Render Preview tab present
Formatted JSON tab present
Raw Markdown tab present
```
