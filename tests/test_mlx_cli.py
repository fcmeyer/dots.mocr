import json
import threading
import time
from pathlib import Path

import fitz
from PIL import Image

from dots_mocr.mlx_cli import (
    OmlxApiRunner,
    build_viewer_html,
    layout_cells_to_markdown_with_crops,
    parse_args,
    resolve_max_pixels,
    run_pipeline,
)
from tools.benchmark_omlx_cli import summarize_run


def test_layout_cells_to_markdown_writes_picture_crops_and_relative_links(tmp_path):
    image = Image.new("RGB", (100, 80), "white")
    for x in range(10, 40):
        for y in range(20, 60):
            image.putpixel((x, y), (255, 0, 0))

    cells = [
        {"category": "Title", "bbox": [0, 0, 90, 15], "text": "A title"},
        {"category": "Picture", "bbox": [10, 20, 40, 60], "text": "ignored"},
        {"category": "Text", "bbox": [0, 61, 99, 79], "text": "Body text"},
    ]

    imgs_dir = tmp_path / "sample_imgs"
    markdown = layout_cells_to_markdown_with_crops(
        image=image,
        cells=cells,
        imgs_dir=imgs_dir,
        relative_img_dir="sample_imgs",
        page_no=0,
    )

    crop_path = imgs_dir / "page_001_img_002.png"
    assert crop_path.exists()
    assert Image.open(crop_path).size == (30, 40)
    assert "A title" in markdown
    assert "![Picture page 1 cell 2](sample_imgs/page_001_img_002.png)" in markdown
    assert "Body text" in markdown
    assert "data:image" not in markdown


def test_layout_cells_to_markdown_marks_caption_cells_in_markdown(tmp_path):
    image = Image.new("RGB", (120, 100), "white")
    cells = [
        {"category": "Picture", "bbox": [10, 10, 50, 40], "text": ""},
        {"category": "Caption", "bbox": [10, 45, 50, 60], "text": "Bow"},
        {"category": "Caption", "bbox": [10, 70, 110, 85], "text": "Figure 1. System diagram."},
        {"category": "Caption", "bbox": [10, 85, 110, 98], "text": "**Table 1. Example data.**"},
    ]

    markdown = layout_cells_to_markdown_with_crops(
        image=image,
        cells=cells,
        imgs_dir=tmp_path / "imgs",
        relative_img_dir="imgs",
    )

    assert markdown.split("\n\n") == [
        "![Picture page 1 cell 1](imgs/page_001_img_001.png)",
        "_Caption: Bow_",
        "_Figure 1. System diagram._",
        "_**Table 1. Example data.**_",
    ]


def test_layout_cells_to_markdown_formats_detected_headers_as_markdown(tmp_path):
    image = Image.new("RGB", (200, 160), "white")
    cells = [
        {"category": "Page-header", "bbox": [0, 0, 200, 20], "text": "Issue 2"},
        {"category": "Title", "bbox": [0, 22, 200, 45], "text": "Main Article"},
        {"category": "Section-header", "bbox": [0, 50, 200, 75], "text": "Section\nTitle"},
        {"category": "Subsection-header", "bbox": [0, 80, 200, 100], "text": "Smaller part"},
        {"category": "Section-header", "heading_level": "h4", "bbox": [0, 102, 200, 120], "text": "Explicit level"},
        {"category": "Text", "bbox": [0, 125, 200, 150], "text": "Body text"},
    ]

    markdown = layout_cells_to_markdown_with_crops(
        image=image,
        cells=cells,
        imgs_dir=tmp_path / "imgs",
        relative_img_dir="imgs",
    )

    assert markdown.split("\n\n") == [
        "Issue 2",
        "# Main Article",
        "## Section Title",
        "### Smaller part",
        "#### Explicit level",
        "Body text",
    ]


def test_layout_cells_to_markdown_preserves_model_markdown_heading_depth(tmp_path):
    image = Image.new("RGB", (200, 120), "white")
    cells = [
        {"category": "Section-header", "bbox": [0, 0, 200, 30], "text": "## Description"},
        {"category": "Section-header", "bbox": [0, 40, 200, 70], "text": "### 2016–2019: From order to delivery"},
    ]

    markdown = layout_cells_to_markdown_with_crops(
        image=image,
        cells=cells,
        imgs_dir=tmp_path / "imgs",
        relative_img_dir="imgs",
    )

    assert markdown.split("\n\n") == [
        "## Description",
        "### 2016–2019: From order to delivery",
    ]


def test_layout_cells_to_markdown_infers_web_article_heading_hierarchy_when_enabled(tmp_path):
    image = Image.new("RGB", (1000, 1600), "white")
    cells = [
        {"category": "Page-header", "bbox": [20, 30, 250, 90], "text": "MV *Hondius*"},
        {"category": "Page-header", "bbox": [820, 35, 950, 75], "text": "Tools"},
        {"category": "Text", "bbox": [20, 130, 500, 165], "text": "From Wikipedia, the free encyclopedia"},
        {
            "category": "Text",
            "bbox": [20, 210, 850, 330],
            "text": "MV *Hondius* is a Dutch expedition cruise ship owned by Oceanwide Expeditions.",
        },
        {"category": "Section-header", "bbox": [20, 430, 250, 488], "text": "Description"},
        {"category": "Text", "bbox": [20, 520, 850, 650], "text": "Hondius is 107.6 metres long overall."},
        {"category": "Section-header", "bbox": [20, 760, 180, 818], "text": "History"},
        {"category": "Section-header", "bbox": [20, 860, 520, 908], "text": "2016–2019: From order to delivery"},
    ]

    markdown = layout_cells_to_markdown_with_crops(
        image=image,
        cells=cells,
        imgs_dir=tmp_path / "imgs",
        relative_img_dir="imgs",
        no_page_header_footer=True,
        infer_heading_hierarchy=True,
    )

    assert markdown.split("\n\n") == [
        "# MV *Hondius*",
        "From Wikipedia, the free encyclopedia",
        "MV *Hondius* is a Dutch expedition cruise ship owned by Oceanwide Expeditions.",
        "## Description",
        "Hondius is 107.6 metres long overall.",
        "## History",
        "### 2016–2019: From order to delivery",
    ]


def test_layout_cells_to_markdown_infers_book_title_from_first_section_header(tmp_path):
    image = Image.new("RGB", (1534, 2134), "white")
    cells = [
        {"category": "Page-header", "bbox": [1103, 447, 1285, 487], "text": "CHAPTER 17"},
        {
            "category": "Section-header",
            "bbox": [615, 529, 1285, 650],
            "text": "Design of Treatment Trials for\nDisorders of Gut-Brain Interaction",
        },
        {
            "category": "Text",
            "bbox": [788, 720, 1285, 1118],
            "text": "Gregory S. Sayuk, MD, MPH, *Chair*\nAlexander C. Ford, MD, MBChB, FRCP (Edin), *Co-Chair*",
        },
    ]

    markdown = layout_cells_to_markdown_with_crops(
        image=image,
        cells=cells,
        imgs_dir=tmp_path / "imgs",
        relative_img_dir="imgs",
        no_page_header_footer=True,
        infer_heading_hierarchy=True,
    )

    assert markdown.split("\n\n") == [
        "# Design of Treatment Trials for Disorders of Gut-Brain Interaction",
        "Gregory S. Sayuk, MD, MPH, *Chair*\nAlexander C. Ford, MD, MBChB, FRCP (Edin), *Co-Chair*",
    ]


def test_layout_cells_to_markdown_infers_numbered_section_depth_when_enabled(tmp_path):
    image = Image.new("RGB", (1200, 1600), "white")
    cells = [
        {"category": "Title", "bbox": [20, 20, 600, 90], "text": "Documentation Template"},
        {"category": "Section-header", "bbox": [20, 160, 240, 195], "text": "1. RHO CLASS"},
        {"category": "Section-header", "bbox": [20, 250, 260, 285], "text": "1.1. License"},
        {"category": "Section-header", "bbox": [20, 340, 300, 375], "text": "6.2.1. Headers"},
    ]

    markdown = layout_cells_to_markdown_with_crops(
        image=image,
        cells=cells,
        imgs_dir=tmp_path / "imgs",
        relative_img_dir="imgs",
        infer_heading_hierarchy=True,
    )

    assert markdown.split("\n\n") == [
        "# Documentation Template",
        "## 1. RHO CLASS",
        "### 1.1. License",
        "#### 6.2.1. Headers",
    ]


def test_layout_cells_to_markdown_does_not_promote_unmatched_page_header(tmp_path):
    image = Image.new("RGB", (1000, 1400), "white")
    cells = [
        {"category": "Page-header", "bbox": [20, 20, 300, 70], "text": "Journal Issue 2"},
        {"category": "Section-header", "bbox": [20, 180, 200, 230], "text": "Abstract"},
        {
            "category": "Text",
            "bbox": [20, 260, 900, 400],
            "text": "This article starts with body text that does not repeat the running header.",
        },
    ]

    markdown = layout_cells_to_markdown_with_crops(
        image=image,
        cells=cells,
        imgs_dir=tmp_path / "imgs",
        relative_img_dir="imgs",
        infer_heading_hierarchy=True,
    )

    assert markdown.split("\n\n") == [
        "Journal Issue 2",
        "## Abstract",
        "This article starts with body text that does not repeat the running header.",
    ]


def test_layout_cells_to_markdown_demotes_later_page_model_h1_when_infer_enabled(tmp_path):
    image = Image.new("RGB", (1000, 1400), "white")
    cells = [
        {"category": "Page-header", "bbox": [20, 20, 600, 60], "text": "Book title 1423"},
        {"category": "Section-header", "bbox": [20, 160, 260, 210], "text": "# Introduction"},
        {"category": "Text", "bbox": [20, 250, 900, 400], "text": "Body text continues on a later PDF page."},
    ]

    markdown = layout_cells_to_markdown_with_crops(
        image=image,
        cells=cells,
        imgs_dir=tmp_path / "imgs",
        relative_img_dir="imgs",
        page_no=1,
        no_page_header_footer=True,
        infer_heading_hierarchy=True,
    )

    assert markdown.split("\n\n") == [
        "## Introduction",
        "Body text continues on a later PDF page.",
    ]


def test_build_viewer_html_contains_source_markdown_preview_and_json():
    html = build_viewer_html(
        title="sample",
        markdown_html='<p>Hello <img src="sample_imgs/page_001_img_001.png"></p>',
        raw_markdown="Hello\n\n![x](sample_imgs/page_001_img_001.png)",
        json_text=json.dumps([{"category": "Text", "text": "Hello"}], ensure_ascii=False),
        source_image_paths=["sample_pages/page_001.png"],
        layout_image_paths=["sample_layout/page_001.jpg"],
    )

    assert "File Preview" in html
    assert "Result Display" in html
    assert "Markdown Render Preview" in html
    assert "Formatted JSON" in html
    assert "sample_layout/page_001.jpg" in html
    assert "sample_imgs/page_001_img_001.png" in html
    assert "Hello" in html


def test_build_viewer_html_keeps_result_display_scrollable_and_compact():
    html = build_viewer_html(
        title="sample",
        markdown_html="<p>Hello</p>",
        raw_markdown="Hello",
        json_text="[]",
        source_image_paths=["sample_pages/page_001.png"],
        layout_image_paths=["sample_layout/page_001.jpg"],
    )

    assert ".preview-scroll, .result-scroll { max-height: calc(100vh - 160px); overflow: auto; }" in html
    assert '<div class="card result-scroll">' in html
    assert ".markdown-body { font-size: 16px; line-height: 1.6;" in html


class FakeRunner:
    def __init__(self):
        self.calls = []

    def infer(self, image_path, prompt):
        self.calls.append((Path(image_path), prompt))
        return json.dumps(
            [
                {"category": "Text", "bbox": [0, 0, 100, 20], "text": "Recognized text"},
                {"category": "Picture", "bbox": [10, 20, 50, 60], "text": ""},
            ],
            ensure_ascii=False,
        )


class FakeOpenAIResponse:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


class FakeOpenAICompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeOpenAIResponse("api response")


class FakeOpenAIClient:
    def __init__(self):
        self.completions = FakeOpenAICompletions()
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_omlx_api_runner_sends_direct_prompt_and_explicit_generation_params(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (64, 64), "white").save(image_path)
    client = FakeOpenAIClient()
    runner = OmlxApiRunner(
        client=client,
        model_name="dots.mocr-bf16",
        max_completion_tokens=123,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=0,
        repetition_penalty=1.05,
    )

    response = runner.infer(image_path, "layout prompt")

    assert response == "api response"
    call = client.completions.calls[0]
    assert call["model"] == "dots.mocr-bf16"
    assert call["max_completion_tokens"] == 123
    assert call["temperature"] == 0.0
    assert call["top_p"] == 1.0
    assert call["frequency_penalty"] == 0.0
    assert call["presence_penalty"] == 0.0
    assert call["seed"] == 0
    assert call["extra_body"] == {"repetition_penalty": 1.05}
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "layout prompt"}
    assert "<|img|><|imgpad|><|endofimg|>" not in content[1]["text"]


class ThreadRecordingRunner:
    def __init__(self):
        self.thread_ids = set()
        self.lock = threading.Lock()

    def infer(self, image_path, prompt):
        with self.lock:
            self.thread_ids.add(threading.get_ident())
        time.sleep(0.05)
        return json.dumps(
            [{"category": "Text", "bbox": [0, 0, 100, 20], "text": Path(image_path).stem}],
            ensure_ascii=False,
        )


class SlowFirstPageRunner:
    def infer(self, image_path, prompt):
        if Path(image_path).stem == "page_001":
            time.sleep(0.08)
        return json.dumps(
            [{"category": "Text", "bbox": [0, 0, 100, 20], "text": Path(image_path).stem}],
            ensure_ascii=False,
        )


class ImageRecordingRunner:
    def __init__(self):
        self.images = []

    def infer_image(self, image, prompt):
        self.images.append(image)
        return json.dumps(
            [{"category": "Text", "bbox": [0, 0, 100, 20], "text": "from image"}],
            ensure_ascii=False,
        )


def _write_pdf(path: Path, page_count: int = 2) -> None:
    doc = fitz.open()
    for index in range(page_count):
        page = doc.new_page(width=120, height=80)
        page.insert_text((20, 40), f"page {index + 1}")
    doc.save(path)
    doc.close()


def test_benchmark_summary_counts_cells_and_categories(tmp_path):
    timings_path = tmp_path / "sample_timings.json"
    json_path = tmp_path / "sample.json"
    timings_path.write_text(
        json.dumps(
            {
                "total_seconds": 12.5,
                "worker_count": 1,
                "settings": {"max_pixels": 1000000},
                "pages": [
                    {"inference_seconds": 3.0, "inference_width": 840, "inference_height": 1176},
                    {"inference_seconds": 4.0, "inference_width": 840, "inference_height": 1176},
                ],
            }
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            [
                {"cells": [{"category": "Text"}, {"category": "Title"}]},
                {"cells": [{"category": "Text"}]},
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_run(timings_path=timings_path, json_path=json_path, threads=1, max_pixels=1000000)

    assert summary["total_seconds"] == 12.5
    assert summary["cell_counts"] == [2, 1]
    assert summary["category_counts"] == {"Text": 2, "Title": 1}
    assert summary["inference_seconds_sum"] == 7.0
    assert summary["input_dimensions"] == ["840x1176"]


def test_run_pipeline_can_process_multiple_pages_with_requested_threads(tmp_path):
    input_path = tmp_path / "input.pdf"
    _write_pdf(input_path)
    runner = ThreadRecordingRunner()

    result = run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "out",
        output_name="sample",
        runner=runner,
        launch_viewer=False,
        num_thread=2,
    )

    assert len(result.page_outputs) == 2
    assert result.page_outputs[0].page_no == 0
    assert result.page_outputs[1].page_no == 1
    assert len(runner.thread_ids) == 2


def test_run_pipeline_preserves_page_order_when_parallel_pages_finish_out_of_order(tmp_path):
    input_path = tmp_path / "input.pdf"
    _write_pdf(input_path)

    result = run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "out",
        output_name="sample",
        runner=SlowFirstPageRunner(),
        launch_viewer=False,
        num_thread=2,
    )

    assert [page.page_no for page in result.page_outputs] == [0, 1]


def test_run_pipeline_server_safe_mode_limits_omlx_worker_count(tmp_path):
    input_path = tmp_path / "input.pdf"
    _write_pdf(input_path, page_count=4)

    result = run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "out",
        output_name="sample",
        runner=ThreadRecordingRunner(),
        launch_viewer=False,
        backend="omlx-api",
        num_thread=4,
        parallel_mode="server-safe",
    )
    timings = json.loads(result.timings_path.read_text(encoding="utf-8"))

    assert timings["requested_worker_count"] == 4
    assert timings["worker_count"] == 1


def test_run_pipeline_fixed_mode_honors_requested_omlx_worker_count(tmp_path):
    input_path = tmp_path / "input.pdf"
    _write_pdf(input_path, page_count=4)

    result = run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "out",
        output_name="sample",
        runner=ThreadRecordingRunner(),
        launch_viewer=False,
        backend="omlx-api",
        num_thread=4,
        parallel_mode="fixed",
    )
    timings = json.loads(result.timings_path.read_text(encoding="utf-8"))

    assert timings["requested_worker_count"] == 4
    assert timings["worker_count"] == 4


def test_resolve_max_pixels_preserves_explicit_value_over_performance_preset():
    args = parse_args(
        [
            "input.pdf",
            "--performance-preset",
            "fast",
            "--max-pixels",
            "2000000",
        ]
    )

    assert resolve_max_pixels(args.max_pixels, args.performance_preset) == 2_000_000


def test_run_pipeline_uses_in_memory_image_inference_when_runner_supports_it(tmp_path):
    input_path = tmp_path / "input.png"
    Image.new("RGB", (100, 80), "white").save(input_path)
    runner = ImageRecordingRunner()

    result = run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "out",
        output_name="sample",
        runner=runner,
        launch_viewer=False,
    )

    assert len(runner.images) == 1
    assert isinstance(runner.images[0], Image.Image)
    assert result.page_outputs[0].markdown == "from image"


def test_run_pipeline_with_fake_runner_writes_requested_file_structure(tmp_path):
    input_path = tmp_path / "input.png"
    Image.new("RGB", (100, 80), "white").save(input_path)

    result = run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "out",
        output_name="sample",
        runner=FakeRunner(),
        launch_viewer=False,
    )

    assert result.markdown_path == tmp_path / "out" / "sample.md"
    assert result.markdown_path.exists()
    assert (tmp_path / "out" / "sample_imgs" / "page_001_img_002.png").exists()
    assert (tmp_path / "out" / "sample_viewer.html").exists()
    assert (tmp_path / "out" / "sample.json").exists()
    assert (tmp_path / "out" / "sample_timings.json").exists()
    assert "sample_imgs/page_001_img_002.png" in result.markdown_path.read_text(encoding="utf-8")
    assert result.viewer_url is None


def test_run_pipeline_writes_page_timing_details(tmp_path):
    input_path = tmp_path / "input.png"
    Image.new("RGB", (100, 80), "white").save(input_path)

    result = run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "out",
        output_name="sample",
        runner=FakeRunner(),
        launch_viewer=False,
        backend="omlx-api",
        api_model="dots.mocr-bf16",
        max_tokens=512,
        num_thread=3,
    )

    assert result.timings_path == tmp_path / "out" / "sample_timings.json"
    timings = json.loads(result.timings_path.read_text(encoding="utf-8"))

    assert timings["output_name"] == "sample"
    assert timings["backend"] == "omlx-api"
    assert timings["model"] == "dots.mocr-bf16"
    assert timings["requested_num_thread"] == 3
    assert timings["worker_count"] == 1
    assert timings["page_count"] == 1
    assert timings["total_seconds"] >= 0

    page = timings["pages"][0]
    assert page["page_no"] == 0
    assert page["display_page_no"] == 1
    assert page["source_image_path"] == "sample_pages/page_001.png"
    assert page["inference_image_path"] == "sample_mlx_input/page_001.png"
    assert page["source_width"] == 100
    assert page["source_height"] == 80
    assert page["inference_width"] > 0
    assert page["inference_height"] > 0
    assert page["inference_seconds"] >= 0
    assert page["postprocess_seconds"] >= 0
    assert page["total_seconds"] >= page["inference_seconds"]
    assert page["raw_chars"] > 0
    assert page["raw_bytes"] >= page["raw_chars"]
    assert page["filtered"] is False


def test_run_pipeline_viewer_links_layout_boxes_to_rendered_output(tmp_path):
    input_path = tmp_path / "input.png"
    Image.new("RGB", (100, 80), "white").save(input_path)

    result = run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "out",
        output_name="sample",
        runner=FakeRunner(),
        launch_viewer=False,
    )

    viewer_html = result.viewer_path.read_text(encoding="utf-8")

    assert 'class="layout-hotspot' in viewer_html
    assert 'data-cell-id="p1-c1"' in viewer_html
    assert 'data-cell-id="p1-c2"' in viewer_html
    assert 'id="result-p1-c1"' in viewer_html
    assert 'id="result-p1-c2"' in viewer_html
    assert "activateMappedCell" in viewer_html
    assert ".result-block.flash" in viewer_html
    assert ".layout-hotspot.active-source" in viewer_html
