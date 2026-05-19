"""Single-command MLX pipeline for dots.mocr document parsing.

This module intentionally avoids importing the heavier vLLM/HF parser path at
module import time. The MLX model is loaded only when the command actually runs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import html
import json
import os
import re
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Protocol
from urllib.parse import quote

from PIL import Image

from dots_mocr.utils.consts import MAX_PIXELS, MIN_PIXELS, image_extensions
from dots_mocr.utils.format_transformer import clean_text, get_formula_in_markdown
from dots_mocr.utils.image_utils import PILimage_to_base64, fetch_image, get_image_by_fitz_doc
from dots_mocr.utils.layout_utils import draw_layout_on_image, post_process_output
from dots_mocr.utils.prompts import dict_promptmode_to_prompt

DEFAULT_MLX_MODEL = os.environ.get("DOTS_MOCR_MLX_MODEL", "mlx-community/dots.mocr-4bit")
DEFAULT_OMLX_API_BASE_URL = os.environ.get("DOTS_MOCR_OMLX_API_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_OMLX_API_KEY = os.environ.get("DOTS_MOCR_OMLX_API_KEY", "1220")
DEFAULT_OMLX_MODEL = os.environ.get("DOTS_MOCR_OMLX_MODEL", "dots.mocr-bf16")
PERFORMANCE_PRESET_MAX_PIXELS = {
    "quality": None,
    "balanced": 1_600_000,
    "fast": 1_000_000,
}
LAYOUT_PROMPTS = {"prompt_layout_all_en", "prompt_layout_only_en", "prompt_grounding_ocr", "prompt_web_parsing"}
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+\S")
NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[.)])?\s+\S")
CAPTION_LABEL_RE = re.compile(r"^(fig\.?|figure|table|code|listing|equation|eq\.?)\s+\d+\b", re.IGNORECASE)
COMMON_SECTION_LABELS = {
    "abstract",
    "acknowledgement",
    "acknowledgements",
    "bibliography",
    "contents",
    "introduction",
    "keywords",
    "materials and methods",
    "methods",
    "references",
    "table of contents",
}
UI_HEADER_TEXTS = {
    "article",
    "edit",
    "read",
    "talk",
    "tools",
    "view history",
}


class InferenceRunner(Protocol):
    def infer(self, image_path: str | Path, prompt: str) -> str:
        """Run OCR/layout inference on one image and return the raw model text."""


@dataclass
class PageInput:
    page_no: int
    image: Image.Image
    source_image_path: Path


@dataclass
class PageOutput:
    page_no: int
    source_image_path: Path
    inference_image_path: Path
    layout_image_path: Path | None
    markdown: str
    markdown_fragments: list[MarkdownFragment]
    cells: Any
    filtered: bool
    raw_response: str
    page_width: int
    page_height: int
    timing: dict[str, Any]


@dataclass
class PipelineResult:
    markdown_path: Path
    images_dir: Path
    json_path: Path
    timings_path: Path
    viewer_path: Path
    page_outputs: list[PageOutput]
    viewer_url: str | None = None


@dataclass
class MarkdownFragment:
    page_no: int
    cell_index: int
    category: str
    markdown: str
    bbox: tuple[int, int, int, int] | None = None


class MlxDotsRunner:
    """Small adapter around mlx-vlm for mlx-community dots.mocr checkpoints."""

    def __init__(
        self,
        model_id: str = DEFAULT_MLX_MODEL,
        *,
        max_tokens: int = 24000,
        temperature: float = 0.0,
        trust_remote_code: bool = True,
        revision: str | None = None,
        prefill_step_size: int | None = None,
        skip_special_tokens: bool = True,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prefill_step_size = prefill_step_size
        self.skip_special_tokens = skip_special_tokens

        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        self._generate = generate
        self._apply_chat_template = apply_chat_template
        self.model, self.processor = load(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def infer(self, image_path: str | Path, prompt: str) -> str:
        image_path = str(image_path)
        prompt_text = self._apply_chat_template(
            self.processor,
            self.model.config,
            prompt,
            num_images=1,
        )
        kwargs: dict[str, Any] = {
            "image": [image_path],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "verbose": False,
            "skip_special_tokens": self.skip_special_tokens,
        }
        if self.prefill_step_size is not None:
            kwargs["prefill_step_size"] = self.prefill_step_size
        result = self._generate(self.model, self.processor, prompt_text, **kwargs)
        return result.text


class OmlxApiRunner:
    """OpenAI-compatible API runner for a local oMLX dots.mocr server."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OMLX_API_BASE_URL,
        api_key: str = DEFAULT_OMLX_API_KEY,
        model_name: str = DEFAULT_OMLX_MODEL,
        max_completion_tokens: int = 24000,
        temperature: float = 0.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: int | None = 0,
        repetition_penalty: float | None = 1.05,
        timeout: float = 300.0,
        max_retries: int = 1,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.repetition_penalty = repetition_penalty
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )
        self.client = client

    def infer(self, image_path: str | Path, prompt: str) -> str:
        with Image.open(image_path) as image:
            return self.infer_image(image, prompt)

    def infer_image(self, image: Image.Image, prompt: str) -> str:
        image_url = PILimage_to_base64(_as_rgb(image))
        return self._create_completion(image_url=image_url, prompt=prompt)

    def _create_completion(self, *, image_url: str, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.repetition_penalty is not None:
            kwargs["extra_body"] = {"repetition_penalty": self.repetition_penalty}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def _as_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def _safe_output_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def resolve_max_pixels(explicit_max_pixels: int | None, performance_preset: str) -> int | None:
    if explicit_max_pixels is not None:
        return explicit_max_pixels
    try:
        return PERFORMANCE_PRESET_MAX_PIXELS[performance_preset]
    except KeyError as exc:
        expected = ", ".join(sorted(PERFORMANCE_PRESET_MAX_PIXELS))
        raise ValueError(f"Unsupported performance preset {performance_preset!r}; expected one of {expected}") from exc


def _relative(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path, base_dir)).as_posix()


def _clamp_bbox(bbox: Iterable[Any], width: int, height: int) -> tuple[int, int, int, int]:
    values = [int(round(float(coord))) for coord in bbox]
    if len(values) != 4:
        raise ValueError(f"Expected bbox with 4 coordinates, got {bbox!r}")
    x1, y1, x2, y2 = values
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _level_from_cell_metadata(cell: dict[str, Any]) -> int | None:
    """Return an explicit Markdown heading level if the model supplied one."""

    for key in ("markdown_level", "heading_level", "header_level", "level"):
        value = cell.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip().lower().removeprefix("h")
        try:
            level = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= level <= 6:
            return level
    return None


def _level_from_markdown_heading_text(cell: dict[str, Any]) -> int | None:
    """Return a Markdown heading level already emitted in the cell text.

    Some dots.mocr checkpoints encode hierarchy in the text itself, e.g.
    ``{"category": "Section-header", "text": "### Details"}``, while still
    using only the broad ``Section-header`` category. Treat that prefix as an
    explicit model signal before applying category fallbacks.
    """

    text = str(cell.get("text", "") or "")
    match = MARKDOWN_HEADING_RE.match(text)
    if match:
        return len(match.group(1))
    return None


def _heading_level_for_cell(cell: dict[str, Any], inferred_level: int | None = None) -> int | None:
    """Map dots.mocr layout categories to conservative Markdown heading levels.

    dots.mocr reliably labels many headings as layout categories, but the current
    JSON does not always encode a full semantic outline. Prefer any explicit
    model-supplied level, optional context inference, then stable category names.
    """

    explicit_level = _level_from_cell_metadata(cell)
    if explicit_level is not None:
        return explicit_level

    markdown_level = _level_from_markdown_heading_text(cell)
    if markdown_level is not None:
        return markdown_level

    if inferred_level is not None:
        return inferred_level

    category = str(cell.get("category", "")).strip().lower().replace("_", "-")
    if not category or category in {"page-header", "page-footer"}:
        return None
    if category in {"title", "document-title", "doc-title", "page-title"}:
        return 1
    if "subsubsection" in category or "sub-subsection" in category:
        return 4
    if "subsection" in category or "sub-section" in category:
        return 3
    if category in {"section-header", "section-title"} or "section-header" in category:
        return 2
    if category in {"header", "heading"}:
        return 2
    return None


def _format_heading_text(text: str) -> str:
    """Collapse OCR line breaks and strip pre-existing Markdown heading marks."""

    text = clean_text(text)
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    while text.startswith("#"):
        text = text[1:].strip()
    return text


def _format_caption_text(text: str) -> str:
    text = clean_text(text)
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        return ""
    label_probe = re.sub(r"^[*_`~\s]+", "", text)
    if CAPTION_LABEL_RE.match(label_probe):
        return f"_{text}_"
    return f"_Caption: {text}_"


def _cell_category(cell: dict[str, Any]) -> str:
    return str(cell.get("category", "") or "").strip().lower().replace("_", "-")


def _cell_bbox(cell: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = cell.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def _fragment_bbox(cell: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    bbox = cell.get("bbox")
    if bbox is None:
        return None
    try:
        x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _cell_height(cell: dict[str, Any]) -> float | None:
    bbox = _cell_bbox(cell)
    if bbox is None:
        return None
    return max(0.0, bbox[3] - bbox[1])


def _normalized_semantic_text(text: str) -> str:
    text = _format_heading_text(text).lower()
    text = re.sub(r"[*_`~\[\]().,:;!?]+", " ", text)
    return " ".join(text.split())


def _is_common_section_label(text: str) -> bool:
    normalized = _normalized_semantic_text(text)
    return normalized in COMMON_SECTION_LABELS


def _is_obvious_ui_header(text: str) -> bool:
    normalized = _normalized_semantic_text(text)
    return normalized in UI_HEADER_TEXTS or normalized.endswith(" languages")


def _explicit_or_category_h1(cell: dict[str, Any]) -> bool:
    metadata_level = _level_from_cell_metadata(cell)
    if metadata_level == 1:
        return True
    markdown_level = _level_from_markdown_heading_text(cell)
    if markdown_level == 1:
        return True
    return _cell_category(cell) in {"title", "document-title", "doc-title", "page-title"}


def _has_existing_h1(cells: list[dict[str, Any]]) -> bool:
    return any(_explicit_or_category_h1(cell) for cell in cells)


def _first_substantial_text_after(cells: list[dict[str, Any]], start_index: int) -> str:
    for cell in cells[start_index + 1 :]:
        if _cell_category(cell) != "text":
            continue
        text = str(cell.get("text", "") or "")
        normalized = _normalized_semantic_text(text)
        if len(normalized) >= 40 and normalized not in {"from wikipedia the free encyclopedia"}:
            return normalized
    return ""


def _text_starts_with_heading(text: str, heading: str) -> bool:
    normalized_heading = _normalized_semantic_text(heading)
    if len(normalized_heading) < 4:
        return False
    return text.startswith(normalized_heading)


def _numbered_heading_level(text: str) -> int | None:
    match = NUMBERED_HEADING_RE.match(_format_heading_text(text))
    if not match:
        return None
    depth = match.group(1).count(".") + 1
    return min(6, depth + 1)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2


def _infer_title_heading_index(
    *,
    cells: list[dict[str, Any]],
    width: int,
    height: int,
    page_no: int,
) -> int | None:
    if page_no != 0 or _has_existing_h1(cells):
        return None

    for index, cell in enumerate(cells):
        if _cell_category(cell) != "page-header":
            continue
        bbox = _cell_bbox(cell)
        text = str(cell.get("text", "") or "")
        if bbox is None or not text or _is_obvious_ui_header(text):
            continue
        x1, y1, _x2, _y2 = bbox
        if x1 > width * 0.35 or y1 > height * 0.20:
            continue
        if _text_starts_with_heading(_first_substantial_text_after(cells, index), text):
            return index

    for index, cell in enumerate(cells):
        if _cell_category(cell) != "section-header":
            continue
        if _level_from_cell_metadata(cell) is not None or _level_from_markdown_heading_text(cell) is not None:
            continue
        bbox = _cell_bbox(cell)
        text = str(cell.get("text", "") or "")
        if bbox is None or not text or _is_common_section_label(text):
            continue
        _x1, y1, _x2, _y2 = bbox
        if y1 > height * 0.35:
            continue
        formatted = _format_heading_text(text)
        if len(formatted) >= 18 or "\n" in text:
            return index
    return None


def _infer_heading_levels(
    *,
    cells: list[dict[str, Any]],
    width: int,
    height: int,
    page_no: int,
) -> dict[int, int]:
    inferred: dict[int, int] = {}
    title_index = _infer_title_heading_index(cells=cells, width=width, height=height, page_no=page_no)
    if title_index is not None:
        inferred[title_index] = 1

    section_heights = [
        cell_height
        for index, cell in enumerate(cells)
        if index != title_index
        and _cell_category(cell) == "section-header"
        and _level_from_cell_metadata(cell) is None
        and _level_from_markdown_heading_text(cell) is None
        and _numbered_heading_level(str(cell.get("text", "") or "")) is None
        for cell_height in [_cell_height(cell)]
        if cell_height is not None and cell_height > 0
    ]
    median_section_height = _median(section_heights)

    for index, cell in enumerate(cells):
        if index == title_index or _cell_category(cell) != "section-header":
            continue
        if _level_from_cell_metadata(cell) is not None or _level_from_markdown_heading_text(cell) is not None:
            continue

        numbered_level = _numbered_heading_level(str(cell.get("text", "") or ""))
        if numbered_level is not None:
            inferred[index] = numbered_level
            continue

        cell_height = _cell_height(cell)
        if (
            median_section_height is not None
            and cell_height is not None
            and cell_height < median_section_height * 0.9
        ):
            inferred[index] = 3

    return inferred


def _cell_id(page_no: int, cell_index: int) -> str:
    return f"p{page_no + 1}-c{cell_index}"


def layout_cells_to_markdown_fragments(
    *,
    image: Image.Image,
    cells: list[dict[str, Any]],
    imgs_dir: Path,
    relative_img_dir: str,
    page_no: int = 0,
    text_key: str = "text",
    no_page_header_footer: bool = False,
    infer_heading_hierarchy: bool = False,
) -> list[MarkdownFragment]:
    """Convert dots.mocr layout cells to mapped markdown fragments.

    The upstream helper embeds Picture cells as base64 data URIs. For a durable
    document export we instead save each crop under ``imgs_dir`` and link it
    from markdown with a relative path.
    """

    imgs_dir.mkdir(parents=True, exist_ok=True)
    fragments: list[MarkdownFragment] = []
    width, height = image.size
    inferred_heading_levels = (
        _infer_heading_levels(cells=cells, width=width, height=height, page_no=page_no)
        if infer_heading_hierarchy
        else {}
    )

    for index, cell in enumerate(cells, start=1):
        category = cell.get("category", "")
        inferred_heading_level = inferred_heading_levels.get(index - 1)
        if no_page_header_footer and category in {"Page-header", "Page-footer"} and inferred_heading_level is None:
            continue

        if category == "Picture":
            bbox = cell.get("bbox", [0, 0, width, height])
            x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_name = f"page_{page_no + 1:03d}_img_{index:03d}.png"
            crop_path = imgs_dir / crop_name
            image.crop((x1, y1, x2, y2)).save(crop_path)
            fragments.append(
                MarkdownFragment(
                    page_no=page_no,
                    cell_index=index,
                    category=str(category or ""),
                    markdown=f"![Picture page {page_no + 1} cell {index}]({relative_img_dir}/{crop_name})",
                    bbox=(x1, y1, x2, y2),
                )
            )
            continue

        text = str(cell.get(text_key, "") or "")
        fragment_bbox = _fragment_bbox(cell, width, height)
        if _cell_category(cell) == "caption":
            text = _format_caption_text(text)
        elif category == "Formula":
            text = get_formula_in_markdown(text)
        else:
            heading_level = _heading_level_for_cell(cell, inferred_heading_level)
            if heading_level is not None:
                if infer_heading_hierarchy and page_no > 0 and heading_level == 1:
                    heading_level = 2
                text = _format_heading_text(text)
                if text:
                    fragments.append(
                        MarkdownFragment(
                            page_no=page_no,
                            cell_index=index,
                            category=str(category or ""),
                            markdown=f"{'#' * heading_level} {text}",
                            bbox=fragment_bbox,
                        )
                    )
                continue
            text = clean_text(text)
        if text:
            fragments.append(
                MarkdownFragment(
                    page_no=page_no,
                    cell_index=index,
                    category=str(category or ""),
                    markdown=text,
                    bbox=fragment_bbox,
                )
            )

    return fragments


def join_markdown_fragments(fragments: list[MarkdownFragment]) -> str:
    return "\n\n".join(fragment.markdown for fragment in fragments if fragment.markdown)


def layout_cells_to_markdown_with_crops(
    *,
    image: Image.Image,
    cells: list[dict[str, Any]],
    imgs_dir: Path,
    relative_img_dir: str,
    page_no: int = 0,
    text_key: str = "text",
    no_page_header_footer: bool = False,
    infer_heading_hierarchy: bool = False,
) -> str:
    """Convert dots.mocr layout cells to markdown with file-backed crops."""

    return join_markdown_fragments(
        layout_cells_to_markdown_fragments(
            image=image,
            cells=cells,
            imgs_dir=imgs_dir,
            relative_img_dir=relative_img_dir,
            page_no=page_no,
            text_key=text_key,
            no_page_header_footer=no_page_header_footer,
            infer_heading_hierarchy=infer_heading_hierarchy,
        )
    )


def _render_pdf_pages(input_path: Path, pages_dir: Path, dpi: int) -> list[PageInput]:
    import fitz

    doc = fitz.open(str(input_path))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[PageInput] = []
    try:
        for page_index in range(len(doc)):
            pix = doc[page_index].get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            source_path = pages_dir / f"page_{page_index + 1:03d}.png"
            image.save(source_path)
            pages.append(PageInput(page_index, image, source_path))
    finally:
        doc.close()
    return pages


def load_input_pages(
    input_path: str | Path,
    *,
    pages_dir: Path,
    dpi: int = 200,
    fitz_preprocess: bool = False,
) -> list[PageInput]:
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    pages_dir.mkdir(parents=True, exist_ok=True)
    ext = input_path.suffix.lower()
    if ext == ".pdf":
        return _render_pdf_pages(input_path, pages_dir, dpi)
    if ext not in image_extensions:
        supported = ", ".join(sorted(image_extensions | {".pdf"}))
        raise ValueError(f"Unsupported input extension {ext!r}; supported: {supported}")

    image = _as_rgb(Image.open(input_path))
    if fitz_preprocess:
        image = _as_rgb(get_image_by_fitz_doc(image, target_dpi=dpi))
    source_path = pages_dir / "page_001.png"
    image.save(source_path)
    return [PageInput(0, image, source_path)]


def process_page(
    *,
    page: PageInput,
    runner: InferenceRunner,
    prompt: str,
    prompt_mode: str,
    output_name: str,
    output_dir: Path,
    imgs_dir: Path,
    layout_dir: Path,
    inference_dir: Path,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    no_page_header_footer: bool = False,
    infer_heading_hierarchy: bool = False,
) -> PageOutput:
    page_started_at = _utc_timestamp()
    page_start = time.perf_counter()

    preprocess_start = time.perf_counter()
    effective_min_pixels = min_pixels or MIN_PIXELS
    effective_max_pixels = max_pixels or MAX_PIXELS
    inference_image = fetch_image(
        page.image,
        min_pixels=effective_min_pixels,
        max_pixels=effective_max_pixels,
    )
    inference_image_path = inference_dir / f"page_{page.page_no + 1:03d}.png"
    inference_image.save(inference_image_path)
    preprocess_seconds = time.perf_counter() - preprocess_start

    inference_started_at = _utc_timestamp()
    inference_start = time.perf_counter()
    infer_image = getattr(runner, "infer_image", None)
    if infer_image is not None:
        raw_response = infer_image(inference_image, prompt)
    else:
        raw_response = runner.infer(inference_image_path, prompt)
    inference_seconds = time.perf_counter() - inference_start
    inference_completed_at = _utc_timestamp()
    raw_path = output_dir / f"{output_name}_page_{page.page_no + 1:03d}_raw.txt"
    raw_path.write_text(raw_response, encoding="utf-8")

    postprocess_start = time.perf_counter()
    layout_image_path: Path | None = None
    cells: Any = raw_response
    filtered = False
    markdown_fragments: list[MarkdownFragment] = []

    if prompt_mode in LAYOUT_PROMPTS:
        cells, filtered = post_process_output(
            raw_response,
            prompt_mode,
            page.image,
            inference_image,
            min_pixels=effective_min_pixels,
            max_pixels=effective_max_pixels,
        )
        if filtered:
            markdown = str(cells)
        else:
            layout_dir.mkdir(parents=True, exist_ok=True)
            layout_image = draw_layout_on_image(page.image, cells)
            layout_image_path = layout_dir / f"page_{page.page_no + 1:03d}.jpg"
            layout_image.save(layout_image_path)
            markdown_fragments = layout_cells_to_markdown_fragments(
                image=page.image,
                cells=cells,
                imgs_dir=imgs_dir,
                relative_img_dir=imgs_dir.name,
                page_no=page.page_no,
                no_page_header_footer=no_page_header_footer,
                infer_heading_hierarchy=infer_heading_hierarchy,
            )
            markdown = join_markdown_fragments(markdown_fragments)
    else:
        markdown = raw_response

    page_width, page_height = page.image.size
    inference_width, inference_height = inference_image.size
    raw_bytes = len(raw_response.encode("utf-8"))
    raw_chars = len(raw_response)
    postprocess_seconds = time.perf_counter() - postprocess_start
    page_completed_at = _utc_timestamp()
    total_seconds = time.perf_counter() - page_start
    timing = {
        "page_no": page.page_no,
        "display_page_no": page.page_no + 1,
        "started_at": page_started_at,
        "completed_at": page_completed_at,
        "preprocess_seconds": preprocess_seconds,
        "inference_started_at": inference_started_at,
        "inference_completed_at": inference_completed_at,
        "inference_seconds": inference_seconds,
        "postprocess_seconds": postprocess_seconds,
        "total_seconds": total_seconds,
        "source_image_path": _relative(page.source_image_path, output_dir),
        "inference_image_path": _relative(inference_image_path, output_dir),
        "raw_response_path": _relative(raw_path, output_dir),
        "layout_image_path": _relative(layout_image_path, output_dir) if layout_image_path else None,
        "source_width": page_width,
        "source_height": page_height,
        "inference_width": inference_width,
        "inference_height": inference_height,
        "raw_chars": raw_chars,
        "raw_bytes": raw_bytes,
        "filtered": filtered,
    }
    return PageOutput(
        page_no=page.page_no,
        source_image_path=page.source_image_path,
        inference_image_path=inference_image_path,
        layout_image_path=layout_image_path,
        markdown=markdown,
        markdown_fragments=markdown_fragments,
        cells=cells,
        filtered=filtered,
        raw_response=raw_response,
        page_width=page_width,
        page_height=page_height,
        timing=timing,
    )


def render_markdown(markdown_text: str) -> str:
    try:
        import markdown as markdown_lib

        return markdown_lib.markdown(
            markdown_text,
            extensions=["extra", "tables", "fenced_code", "sane_lists"],
            output_format="html5",
        )
    except Exception:
        escaped = html.escape(markdown_text)
        return f"<pre>{escaped}</pre>"


def _category_slug(category: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-")
    return slug or "unknown"


def _overlay_style(bbox: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    left = x1 / width * 100 if width else 0
    top = y1 / height * 100 if height else 0
    box_width = (x2 - x1) / width * 100 if width else 0
    box_height = (y2 - y1) / height * 100 if height else 0
    return (
        f"left:{left:.4f}%;top:{top:.4f}%;"
        f"width:{box_width:.4f}%;height:{box_height:.4f}%;"
    )


def render_interactive_markdown(page_outputs: list[PageOutput], fallback_markdown: str) -> str:
    if not any(page.markdown_fragments for page in page_outputs):
        return render_markdown(fallback_markdown)

    html_parts: list[str] = []
    multi_page = len(page_outputs) > 1
    for page_index, page in enumerate(page_outputs):
        if multi_page:
            html_parts.append(f'<div class="result-page-label">Page {page.page_no + 1}</div>')
        for fragment in page.markdown_fragments:
            cell_id = _cell_id(fragment.page_no, fragment.cell_index)
            category = fragment.category or "Unknown"
            rendered = render_markdown(fragment.markdown)
            html_parts.append(
                (
                    f'<div id="result-{cell_id}" '
                    f'class="result-block result-category-{html.escape(_category_slug(category), quote=True)}" '
                    f'data-cell-id="{html.escape(cell_id, quote=True)}" '
                    f'data-category="{html.escape(category, quote=True)}">'
                    f"{rendered}</div>"
                )
            )
        if multi_page and page_index != len(page_outputs) - 1:
            html_parts.append('<hr class="result-page-separator">')
    return "\n".join(html_parts)


def build_interactive_pages(page_outputs: list[PageOutput], output_dir: Path) -> list[dict[str, Any]]:
    interactive_pages: list[dict[str, Any]] = []
    for page in page_outputs:
        if not page.markdown_fragments:
            continue
        preview_path = page.layout_image_path or page.source_image_path
        cells = []
        for fragment in page.markdown_fragments:
            if fragment.bbox is None:
                continue
            cell_id = _cell_id(fragment.page_no, fragment.cell_index)
            cells.append(
                {
                    "cell_id": cell_id,
                    "cell_index": fragment.cell_index,
                    "category": fragment.category or "Unknown",
                    "bbox": fragment.bbox,
                }
            )
        if cells:
            interactive_pages.append(
                {
                    "page_no": page.page_no,
                    "image_path": _relative(preview_path, output_dir),
                    "width": page.page_width,
                    "height": page.page_height,
                    "cells": cells,
                }
            )
    return interactive_pages


def _build_interactive_preview_html(interactive_pages: list[dict[str, Any]]) -> str:
    page_html: list[str] = []
    for page in interactive_pages:
        width = int(page.get("width") or 1)
        height = int(page.get("height") or 1)
        image_path = html.escape(str(page.get("image_path", "")), quote=True)
        hotspots: list[str] = []
        for cell in page.get("cells", []):
            bbox = cell.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            cell_id = str(cell.get("cell_id", ""))
            category = str(cell.get("category", "Unknown") or "Unknown")
            slug = _category_slug(category)
            try:
                style = _overlay_style(tuple(int(value) for value in bbox), width, height)
            except (TypeError, ValueError):
                continue
            label = f"{category} cell {cell.get('cell_index', '')}".strip()
            hotspots.append(
                (
                    f'<button type="button" class="layout-hotspot hotspot-category-{html.escape(slug, quote=True)}" '
                    f'data-cell-id="{html.escape(cell_id, quote=True)}" '
                    f'data-category="{html.escape(category, quote=True)}" '
                    f'title="{html.escape(label, quote=True)}" '
                    f'aria-label="{html.escape(label, quote=True)}" '
                    f'style="{html.escape(style, quote=True)}"></button>'
                )
            )
        page_html.append(
            (
                f'<div class="preview-page" data-page-no="{int(page.get("page_no", 0)) + 1}">'
                f'<img class="source-page" src="{image_path}" alt="page {int(page.get("page_no", 0)) + 1}">'
                f'{"".join(hotspots)}</div>'
            )
        )
    return "\n".join(page_html)


def build_viewer_html(
    *,
    title: str,
    markdown_html: str,
    raw_markdown: str,
    json_text: str,
    source_image_paths: list[str],
    layout_image_paths: list[str],
    interactive_pages: list[dict[str, Any]] | None = None,
) -> str:
    preview_images = layout_image_paths or source_image_paths
    if interactive_pages:
        left_images = _build_interactive_preview_html(interactive_pages)
    else:
        left_images = "\n".join(
            f'<img class="source-page" src="{html.escape(path, quote=True)}" alt="page {idx + 1}">'
            for idx, path in enumerate(preview_images)
        )
    source_downloads = "\n".join(
        f'<li><a href="{html.escape(path, quote=True)}">source page {idx + 1}</a></li>'
        for idx, path in enumerate(source_image_paths)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · dots.mocr MLX comparison</title>
<style>
:root {{ --accent: #10b981; --border: #e5e7eb; --text: #24262b; }}
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: #fafafa; }}
main {{ display: grid; grid-template-columns: minmax(360px, 1fr) minmax(420px, 1fr); gap: 32px; padding: 24px; }}
h1 {{ margin: 0 0 18px; font-size: 28px; }}
.panel {{ min-width: 0; }}
.card {{ background: white; border: 1px solid var(--border); border-radius: 14px; box-shadow: 0 2px 8px rgba(0,0,0,.08); padding: 18px; }}
.preview-scroll, .result-scroll {{ max-height: calc(100vh - 160px); overflow: auto; }}
.preview-scroll {{ text-align: center; }}
.preview-page {{ position: relative; display: inline-block; max-width: 100%; margin: 0 auto 24px; }}
.source-page {{ max-width: 100%; height: auto; margin: 0 auto 24px; display: block; }}
.preview-page .source-page {{ margin: 0; }}
.layout-hotspot {{ position: absolute; box-sizing: border-box; border: 2px solid rgba(16,185,129,.72); background: rgba(16,185,129,.08); border-radius: 3px; padding: 0; cursor: pointer; opacity: .22; transition: opacity .15s, background-color .15s, border-color .15s; }}
.layout-hotspot:hover, .layout-hotspot.active-source {{ opacity: 1; background: rgba(16,185,129,.18); border-color: rgba(5,150,105,.95); }}
.hotspot-category-title, .hotspot-category-section-header {{ border-color: rgba(59,130,246,.8); background: rgba(59,130,246,.08); }}
.hotspot-category-caption {{ border-color: rgba(245,158,11,.85); background: rgba(245,158,11,.10); }}
.hotspot-category-picture {{ border-color: rgba(168,85,247,.85); background: rgba(168,85,247,.10); }}
.hotspot-category-table {{ border-color: rgba(239,68,68,.8); background: rgba(239,68,68,.08); }}
.tabs {{ display: flex; gap: 28px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }}
.tab-button {{ border: 0; background: transparent; padding: 12px 0 10px; font-size: 20px; cursor: pointer; color: #333; }}
.tab-button.active {{ color: var(--accent); border-bottom: 3px solid var(--accent); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
.markdown-body {{ font-size: 16px; line-height: 1.6; }}
.result-block {{ border-left: 3px solid transparent; border-radius: 6px; margin: 0 0 12px; padding: 4px 8px; transition: background-color .2s, border-color .2s; }}
.result-block.flash, .result-block.active-result {{ background: #ecfdf5; border-color: var(--accent); }}
.result-page-label {{ color: #6b7280; font-size: 13px; font-weight: 600; margin: 0 0 12px; text-transform: uppercase; }}
.result-page-separator {{ border: 0; border-top: 1px solid var(--border); margin: 20px 0; }}
.markdown-body img {{ max-width: 100%; display: block; margin: 16px 0; }}
.markdown-body table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
.markdown-body th, .markdown-body td {{ border: 1px solid var(--border); padding: 6px 8px; }}
pre {{ white-space: pre-wrap; word-break: break-word; background: #f3f4f6; border-radius: 10px; padding: 18px; }}
.meta {{ margin-top: 20px; font-size: 14px; color: #4b5563; text-align: left; }}
@media (max-width: 980px) {{ main {{ grid-template-columns: 1fr; }} .preview-scroll, .result-scroll {{ max-height: none; }} }}
</style>
</head>
<body>
<main>
  <section class="panel">
    <h1>👁️ File Preview</h1>
    <div class="card preview-scroll">
      {left_images}
      <div class="meta">
        <strong>Source pages:</strong>
        <ul>{source_downloads}</ul>
      </div>
    </div>
  </section>
  <section class="panel">
    <h1>✔ Result Display</h1>
    <div class="card result-scroll">
      <div class="tabs" role="tablist">
        <button class="tab-button active" data-tab="markdown">Markdown Render Preview</button>
        <button class="tab-button" data-tab="json">Formatted JSON</button>
        <button class="tab-button" data-tab="raw">Raw Markdown</button>
      </div>
      <div id="markdown" class="tab-content active markdown-body">{markdown_html}</div>
      <div id="json" class="tab-content"><pre>{html.escape(json_text)}</pre></div>
      <div id="raw" class="tab-content"><pre>{html.escape(raw_markdown)}</pre></div>
    </div>
  </section>
</main>
<script>
function activateTab(tabName) {{
  const button = document.querySelector(`.tab-button[data-tab="${{tabName}}"]`);
  if (button && !button.classList.contains('active')) {{
    button.click();
  }}
}}

function activateMappedCell(cellId) {{
  const target = document.getElementById(`result-${{cellId}}`);
  if (!target) {{
    return;
  }}
  activateTab('markdown');
  document.querySelectorAll('.layout-hotspot.active-source').forEach(el => el.classList.remove('active-source'));
  document.querySelectorAll('.result-block.active-result, .result-block.flash').forEach(el => {{
    el.classList.remove('active-result');
    el.classList.remove('flash');
  }});
  const source = document.querySelector(`.layout-hotspot[data-cell-id="${{cellId}}"]`);
  if (source) {{
    source.classList.add('active-source');
  }}
  target.classList.add('active-result');
  target.classList.add('flash');
  requestAnimationFrame(() => {{
    target.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
  }});
  window.setTimeout(() => target.classList.remove('flash'), 1600);
}}

for (const button of document.querySelectorAll('.tab-button')) {{
  button.addEventListener('click', () => {{
    document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    button.classList.add('active');
    document.getElementById(button.dataset.tab).classList.add('active');
  }});
}}

for (const hotspot of document.querySelectorAll('.layout-hotspot')) {{
  hotspot.addEventListener('click', () => activateMappedCell(hotspot.dataset.cellId));
}}
</script>
</body>
</html>
"""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def run_pipeline(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    output_name: str | None = None,
    runner: InferenceRunner | None = None,
    backend: str = "mlx",
    model_id: str = DEFAULT_MLX_MODEL,
    api_base_url: str = DEFAULT_OMLX_API_BASE_URL,
    api_key: str = DEFAULT_OMLX_API_KEY,
    api_model: str = DEFAULT_OMLX_MODEL,
    prompt_mode: str = "prompt_layout_all_en",
    custom_prompt: str | None = None,
    dpi: int = 200,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    fitz_preprocess: bool = False,
    no_page_header_footer: bool = False,
    infer_heading_hierarchy: bool = False,
    launch_viewer: bool = False,
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = False,
    max_tokens: int = 24000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: int | None = 0,
    repetition_penalty: float | None = 1.05,
    api_timeout: float = 300.0,
    api_max_retries: int = 1,
    num_thread: int = 1,
    parallel_mode: str = "server-safe",
    trust_remote_code: bool = True,
    revision: str | None = None,
    prefill_step_size: int | None = None,
    skip_special_tokens: bool = True,
) -> PipelineResult:
    pipeline_started_at = _utc_timestamp()
    pipeline_start = time.perf_counter()
    input_path = Path(input_path).expanduser().resolve()
    output_name = output_name or _safe_output_name(input_path)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    imgs_dir = output_dir / f"{output_name}_imgs"
    pages_dir = output_dir / f"{output_name}_pages"
    layout_dir = output_dir / f"{output_name}_layout"
    inference_dir = output_dir / f"{output_name}_mlx_input"
    for directory in (imgs_dir, pages_dir, layout_dir, inference_dir):
        directory.mkdir(parents=True, exist_ok=True)

    render_start = time.perf_counter()
    pages = load_input_pages(input_path, pages_dir=pages_dir, dpi=dpi, fitz_preprocess=fitz_preprocess)
    render_seconds = time.perf_counter() - render_start
    prompt = custom_prompt if custom_prompt is not None else dict_promptmode_to_prompt[prompt_mode]

    if runner is None:
        if backend == "mlx":
            runner = MlxDotsRunner(
                model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                trust_remote_code=trust_remote_code,
                revision=revision,
                prefill_step_size=prefill_step_size,
                skip_special_tokens=skip_special_tokens,
            )
        elif backend == "omlx-api":
            runner = OmlxApiRunner(
                base_url=api_base_url,
                api_key=api_key,
                model_name=api_model,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                repetition_penalty=repetition_penalty,
                timeout=api_timeout,
                max_retries=api_max_retries,
            )
        else:
            raise ValueError(f"Unsupported backend {backend!r}; expected 'mlx' or 'omlx-api'")

    def _process_page(page: PageInput) -> PageOutput:
        return process_page(
            page=page,
            runner=runner,
            prompt=prompt,
            prompt_mode=prompt_mode,
            output_name=output_name,
            output_dir=output_dir,
            imgs_dir=imgs_dir,
            layout_dir=layout_dir,
            inference_dir=inference_dir,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            no_page_header_footer=no_page_header_footer,
            infer_heading_hierarchy=infer_heading_hierarchy,
        )

    requested_worker_count = max(1, min(num_thread, len(pages)))
    if backend == "omlx-api" and parallel_mode == "server-safe":
        worker_count = 1
    else:
        worker_count = requested_worker_count
    if worker_count == 1:
        page_outputs = [_process_page(page) for page in pages]
    else:
        page_outputs = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_page = {executor.submit(_process_page, page): page for page in pages}
            for future in as_completed(future_to_page):
                page_outputs.append(future.result())
        page_outputs.sort(key=lambda page: page.page_no)

    if len(page_outputs) == 1:
        combined_markdown = page_outputs[0].markdown
    else:
        combined_markdown = "\n\n---\n\n".join(
            f"<!-- page {page.page_no + 1} -->\n\n{page.markdown}" for page in page_outputs
        )

    markdown_path = output_dir / f"{output_name}.md"
    markdown_path.write_text(combined_markdown, encoding="utf-8")

    json_payload = [
        {
            "page_no": page.page_no,
            "source_image_path": _relative(page.source_image_path, output_dir),
            "inference_image_path": _relative(page.inference_image_path, output_dir),
            "layout_image_path": _relative(page.layout_image_path, output_dir) if page.layout_image_path else None,
            "filtered": page.filtered,
            "cells": page.cells,
        }
        for page in page_outputs
    ]
    json_text = json.dumps(json_payload, ensure_ascii=False, indent=2, default=_json_default)
    json_path = output_dir / f"{output_name}.json"
    json_path.write_text(json_text, encoding="utf-8")

    pipeline_completed_at = _utc_timestamp()
    timings_payload = {
        "output_name": output_name,
        "input_path": input_path.as_posix(),
        "backend": backend,
        "model": api_model if backend == "omlx-api" else model_id,
        "prompt_mode": prompt_mode,
        "page_count": len(page_outputs),
        "requested_num_thread": num_thread,
        "requested_worker_count": requested_worker_count,
        "worker_count": worker_count,
        "started_at": pipeline_started_at,
        "completed_at": pipeline_completed_at,
        "total_seconds": time.perf_counter() - pipeline_start,
        "render_seconds": render_seconds,
        "settings": {
            "dpi": dpi,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "fitz_preprocess": fitz_preprocess,
            "no_page_header_footer": no_page_header_footer,
            "infer_heading_hierarchy": infer_heading_hierarchy,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "seed": seed,
            "repetition_penalty": repetition_penalty,
            "api_timeout": api_timeout,
            "api_max_retries": api_max_retries,
            "parallel_mode": parallel_mode,
        },
        "pages": [page.timing for page in page_outputs],
    }
    timings_path = output_dir / f"{output_name}_timings.json"
    timings_path.write_text(json.dumps(timings_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    interactive_pages = build_interactive_pages(page_outputs, output_dir)
    markdown_html = (
        render_interactive_markdown(page_outputs, combined_markdown)
        if interactive_pages
        else render_markdown(combined_markdown)
    )
    viewer_html = build_viewer_html(
        title=output_name,
        markdown_html=markdown_html,
        raw_markdown=combined_markdown,
        json_text=json_text,
        source_image_paths=[_relative(page.source_image_path, output_dir) for page in page_outputs],
        layout_image_paths=[
            _relative(page.layout_image_path, output_dir)
            for page in page_outputs
            if page.layout_image_path is not None
        ],
        interactive_pages=interactive_pages,
    )
    viewer_path = output_dir / f"{output_name}_viewer.html"
    viewer_path.write_text(viewer_html, encoding="utf-8")

    result = PipelineResult(
        markdown_path=markdown_path,
        images_dir=imgs_dir,
        json_path=json_path,
        timings_path=timings_path,
        viewer_path=viewer_path,
        page_outputs=page_outputs,
    )
    if launch_viewer:
        result.viewer_url = serve_viewer(viewer_path, host=host, port=port, open_browser=open_browser, block=True)
    return result


class _ViewerRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve_viewer(
    viewer_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = False,
    block: bool = True,
) -> str:
    viewer_path = Path(viewer_path).expanduser().resolve()
    directory = viewer_path.parent

    def handler(*args: Any, **kwargs: Any) -> _ViewerRequestHandler:
        return _ViewerRequestHandler(*args, directory=str(directory), **kwargs)

    server = ThreadingHTTPServer((host, port), handler)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/{quote(viewer_path.name)}"
    print(f"Viewer: {url}")
    if open_browser:
        webbrowser.open(url)

    if not block:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return url

    print("Serving comparison viewer. Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        server.server_close()
    return url


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    prompts = sorted(dict_promptmode_to_prompt.keys())
    parser = argparse.ArgumentParser(
        description="Parse a PDF/image with an MLX-optimized dots.mocr model and export markdown + cropped image links.",
    )
    parser.add_argument("input_path", help="Input PDF/image path")
    parser.add_argument("--output-dir", "-o", default="./output_mlx", help="Directory for all generated files")
    parser.add_argument("--output-name", help="Base output name; default: input filename stem")
    parser.add_argument(
        "--backend",
        choices=["mlx", "omlx-api"],
        default="mlx",
        help="Inference backend: load an MLX model in-process, or call an OpenAI-compatible oMLX API.",
    )
    parser.add_argument("--model", default=DEFAULT_MLX_MODEL, help=f"MLX model id/path (default: {DEFAULT_MLX_MODEL})")
    parser.add_argument("--api-base-url", default=DEFAULT_OMLX_API_BASE_URL, help="oMLX API base URL")
    parser.add_argument("--api-key", default=DEFAULT_OMLX_API_KEY, help="oMLX API key")
    parser.add_argument("--api-model", default=DEFAULT_OMLX_MODEL, help="oMLX model name")
    parser.add_argument("--api-timeout", type=float, default=300.0, help="oMLX API request timeout in seconds")
    parser.add_argument("--api-max-retries", type=int, default=1, help="OpenAI client retries for oMLX API requests")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision")
    parser.add_argument("--prompt", choices=prompts, default="prompt_layout_all_en", help="dots.mocr prompt mode")
    parser.add_argument("--custom-prompt", default=None, help="Override prompt text")
    parser.add_argument("--dpi", type=int, default=200, help="PDF render DPI and optional image fitz-preprocess DPI")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--fitz-preprocess", action="store_true", help="Pre-render image inputs through PyMuPDF before inference")
    parser.add_argument("--no-page-header-footer", action="store_true", help="Skip Page-header/Page-footer cells in markdown")
    parser.add_argument(
        "--infer-heading-hierarchy",
        action="store_true",
        help="Opt in to conservative Markdown heading inference for article/PDF title and section depth.",
    )
    parser.add_argument("--max-tokens", type=int, default=24000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0, help="oMLX API nucleus sampling value")
    parser.add_argument("--frequency-penalty", type=float, default=0.0, help="oMLX API frequency penalty")
    parser.add_argument("--presence-penalty", type=float, default=0.0, help="oMLX API presence penalty")
    parser.add_argument("--seed", type=int, default=0, help="oMLX API deterministic seed")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="oMLX API repetition penalty passed through extra_body; set to 1.0 to disable effect.",
    )
    parser.add_argument("--num-thread", type=int, default=1, help="Page concurrency; use >1 primarily with API backend")
    parser.add_argument(
        "--parallel-mode",
        choices=["fixed", "server-safe"],
        default="server-safe",
        help="Page scheduling policy. server-safe limits oMLX API runs to one in-flight request unless fixed is selected.",
    )
    parser.add_argument(
        "--performance-preset",
        choices=sorted(PERFORMANCE_PRESET_MAX_PIXELS),
        default="quality",
        help=(
            "Preset for oMLX image token cost. quality keeps existing max-pixels, "
            "balanced uses 1600000, fast uses 1000000 unless --max-pixels is set."
        ),
    )
    parser.add_argument("--prefill-step-size", type=int, default=None, help="Lower this, e.g. 512, if MLX prefill runs out of memory")
    parser.add_argument("--skip-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--view", action="store_true", help="Launch the side-by-side comparison web viewer after parsing")
    parser.add_argument("--host", default="127.0.0.1", help="Viewer host")
    parser.add_argument("--port", type=int, default=7860, help="Viewer port; use 0 for an available port")
    parser.add_argument("--open-browser", action="store_true", help="Open viewer URL in the default browser")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    max_pixels = resolve_max_pixels(args.max_pixels, args.performance_preset)
    result = run_pipeline(
        input_path=args.input_path,
        output_dir=args.output_dir,
        output_name=args.output_name,
        backend=args.backend,
        model_id=args.model,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        api_model=args.api_model,
        prompt_mode=args.prompt,
        custom_prompt=args.custom_prompt,
        dpi=args.dpi,
        min_pixels=args.min_pixels,
        max_pixels=max_pixels,
        fitz_preprocess=args.fitz_preprocess,
        no_page_header_footer=args.no_page_header_footer,
        infer_heading_hierarchy=args.infer_heading_hierarchy,
        launch_viewer=False,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        api_timeout=args.api_timeout,
        api_max_retries=args.api_max_retries,
        num_thread=args.num_thread,
        parallel_mode=args.parallel_mode,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        prefill_step_size=args.prefill_step_size,
        skip_special_tokens=args.skip_special_tokens,
    )
    print(f"Markdown: {result.markdown_path}")
    print(f"Images:   {result.images_dir}")
    print(f"JSON:     {result.json_path}")
    print(f"Timings:  {result.timings_path}")
    print(f"Viewer:   {result.viewer_path}")
    if args.view:
        serve_viewer(result.viewer_path, host=args.host, port=args.port, open_browser=args.open_browser, block=True)


if __name__ == "__main__":
    main()
