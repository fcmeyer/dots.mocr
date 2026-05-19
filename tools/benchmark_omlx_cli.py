"""Benchmark the dots-mocr-mlx oMLX API path on a fixed PDF subset."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _max_pixels_label(max_pixels: int | None) -> str:
    return "default" if max_pixels is None else str(max_pixels)


def parse_max_pixels(value: str) -> int | None:
    if value == "default":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-pixels values must be positive integers or 'default'")
    return parsed


def create_pdf_subset(input_path: Path, output_path: Path, pages: int) -> Path:
    import fitz

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    source = fitz.open(input_path)
    try:
        page_count = min(pages, source.page_count)
        if page_count <= 0:
            raise ValueError(f"Cannot benchmark {input_path}: no pages found")
        dest = fitz.open()
        try:
            dest.insert_pdf(source, from_page=0, to_page=page_count - 1)
            dest.save(output_path)
        finally:
            dest.close()
    finally:
        source.close()
    return output_path


def summarize_run(*, timings_path: Path, json_path: Path, threads: int, max_pixels: int | None) -> dict[str, Any]:
    timings = json.loads(timings_path.read_text(encoding="utf-8"))
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    cell_counts: list[int] = []
    category_counts: Counter[str] = Counter()
    for page in payload:
        cells = page.get("cells")
        if not isinstance(cells, list):
            cells = []
        cell_counts.append(len(cells))
        for cell in cells:
            if isinstance(cell, dict):
                category_counts[str(cell.get("category", ""))] += 1

    pages = timings.get("pages", [])
    dimensions = sorted(
        {
            f"{page['inference_width']}x{page['inference_height']}"
            for page in pages
            if "inference_width" in page and "inference_height" in page
        }
    )

    return {
        "threads": threads,
        "max_pixels": max_pixels,
        "total_seconds": timings["total_seconds"],
        "inference_seconds_sum": sum(page.get("inference_seconds", 0.0) for page in pages),
        "input_dimensions": dimensions,
        "cell_counts": cell_counts,
        "category_counts": dict(sorted(category_counts.items())),
        "worker_count": timings.get("worker_count"),
        "requested_worker_count": timings.get("requested_worker_count"),
    }


def _default_cli_path() -> str:
    sibling = Path(sys.executable).with_name("dots-mocr-mlx")
    if sibling.exists():
        return sibling.as_posix()
    return shutil.which("dots-mocr-mlx") or "dots-mocr-mlx"


def build_command(
    *,
    cli_path: str,
    subset_pdf: Path,
    output_dir: Path,
    output_name: str,
    threads: int,
    max_pixels: int | None,
    api_base_url: str,
    api_key: str,
    api_model: str,
    parallel_mode: str,
) -> list[str]:
    command = [
        cli_path,
        subset_pdf.as_posix(),
        "--backend",
        "omlx-api",
        "--api-base-url",
        api_base_url,
        "--api-key",
        api_key,
        "--api-model",
        api_model,
        "--output-dir",
        output_dir.as_posix(),
        "--output-name",
        output_name,
        "--temperature",
        "0",
        "--top-p",
        "1.0",
        "--max-tokens",
        "24000",
        "--frequency-penalty",
        "0",
        "--presence-penalty",
        "0",
        "--seed",
        "0",
        "--repetition-penalty",
        "1.05",
        "--num-thread",
        str(threads),
        "--parallel-mode",
        parallel_mode,
    ]
    if max_pixels is not None:
        command.extend(["--max-pixels", str(max_pixels)])
    return command


def run_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_pdf = output_dir / f"{input_path.stem}-first{args.pages}.pdf"
    create_pdf_subset(input_path, subset_pdf, args.pages)

    runs: list[dict[str, Any]] = []
    for threads in args.threads:
        for max_pixels in args.max_pixels:
            output_name = f"bench_p{args.pages}_t{threads}_max{_max_pixels_label(max_pixels)}"
            run_output_dir = output_dir / output_name
            if run_output_dir.exists():
                shutil.rmtree(run_output_dir)
            command = build_command(
                cli_path=args.cli_path,
                subset_pdf=subset_pdf,
                output_dir=run_output_dir,
                output_name=output_name,
                threads=threads,
                max_pixels=max_pixels,
                api_base_url=args.api_base_url,
                api_key=args.api_key,
                api_model=args.api_model,
                parallel_mode=args.parallel_mode,
            )
            completed = subprocess.run(command, check=False)
            run_summary = {
                "returncode": completed.returncode,
                "command": command,
            }
            timings_path = run_output_dir / f"{output_name}_timings.json"
            json_path = run_output_dir / f"{output_name}.json"
            if timings_path.exists() and json_path.exists():
                run_summary.update(
                    summarize_run(
                        timings_path=timings_path,
                        json_path=json_path,
                        threads=threads,
                        max_pixels=max_pixels,
                    )
                )
            runs.append(run_summary)

    summary = {
        "input": input_path.as_posix(),
        "subset_pdf": subset_pdf.as_posix(),
        "pages": args.pages,
        "api_base_url": args.api_base_url,
        "api_model": args.api_model,
        "parallel_mode": args.parallel_mode,
        "runs": runs,
    }
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dots-mocr-mlx with a local oMLX API server.")
    parser.add_argument("--input", default="assets/mlx/book-chapter-2.pdf")
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--output-dir", default="/tmp/dots_mocr_omlx_bench")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--max-pixels", type=parse_max_pixels, nargs="+", default=[None, 1_600_000, 1_000_000])
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="1220")
    parser.add_argument("--api-model", default="dots.mocr-bf16")
    parser.add_argument("--parallel-mode", choices=["fixed", "server-safe"], default="fixed")
    parser.add_argument("--cli-path", default=_default_cli_path())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    summary = run_benchmarks(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
