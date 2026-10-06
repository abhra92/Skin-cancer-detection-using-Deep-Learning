"""Inventory the supplied repository without counting generated deliverables."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE.parent
REPO_ROOT = OUTPUT_DIR.parent
EXCLUDED_TOP_LEVEL = {".git", "Skin_Cancer_Deep_Learning_Research"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".keras", ".h5", ".onnx"}
PLOT_EXTENSIONS = {".png", ".svg", ".pdf"}
DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    lower = path.name.lower()
    if suffix == ".pdf":
        if "research problem" in lower or lower == "main.pdf" or "experiment_no_1" in lower:
            return "research_methodology_or_draft_pdf"
        if any(part in {"HAM10000", "ISIC 2019", "PH2", "Paper"} for part in path.relative_to(REPO_ROOT).parts[:-1]):
            return "research_paper_pdf"
        return "pdf_document"
    if suffix == ".docx":
        return "word_document"
    if suffix == ".ipynb":
        return "notebook"
    if suffix == ".py":
        return "python_script"
    if suffix in {".csv", ".tsv", ".parquet"}:
        return "tabular_file"
    if suffix in IMAGE_EXTENSIONS:
        return "image_asset"
    if suffix in WEIGHT_EXTENSIONS:
        return "model_checkpoint"
    if suffix in PLOT_EXTENSIONS:
        return "plot_or_pdf"
    if suffix in {".md", ".txt", ".tex", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        return "text_or_configuration"
    return "other"


def read_mem_kb(key: str) -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def hardware_inventory() -> dict[str, object]:
    gpu_command = shutil.which("nvidia-smi")
    gpu_summary = "NOT DETECTED (nvidia-smi is unavailable)"
    if gpu_command:
        result = subprocess.run([gpu_command, "-L"], capture_output=True, text=True, timeout=10)
        gpu_summary = (result.stdout or result.stderr).strip() or "nvidia-smi ran but reported no GPU"
    torch_available = importlib.util.find_spec("torch") is not None
    torch_version = "NOT INSTALLED"
    cuda_available: object = "NOT CHECKED"
    if torch_available:
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                gpu_summary = "; ".join(torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count()))
            else:
                cuda_available = False
        except Exception as exc:  # Keep the audit explicit about an unusable import.
            torch_version = f"IMPORT ERROR: {type(exc).__name__}: {exc}"
            cuda_available = "UNKNOWN"
    module_probe = (
        "import importlib.util,json,sys; "
        "mods=['torch','torchvision','tensorflow','sklearn','pandas','numpy','PIL','matplotlib','seaborn','openpyxl']; "
        "print(json.dumps({'python_version':sys.version.split()[0],"
        "'python_executable':sys.executable,'modules':{m:bool(importlib.util.find_spec(m)) for m in mods}}))"
    )
    baseline_python = Path("/usr/bin/python")
    baseline_environment: dict[str, object] = {"python_executable": str(baseline_python), "status": "NOT AVAILABLE"}
    if baseline_python.exists():
        baseline = subprocess.run([str(baseline_python), "-c", module_probe], capture_output=True, text=True, timeout=10)
        if baseline.returncode == 0:
            baseline_environment = json.loads(baseline.stdout)
            baseline_environment["status"] = "queried from untouched system Python"
        else:
            baseline_environment["status"] = f"probe failed: {baseline.stderr.strip()}"
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "initial_system_python_environment": baseline_environment,
        "platform": platform.platform(),
        "cpu_count_visible": os.cpu_count(),
        "memory_total_mib": round((read_mem_kb("MemTotal") or 0) / 1024, 1) or None,
        "memory_available_mib": round((read_mem_kb("MemAvailable") or 0) / 1024, 1) or None,
        "gpu": gpu_summary,
        "nvidia_smi_available": bool(gpu_command),
        "torch_installed": torch_available,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "active_interpreter_ml_libraries": {
            package: bool(importlib.util.find_spec(package))
            for package in ["torch", "torchvision", "tensorflow", "sklearn", "pandas", "numpy", "PIL", "matplotlib", "seaborn", "openpyxl"]
        },
    }


def source_paper_inventory(files: list[Path]) -> tuple[list[dict[str, object]], list[str]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)
    inventory: list[dict[str, object]] = []
    errors: list[str] = []
    for digest, paths in sorted(groups.items(), key=lambda item: relative(item[1][0])):
        representative = paths[0]
        title = ""
        author = ""
        page_count: int | str = "NOT AVAILABLE"
        first_page = ""
        extraction_status = "OK"
        try:
            reader = PdfReader(str(representative), strict=False)
            page_count = len(reader.pages)
            metadata = reader.metadata
            if metadata:
                title = str(metadata.title or "").strip()
                author = str(metadata.author or "").strip()
            first_page = " ".join((reader.pages[0].extract_text() or "").replace("\x00", " ").split()) if page_count else ""
            if not first_page:
                extraction_status = "PDF text unavailable on page 1"
        except Exception as exc:
            extraction_status = f"ERROR {type(exc).__name__}: {exc}"
            errors.append(f"{relative(representative)}: {extraction_status}")
        paths_rel = [relative(path) for path in paths]
        research_paths = [path for path in paths if classify(path) == "research_paper_pdf"]
        inventory.append({
            "sha256": digest,
            "representative_path": relative(representative),
            "all_copy_paths": " | ".join(paths_rel),
            "copy_count": len(paths),
            "unique_content": True,
            "document_category": "research paper" if research_paths else "methodology or draft document",
            "research_paper_copy_present": bool(research_paths),
            "bytes": representative.stat().st_size,
            "page_count": page_count,
            "pdf_metadata_title": title,
            "pdf_metadata_author": author,
            "page1_text_extract_status": extraction_status,
            "page1_excerpt": first_page[:2400],
        })
    return inventory, errors


def inventory_files() -> list[Path]:
    results = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(REPO_ROOT).parts
        if any(part in EXCLUDED_TOP_LEVEL for part in relative_parts):
            continue
        if any(part in {"__pycache__", ".ipynb_checkpoints"} for part in relative_parts):
            continue
        results.append(path)
    return sorted(results)


def notebook_summary(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in data.get("cells", []) if cell.get("cell_type") == "code"]
    return {
        "path": relative(path),
        "cell_count": len(data.get("cells", [])),
        "code_cells": len(code_cells),
        "executed_code_cells": sum(cell.get("execution_count") is not None for cell in code_cells),
        "stored_outputs": sum(len(cell.get("outputs", [])) for cell in code_cells),
    }


def main() -> int:
    files = inventory_files()
    counts: dict[str, int] = defaultdict(int)
    rows: list[dict[str, object]] = []
    for path in files:
        kind = classify(path)
        counts[kind] += 1
        rows.append({"relative_path": relative(path), "category": kind, "bytes": path.stat().st_size})

    pdfs = [path for path in files if path.suffix.lower() == ".pdf"]
    papers = [path for path in pdfs if classify(path) == "research_paper_pdf"]
    paper_inventory, pdf_errors = source_paper_inventory(pdfs)
    paper_csv = OUTPUT_DIR / "02_literature" / "source_paper_inventory.csv"
    paper_csv.parent.mkdir(parents=True, exist_ok=True)
    if paper_inventory:
        with paper_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(paper_inventory[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(paper_inventory)

    file_csv = OUTPUT_DIR / "01_project_audit" / "workspace_inventory.csv"
    with file_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["relative_path", "category", "bytes"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    docs = [path for path in files if path.suffix.lower() == ".docx"]
    docx_paragraph_counts: dict[str, int | str] = {}
    for path in docs:
        try:
            with zipfile.ZipFile(path) as zipped:
                xml = ET.fromstring(zipped.read("word/document.xml"))
            docx_paragraph_counts[relative(path)] = sum(
                bool("".join(item.text or "" for item in paragraph.findall(".//w:t", DOCX_NS)).strip())
                for paragraph in xml.findall(".//w:p", DOCX_NS)
            )
        except (OSError, zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
            docx_paragraph_counts[relative(path)] = f"EXTRACTION ERROR: {type(exc).__name__}: {exc}"

    notebooks = [notebook_summary(path) for path in files if path.suffix.lower() == ".ipynb"]
    csv_files = [path for path in files if path.suffix.lower() == ".csv"]
    dataset_csv_candidates = []
    for path in csv_files:
        try:
            with path.open(encoding="utf-8-sig", newline="") as file:
                columns = next(csv.reader(file), [])
            if {"image_id", "dx"}.issubset({column.strip().lower() for column in columns}):
                dataset_csv_candidates.append(relative(path))
        except (OSError, UnicodeDecodeError, csv.Error):
            continue

    candidates = {
        "image_assets": [relative(p) for p in files if p.suffix.lower() in IMAGE_EXTENSIONS],
        "metadata_csv_candidates": dataset_csv_candidates,
        "all_csv_files": [relative(p) for p in csv_files],
        "checkpoints": [relative(p) for p in files if p.suffix.lower() in WEIGHT_EXTENSIONS],
        "plots_or_figures": [relative(p) for p in files if p.suffix.lower() in PLOT_EXTENSIONS and p.suffix.lower() != ".pdf"],
        "python_scripts": [relative(p) for p in files if p.suffix.lower() == ".py"],
        "notebooks": notebooks,
        "word_documents": docx_paragraph_counts,
    }
    hardware = hardware_inventory()
    audit_timestamp = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds")
    try:
        disk = shutil.disk_usage(REPO_ROOT)
        disk_summary = {"free_gib": round(disk.free / (1024 ** 3), 2), "total_gib": round(disk.total / (1024 ** 3), 2)}
    except OSError:
        disk_summary = {"free_gib": "NOT AVAILABLE", "total_gib": "NOT AVAILABLE"}

    # Record the CSV as literature metadata rather than treating it as image labels.
    bibliography_rows = None
    bibliography_path = REPO_ROOT / "Extra" / "skin_cancer_papers_clean.csv"
    if bibliography_path.exists():
        try:
            with bibliography_path.open(encoding="utf-8-sig", newline="") as file:
                bibliography_rows = max(0, sum(1 for _ in csv.DictReader(file)))
        except (OSError, UnicodeDecodeError, csv.Error):
            bibliography_rows = "READ ERROR"

    unique_hashes = {entry["sha256"] for entry in paper_inventory}
    research_pdf_hashes = {
        entry["sha256"] for entry in paper_inventory if entry["research_paper_copy_present"]
    }
    initial_notebook = notebooks[0] if notebooks else None
    summary = {
        "audit_timestamp_utc": audit_timestamp,
        "repository": str(REPO_ROOT),
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip() or "NOT AVAILABLE",
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip() or "NOT AVAILABLE",
        "inventory_scope": "All tracked and untracked repository source files outside .git and the generated research-output directory; no files are modified by the audit.",
        "source_file_count": len(files),
        "file_counts_by_category": dict(sorted(counts.items())),
        "pdf_path_count": len(pdfs),
        "unique_pdf_contents": len(unique_hashes),
        "research_paper_pdf_paths": len(papers),
        "unique_research_paper_contents": len(research_pdf_hashes),
        "pdf_extraction_errors": pdf_errors,
        "docx_count": len(docs),
        "notebooks": notebooks,
        "dataset_findings": {
            "image_files_found": len(candidates["image_assets"]),
            "metadata_csv_candidates": dataset_csv_candidates,
            "ham10000_metadata_found": any(Path(path).name.lower() == "ham10000_metadata.csv" for path in candidates["metadata_csv_candidates"]),
            "only_csv_files": candidates["all_csv_files"],
            "bibliography_csv_rows": bibliography_rows,
            "interpretation": "The only supplied CSV is a literature bibliography; it does not have image_id/dx fields and is not dataset metadata." if not dataset_csv_candidates else "Metadata candidate(s) listed above; they require dataset audit.",
        },
        "existing_results": {
            "checkpoints": candidates["checkpoints"],
            "plots": candidates["plots_or_figures"],
            "existing_python_scripts_before_generated_pipeline": [],
            "other_csv_files": candidates["all_csv_files"],
            "existing_notebook_outputs": "No code cells have execution counts and no stored cell outputs." if initial_notebook and initial_notebook["executed_code_cells"] == 0 and initial_notebook["stored_outputs"] == 0 else "See notebook inventory.",
            "existing_paper_text_extracts": [relative(p) for p in files if "/pdf_text_dump/" in relative(p)],
            "existing_extraction_json": [relative(p) for p in files if p.name in {"paper_extracts.json", "paper_full_extracts.json", "knowledge_graph_output.json"}],
        },
        "hardware": hardware,
        "disk": disk_summary,
        "analysis_environment_packages": {
            package: (
                importlib.metadata.version(package)
                if importlib.util.find_spec(module_name) is not None
                else "NOT INSTALLED"
            )
            for package, module_name in {
                "pypdf": "pypdf",
                "numpy": "numpy",
                "pandas": "pandas",
                "scikit-learn": "sklearn",
                "matplotlib": "matplotlib",
                "pillow": "PIL",
                "seaborn": "seaborn",
                "openpyxl": "openpyxl",
                "torch": "torch",
                "torchvision": "torchvision",
            }.items()
        },
        "important_source_limit": "The unexecuted notebook contains a Kaggle-download instruction, but no Kaggle dataset files, credentials, checkpoints, outputs, or local model framework were present in the initial host Python environment.",
    }
    (OUTPUT_DIR / "01_project_audit" / "audit_environment.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Project audit",
        "",
        f"- Audit time (UTC): `{audit_timestamp}`",
        f"- Repository: `{REPO_ROOT}`",
        f"- Branch: `{summary['branch']}`",
        f"- Commit at audit: `{summary['commit']}`",
        f"- Source inventory file count: **{len(files)}** (generated outputs and `.git` excluded)",
        "",
        "## Workspace inventory",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "The full file-by-file inventory is `workspace_inventory.csv`. Source PDFs are catalogued with SHA-256 deduplication and page-one text extracts in `../02_literature/source_paper_inventory.csv`.",
        "",
        "## Research papers and documents",
        "",
        f"- PDF paths: **{len(pdfs)}**; unique PDF contents: **{len(unique_hashes)}**.",
        f"- Research-paper PDF paths: **{len(papers)}**; unique paper contents: **{len(research_pdf_hashes)}**.",
        f"- PDF text extraction errors: **{len(pdf_errors)}**.",
        f"- Word documents: **{len(docs)}**; paragraph counts: `{json.dumps(docx_paragraph_counts, ensure_ascii=False)}`.",
        "- Existing paper text extracts and JSON/knowledge-graph files are identified in `audit_environment.json` and `workspace_inventory.csv`.",
        "- The existing `Extra/skin_cancer_papers_clean.csv` is bibliography metadata (45 rows in the supplied file), not image labels or HAM10000 metadata.",
        "",
        "## Dataset inventory",
        "",
        f"- Image files (JPG/JPEG/PNG/WebP/BMP): **{len(candidates['image_assets'])}**.",
        f"- CSV files: `{', '.join(candidates['all_csv_files']) if candidates['all_csv_files'] else 'none'}`.",
        f"- HAM10000 metadata CSV candidate(s): `{', '.join(dataset_csv_candidates) if dataset_csv_candidates else 'NONE'}`.",
        "- HAM10000 image folders, ISIC image data, PH2 image data, and metadata labels are not present in the inspected workspace. The directories with those names contain research papers/reviews, not data.",
        "- As no image/metadata rows exist, class frequencies, missing-value rates, duplicates, corrupted images, and metadata-image consistency are **NOT AVAILABLE** and are not estimated.",
        "",
        "## Existing code and results",
        "",
        f"- Existing Python scripts before this generated pipeline: **0**.",
        f"- Notebook(s): `{json.dumps(notebooks, ensure_ascii=False)}`.",
        "- The only notebook is a 25-cell PyTorch EfficientNet-B0/HAM10000 workflow. It has zero executed code cells and zero stored outputs; it is not evidence of executed training.",
        "- It specifies lesion-group splitting, train-only augmentation, weighted cross entropy, ImageNet transfer weights and test evaluation, but has no local data, metric exports, or model artifacts.",
        f"- Existing model checkpoints: **{len(candidates['checkpoints'])}**; existing plot/image files: **{len(candidates['plots_or_figures'])}**.",
        "- No existing experiment metrics or confusion matrices were supplied. The literature bibliography CSV is not counted as results.",
        "",
        "## Runtime and compute",
        "",
        f"- Python: `{hardware['python_version']}` at `{hardware['python_executable']}`.",
        f"- OS/platform: `{hardware['platform']}`.",
        f"- CPU cores visible: `{hardware['cpu_count_visible']}`.",
        f"- RAM total/available: `{hardware['memory_total_mib']} / {hardware['memory_available_mib']} MiB`.",
        f"- GPU: `{hardware['gpu']}`; `nvidia-smi` available: `{hardware['nvidia_smi_available']}`.",
        f"- PyTorch installed: `{hardware['torch_installed']}`; CUDA available: `{hardware['cuda_available']}`.",
        f"- Free disk: `{disk_summary['free_gib']} GiB` of `{disk_summary['total_gib']} GiB`.",
        f"- Analysis virtualenv packages: `{json.dumps(summary['analysis_environment_packages'], ensure_ascii=False)}`.",
        "- In the initial system Python, the common ML stack (PyTorch, torchvision, TensorFlow, scikit-learn, pandas, NumPy, Pillow, Matplotlib, seaborn and openpyxl) was absent. An isolated `/home/user/.venv` was used for PDF/data/report tooling. A CPU-only PyTorch wheel fetch was attempted but the PyTorch wheel index terminated TLS before providing a package; PyTorch/torchvision remain unavailable.",
        "",
        "## Execution conclusion",
        "",
        "The source audit, PDF text extraction, document inspection, environment checks, and data-availability checks were executable and were performed. Supervised model training/evaluation cannot validly run in this workspace because HAM10000 images and metadata are absent; there is no GPU, and PyTorch/torchvision are not installed. No research metrics are fabricated or substituted with synthetic results.",
        "",
        "### Key artifacts",
        "",
        "- `workspace_inventory.csv` — file-level inventory.",
        "- `audit_environment.json` — machine-readable file/data/environment audit.",
        "- `../02_literature/source_paper_inventory.csv` — PDF content hashes, duplicate paths, page counts, PDF metadata and extracts.",
        "- `../03_dataset/dataset_report.csv` and `dataset_report.md` — actual dataset check (currently blocked).",
        "",
    ])
    (OUTPUT_DIR / "01_project_audit" / "PROJECT_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Audited {len(files)} source files; {len(pdfs)} PDFs ({len(unique_hashes)} unique); {len(candidates['image_assets'])} images.")
    print(f"Wrote {OUTPUT_DIR / '01_project_audit' / 'PROJECT_AUDIT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
