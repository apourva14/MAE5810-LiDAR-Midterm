"""Create a portable source, report and media package with a verified manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("configs", "lidar_project", "tools", "tests")
SOURCE_FILES = ("README.md", "pyproject.toml", "run.py", "start.py", "index.html", "builder.html",
                "Start Project.command", ".gitignore")
EXCLUDE_PARTS = {"__pycache__", ".venv", ".pytest_cache", ".artifacts", "build", "dist"}
MEDIA_DIRS = ("scan", "evaluation", "trajectory", "mapping", "navigation_dependencies",
              "navigation_validation", "reports")


def collect_files(project: Path):
    """Use a fixed ownership list; archives, caches and unrelated files stay out."""
    project = project.resolve()
    selected = {project / name for name in SOURCE_FILES}
    for name in SOURCE_DIRS:
        selected.update((project/name).rglob("*"))
    for name in MEDIA_DIRS:
        selected.update((project/"outputs"/name).rglob("*"))
    for file in (project/"data").rglob("*"):
        # Keep numerical evidence small. Full navigation archives regenerate from code.
        if file.suffix in (".json", ".csv") or (
                file.suffix == ".npz" and "navigation_threshold" in file.parts):
            selected.add(file)
    valid = []
    for file in sorted(selected):
        if not file.is_file():
            continue
        relative = file.relative_to(project)
        if any(part in EXCLUDE_PARTS for part in relative.parts):
            continue
        if file.suffix in (".pyc", ".pyo") or file.name.startswith(".") and file.name != ".gitignore":
            continue
        if "preview" in file.stem or "screenshot" in file.stem:
            continue
        try:
            file.resolve().relative_to(project)
        except ValueError:
            raise ValueError(f"A package input links outside the project: {relative}")
        valid.append(file)
    return valid


def build_package(project: Path = PROJECT, output: Path | None = None):
    project = Path(project).resolve()
    output = Path(output or project/"outputs/submission/Midterm_Project_Submission.zip").resolve()
    report_dir = project/"outputs/reports"
    required = [project/"index.html", project/"start.py", report_dir/"Midterm_Project_Report.docx",
                report_dir/"Midterm_Project_Report.pdf",
                project/"outputs/navigation_validation/navigation_validation.html",
                project/"outputs/navigation_dependencies/navigation_dependencies.html",
                project/"outputs/mapping/mapping_interactive.html"]
    missing = [str(path.relative_to(project)) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Generate the required files before packaging: " + ", ".join(missing))
    sys.path.insert(0, str(project/"tools"))
    from export_report import validate_pdf, docx_metadata
    pages = validate_pdf(report_dir/"Midterm_Project_Report.pdf")
    report = docx_metadata(report_dir/"Midterm_Project_Report.docx")
    files = collect_files(project)
    entries = []
    for file in files:
        relative = file.relative_to(project)
        archive_path = Path("evidence")/relative if relative.parts[0] == "data" else relative
        entries.append({"path": archive_path.as_posix(), "bytes": file.stat().st_size,
                        "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    manifest = {"format_version": 1, "title": "3D LiDAR midterm project",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "report": {"pages": pages, "editable_equations": report["equations"], "figures": report["figures"]},
                "launch": "python start.py", "files": entries,
                "included": ["source code, configurations, tests and setup guide", "editable report and page-checked PDF",
                             "scientific figures and offline interactive media", "CSV/JSON evidence and raw threshold trials"],
                "omitted": ["large raw navigation scan/history NPZ archives, caches, local logs and backup directories"],
                "reproduction": "Open index.html to explore the saved report offline. For regeneration, install the dependencies in README.md, start the local server, and open /builder. Numerical evidence is stored under evidence/data for review. The live data folder is regenerated from the project configurations and fixed seeds.",
                "limitations": "The sensor coefficients remain illustrative. Simulated model consistency does not establish physical LiDAR calibration."}
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_file = output.parent/"deliverables_manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2)+"\n").encode("utf-8")
    temporary = output.with_suffix(".zip.tmp")
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=6) as archive:
            for file, row in zip(files, entries):
                archive.write(file, "LiDAR_Midterm/"+row["path"])
            archive.writestr("LiDAR_Midterm/deliverables_manifest.json", manifest_bytes)
        with ZipFile(temporary) as archive:
            for row in entries:
                if hashlib.sha256(archive.read("LiDAR_Midterm/"+row["path"])).hexdigest() != row["sha256"]:
                    raise ValueError("Package verification failed: "+row["path"])
        temporary.replace(output)
        manifest_file.write_bytes(manifest_bytes)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"zip": str(output), "manifest": str(manifest_file), "files": len(entries), "bytes": output.stat().st_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_package(output=args.output)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Package build failed: {exc}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
