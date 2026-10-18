"""Allowlisted, reproducible deliverable workflow used by the local dashboard.

Numerical generation and presentation remain separate CLI subprocesses. Existing
studies are audited before first reuse; later runs compare content fingerprints.
An imported study records that its original source revision was not recorded.
Changing numerical code never silently triggers an expensive replacement run.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable
import zipfile

from .paths import PROJECT_ROOT


class WorkflowError(RuntimeError):
    """A requested generation stage could not produce verified outputs."""


@dataclass(frozen=True)
class Task:
    id: str
    label: str
    kind: str
    dependencies: tuple[str, ...]
    args: tuple[str, ...]
    outputs: tuple[str, ...]
    sources: tuple[str, ...] = ()
    data_roots: tuple[str, ...] = ()
    description: str = ""


def _tasks() -> tuple[Task, ...]:
    sensor = ("lidar_project/model", "lidar_project/api.py", "lidar_project/config.py",
              "lidar_project/data.py", "configs/sensor.jsonc")
    motion = sensor + ("lidar_project/trajectory", "lidar_project/mapping",
                       "lidar_project/navigation/engine.py", "configs/trajectory.jsonc")
    return (
        Task("scan", "Example 3D LiDAR scan", "data", (),
             ("run.py", "scan", "--config", "configs/sensor.jsonc", "--data-dir", "data/scan"),
             ("data/scan/pointcloud.npz", "data/scan/pointcloud.csv", "data/scan/pointcloud.ply", "data/scan/config_used.json"),
             sensor, ("data/scan",), "Generate one scene's point cloud and configuration snapshot."),
        Task("render_scan", "Point cloud figures and 3D viewer", "media", ("scan",),
             ("run.py", "render-scan", "--data-dir", "data/scan", "--output-dir", "outputs/scan"),
             ("outputs/scan/pointcloud.png", "outputs/scan/pointcloud_interactive.html",
              "outputs/scan/report_figures/range_image.png", "outputs/scan/report_figures/plan_view.png",
              "outputs/scan/report_figures/return_counts.png", "outputs/scan/report_figures/object_cloud.png"),
             ("lidar_project/visualization/scan_figures.py", "lidar_project/visualization/scan_viewer.py")),
        Task("evaluation", "Controlled geometry and response studies", "data", (),
             ("run.py", "evaluate", "--data-dir", "data/evaluation"),
             ("data/evaluation/evaluation.json", "data/evaluation/geometry/summary.json",
              "data/evaluation/geometry/frustum_geometry.npz", "data/evaluation/geometry/cases.csv",
              "data/evaluation/response/summary.json", "data/evaluation/response/trials.npz",
              "data/evaluation/response/laws.csv", "data/evaluation/response/detection.csv"),
             sensor + ("lidar_project/experiments/geometry.py", "lidar_project/experiments/response.py", "lidar_project/experiments/runner.py"),
             ("data/evaluation",), "Analytical geometry, optical response and 54,000 uncertainty trials."),
        Task("render_evaluation", "Controlled-study figures", "media", ("evaluation",),
             ("run.py", "render-evaluation", "--data-dir", "data/evaluation", "--output-dir", "outputs/evaluation"),
             tuple("outputs/evaluation/" + name + ".png" for name in
                   ("fov_volume", "geometry_pose_fov", "los_comparison", "response_laws", "detection_uncertainty")),
             ("lidar_project/visualization/evaluation.py",)),
        Task("trajectory", "Route and tracking factor study", "data", (),
             ("run.py", "evaluate-trajectory", "--config", "configs/trajectory.jsonc", "--data-dir", "data/trajectory"),
             ("data/trajectory/summary.json", "data/trajectory/manifest.json", "data/trajectory/runs.csv",
              "data/trajectory/runs/baseline/seed_1/trajectory.npz"),
             motion + ("lidar_project/experiments/trajectory.py",), ("data/trajectory",),
             "Generate planned routes and retain all 21 tracking-study outcomes."),
        Task("render_trajectory", "Trajectory figures, animation and viewer", "media", ("trajectory",),
             ("run.py", "render-trajectory", "--data-dir", "data/trajectory", "--output-dir", "outputs/trajectory", "--gifs"),
             ("outputs/trajectory/trajectory_report.png", "outputs/trajectory/trajectory_overview.png",
              "outputs/trajectory/trajectory_motion.gif", "outputs/trajectory/trajectory_interactive.html"),
             ("lidar_project/visualization/trajectory.py",)),
        Task("mapping", "Moving 3D maps in three environments", "data", (),
             ("run.py", "map-environments", "--config", "configs/trajectory.jsonc", "--data-dir", "data/mapping"),
             ("data/mapping/comparison.json", "data/mapping/runs.csv",
              "data/mapping/runs/open/seed_1/mapping.npz", "data/mapping/runs/cluttered/seed_1/mapping.npz",
              "data/mapping/runs/narrow/seed_1/mapping.npz"),
             motion + ("lidar_project/experiments/environments.py", "lidar_project/experiments/trajectory.py"), ("data/mapping",),
             "Retain full scans, trajectories and endpoint maps for nine runs."),
        Task("render_mapping", "Moving 3D mapping viewer and comparisons", "media", ("mapping",),
             ("run.py", "render-mapping", "--data-dir", "data/mapping", "--output-dir", "outputs/mapping"),
             ("outputs/mapping/mapping_interactive.html", "outputs/mapping/mapping_3d_comparison.png",
              "outputs/mapping/mapping_comparison.png", "outputs/mapping/trajectory_3d_report.png"),
             ("lidar_project/visualization/mapping.py", "lidar_project/visualization/mapping_report.py")),
        Task("dependencies", "Navigation sensor-dependency experiments", "data", (),
             ("run.py", "evaluate-navigation-dependencies", "--config", "configs/trajectory.jsonc", "--data-dir", "data/navigation_dependencies"),
             ("data/navigation_dependencies/summary.json", "data/navigation_dependencies/manifest.json",
              "data/navigation_dependencies/runs.csv", "data/navigation_dependencies/replay.csv",
              "data/navigation_dependencies/reference_poses_seed_1.npz"),
             motion + ("lidar_project/experiments/navigation_dependencies.py", "lidar_project/experiments/environments.py",
                       "lidar_project/experiments/trajectory.py"), ("data/navigation_dependencies",),
             "22 conditions, 66 closed-loop runs and 66 matched-viewpoint replays."),
        Task("render_dependencies", "Navigation dependency figures and viewer", "media", ("dependencies",),
             ("run.py", "render-navigation-dependencies", "--data-dir", "data/navigation_dependencies", "--output-dir", "outputs/navigation_dependencies"),
             ("outputs/navigation_dependencies/navigation_dependencies.html",
              "outputs/navigation_dependencies/navigation_dependency_measurements.png",
              "outputs/navigation_dependencies/navigation_dependency_outcomes.png",
              "outputs/navigation_dependencies/navigation_dependency_visibility.png",
              "outputs/navigation_dependencies/report_visibility_navigation.png",
              "outputs/navigation_dependencies/report_physical_dependencies.png",
              "outputs/navigation_dependencies/report_navigation_maps.png"),
             ("lidar_project/visualization/navigation_dependencies.py",
              "lidar_project/visualization/navigation_dependency_report.py")),
        Task("coverage", "3D field-of-view and occluded-ray geometry", "data", ("dependencies",),
             ("run.py", "prepare-navigation-coverage", "--source-dir", "data/navigation_dependencies", "--data-dir", "data/navigation_validation"),
             ("data/navigation_validation/fov_los.json",),
             sensor + ("lidar_project/experiments/navigation_coverage.py",), ("data/navigation_validation",)),
        Task("threshold", "Near-threshold detection trials", "data", ("dependencies",),
             ("run.py", "evaluate-navigation-threshold", "--source-dir", "data/navigation_dependencies", "--data-dir", "data/navigation_threshold", "--trials", "1000"),
             ("data/navigation_threshold/summary.json", "data/navigation_threshold/design.json", "data/navigation_threshold/settings.csv"),
             sensor + ("lidar_project/experiments/navigation_threshold.py",), ("data/navigation_threshold",),
             "33,000 independent sensor trials at three frozen navigation viewpoints."),
        Task("robustness", "Multi-layout navigation robustness", "data", (),
             ("run.py", "evaluate-navigation-robustness", "--config", "configs/trajectory.jsonc", "--data-dir", "data/navigation_robustness"),
             ("data/navigation_robustness/summary.json", "data/navigation_robustness/manifest.json", "data/navigation_robustness/runs.csv"),
             motion + ("lidar_project/experiments/navigation_robustness.py", "lidar_project/experiments/environments.py",
                       "lidar_project/experiments/trajectory.py"),
             ("data/navigation_robustness",), "720 runs: 12 sensing conditions, three fixed layouts and 20 noise seeds."),
        Task("diagnostics", "Navigation failure explanations", "analysis", ("robustness",),
             ("run.py", "diagnose-navigation-failures", "--data-dir", "data/navigation_robustness"),
             ("data/navigation_robustness/failure_diagnostics.json",),
             ("lidar_project/experiments/navigation_robustness.py",)),
        Task("render_validation", "3D visibility, detection and robustness evidence", "media", ("coverage", "threshold", "diagnostics"),
             ("run.py", "render-navigation-validation", "--diagram", "data/navigation_validation/fov_los.json",
              "--threshold-dir", "data/navigation_threshold", "--robustness-dir", "data/navigation_robustness",
              "--output-dir", "outputs/navigation_validation"),
             ("outputs/navigation_validation/report_fov_los_3d.png", "outputs/navigation_validation/report_threshold_detection.png",
              "outputs/navigation_validation/report_navigation_robustness.png", "outputs/navigation_validation/navigation_validation.html"),
             ("lidar_project/visualization/navigation_validation.py",)),
        Task("report_docx", "Editable course report", "report", ("render_evaluation", "render_dependencies", "render_validation"),
             ("tools/export_report.py", "--output-dir", "outputs/reports", "--format", "docx", "--use-saved-figures"),
             ("outputs/reports/Midterm_Project_Report.docx",), ("tools/build_report.py", "tools/export_report.py")),
        Task("report_pdf", "PDF course report", "report", ("report_docx",),
             ("tools/export_report.py", "--output-dir", "outputs/reports", "--format", "pdf"),
             ("outputs/reports/Midterm_Project_Report.pdf", "outputs/reports/report_metadata.json"),
             ("tools/export_report.py",)),
        Task("package", "Submission archive", "package",
             ("render_scan", "render_evaluation", "render_trajectory", "render_mapping", "render_dependencies", "render_validation", "report_pdf"),
             ("tools/package_project.py", "--output", "outputs/submission/Midterm_Project_Submission.zip"),
             ("outputs/submission/Midterm_Project_Submission.zip",),
             ("tools/package_project.py", "README.md", "index.html", "start.py", "pyproject.toml", "tests", "configs", "lidar_project")),
    )


TASKS = _tasks()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Workflow:
    """Execute only declared stages, with array arguments and no shell evaluation."""

    def __init__(self, project_root=None, *, python=None,
                 event_callback: Callable[[dict], None] | None = None, tasks=None):
        self.project_root = Path(project_root or PROJECT_ROOT).expanduser().resolve()
        self.python = str(Path(python or sys.executable).resolve())
        self.event_callback = event_callback or (lambda event: None)
        self.tasks = {task.id: task for task in (TASKS if tasks is None else tasks)}
        self.build_dir = self.project_root / "build"
        self.manifest_path = self.build_dir / "manifest.json"
        self._lock = threading.Lock()
        self._manifest = {"format_version": 1, "tasks": {}}
        if self.manifest_path.exists():
            try:
                value = _read(self.manifest_path)
                if value.get("format_version") == 1:
                    self._manifest = value
            except (OSError, ValueError):
                pass

    def _path(self, relative) -> Path:
        path = (self.project_root / relative).resolve()
        if not path.is_relative_to(self.project_root):
            raise WorkflowError(f"Path escapes the project folder: {relative}")
        return path

    def task_catalog(self) -> list[dict]:
        return [dict(id=t.id, label=t.label, kind=t.kind, dependencies=list(t.dependencies),
                     description=t.description,
                     outputs=[dict(path=p, label=Path(p).name, exists=self._path(p).is_file()) for p in t.outputs])
                for t in self.tasks.values()]

    def status(self) -> list[dict]:
        """Cheap dashboard state; a run performs the complete archive/content audit."""
        rows = self.task_catalog()
        for row in rows:
            record = self._manifest.get("tasks", {}).get(row["id"])
            present = all(item["exists"] for item in row["outputs"])
            row["status"] = "ready" if present else "missing"
            row["verified"] = bool(record and record.get("status") == "complete" and present)
            row["updated_utc"] = record.get("finished_utc") if record else None
            if record and present:
                for item in record.get("outputs", []):
                    path = self._path(item["path"])
                    if not path.exists() or path.stat().st_size != item["bytes"] or path.stat().st_mtime_ns != item["mtime_ns"]:
                        row["status"], row["verified"] = "stale", False
                        break
                try:
                    fingerprint, _ = self._input_fingerprint(self.tasks[row["id"]])
                    if fingerprint != record.get("input_fingerprint"):
                        row["status"], row["verified"] = "stale", False
                except WorkflowError:
                    row["status"], row["verified"] = "stale", False
        return rows

    def _event(self, event_type, task, **kwargs):
        self.event_callback(dict(type=event_type, task=task.id, label=task.label, time_utc=_utc(), **kwargs))

    def _write_manifest(self):
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self._manifest.update(project=str(self.project_root), python=self.python, updated_utc=_utc(),
                              interpretation="Model consistency and simulated task outcomes; no physical calibration claim.")
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    def _files(self, relative_paths) -> list[Path]:
        files = set()
        for relative in relative_paths:
            path = self._path(relative)
            if path.is_dir():
                files.update(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                             and p.suffix not in {".pyc", ".tmp"})
            elif path.is_file():
                files.add(path)
        return sorted(files)

    def _inventory(self, paths, previous=()) -> list[dict]:
        previous = {item["path"]: item for item in previous}
        rows = []
        for path in self._files(paths):
            relative = path.relative_to(self.project_root).as_posix()
            stat = path.stat()
            old = previous.get(relative)
            digest = old["sha256"] if old and old["bytes"] == stat.st_size and old["mtime_ns"] == stat.st_mtime_ns else _sha(path)
            rows.append(dict(path=relative, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest))
        return rows

    def _input_fingerprint(self, task):
        sources = task.sources + (("lidar_project/visualization/default_theme.json",
                                   "lidar_project/visualization/theme.py",
                                   "lidar_project/visualization/templates",
                                   "lidar_project/visualization/static") if task.kind == "media" else ())
        inventory = self._inventory(sources)
        dependencies = {}
        for name in task.dependencies:
            record = self._manifest["tasks"].get(name)
            if not record or record.get("status") != "complete":
                raise WorkflowError(f"Dependency {name} did not complete")
            dependencies[name] = record["output_fingerprint"]
        payload = dict(command=[self.python, *task.args], source_files=[(i["path"], i["sha256"]) for i in inventory], dependencies=dependencies)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return digest, payload

    def _owned_files(self, task):
        paths = task.data_roots if task.kind == "data" else task.outputs
        files = self._files(paths)
        # This diagnostic is a downstream derivative stored next to raw runs;
        # regenerating it must not make the parent numerical study appear edited.
        if task.id == "robustness":
            files = [p for p in files if p.name != "failure_diagnostics.json"]
        return tuple(p.relative_to(self.project_root).as_posix() for p in files)

    def _validate(self, task):
        missing = [p for p in task.outputs if not self._path(p).is_file() or self._path(p).stat().st_size == 0]
        if missing:
            raise WorkflowError("Missing required output: " + ", ".join(missing))
        # Audit the raw archives, not only convenient summary files. CRC failures
        # or a truncated JSON block prevent adopting a study as a valid cache.
        audit_paths = list(task.data_roots or task.outputs)
        for path in self._files(audit_paths):
            if path.suffix in {".json", ".jsonc"} and path.suffix != ".jsonc":
                _read(path)
            elif path.suffix in {".npz", ".docx", ".zip"}:
                with zipfile.ZipFile(path) as archive:
                    bad = archive.testzip()
                    if bad is not None or not archive.namelist():
                        raise WorkflowError(f"Invalid archive: {path.name}")
        data = self.project_root / "data"
        if task.id == "dependencies":
            value = _read(data / "navigation_dependencies/summary.json")
            if len(value.get("cases", [])) != 22 or len(value.get("runs", [])) != 66 or len(value.get("replays", [])) != 66:
                raise WorkflowError("Navigation dependency study is incomplete; require 22 conditions and 66 runs/replays")
            for row in value["runs"]:
                self._require_run(data / "navigation_dependencies", row["run_directory"])
            for row in value["cases"]:
                self._require_run(data / "navigation_dependencies", row["representative_run"], mapping=True)
                self._require_files(data / "navigation_dependencies" / row["representative_replay"], ("replay.json", "traces.csv"))
        elif task.id == "threshold":
            root = data / "navigation_threshold"
            value, design = _read(root / "summary.json"), _read(root / "design.json")
            if value.get("trials_per_setting") != 1000 or len(value.get("settings", [])) != 33 or value["aggregate"]["trials"] != 33000:
                raise WorkflowError("Threshold study is incomplete; require 33 settings and 33,000 trials")
            for row in value["settings"]:
                self._require_files(root / "trials", (f"setting_{row['setting_id']:02d}.npz",))
            source = data / "navigation_dependencies"
            if design.get("source_manifest_sha256") != _sha(source / "manifest.json") or design.get("source_pose_sha256") != _sha(source / "reference_poses_seed_1.npz"):
                raise WorkflowError("Threshold design does not match the saved navigation viewpoints")
        elif task.id == "robustness":
            root = data / "navigation_robustness"
            value = _read(root / "summary.json")
            if not value.get("complete") or value.get("runs") != 720 or value.get("planned_runs") != 720 or len(value.get("run_metrics", [])) != 720:
                raise WorkflowError("Robustness study is incomplete; require all 720 runs")
            for row in value["run_metrics"]:
                self._require_run(root, row["run_directory"], mapping=bool(row.get("representative_full_scans")))
        elif task.id == "trajectory":
            value = _read(data / "trajectory/summary.json")
            if value.get("runs") != 21 or value.get("case_count") != 7:
                raise WorkflowError("Trajectory factor study is incomplete")
            for case in value["cases"]:
                for seed in value["seeds"]:
                    self._require_run(data / "trajectory", f"runs/{case['case_id']}/seed_{seed}")
        elif task.id == "mapping":
            value = _read(data / "mapping/comparison.json")
            if len(value.get("cases", [])) != 3 or len(value.get("runs", [])) != 9:
                raise WorkflowError("Environment comparison is incomplete")
            for row in value["runs"]:
                self._require_run(data / "mapping", row["run_directory"], mapping=True)
        elif task.id == "evaluation":
            value = _read(data / "evaluation/response/summary.json")
            if value["detection"]["independent_trials"] != 54000:
                raise WorkflowError("Controlled response trials are incomplete")
        elif task.id == "diagnostics":
            value = _read(data / "navigation_robustness/failure_diagnostics.json")
            if value.get("summary_sha256") != _sha(data / "navigation_robustness/summary.json"):
                raise WorkflowError("Failure diagnostics belong to a different robustness summary")

    def _require_files(self, directory, names):
        directory = directory.resolve()
        if not directory.is_relative_to(self.project_root):
            raise WorkflowError("Saved study references a file outside the project")
        for name in names:
            path = directory / name
            if not path.is_file() or not path.stat().st_size:
                raise WorkflowError(f"Incomplete saved study: {path.relative_to(self.project_root)}")

    def _require_run(self, root, relative, mapping=False):
        self._require_files(root / relative, ("trajectory.json", "trajectory.npz"))
        if mapping:
            self._require_files(root / relative, ("mapping.json", "mapping.npz"))

    def _backup(self, task, run_id):
        moved = []
        for relative in task.data_roots:
            source = self._path(relative)
            if source.exists():
                destination = self.build_dir / "backups" / run_id / task.id / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                moved.append((source, destination))
        return moved

    def _restore(self, moved, run_id, task):
        for source, backup in moved:
            if source.exists():
                failed = self.build_dir / "failed" / run_id / task.id / source.relative_to(self.project_root)
                failed.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(failed))
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup), str(source))

    def _execute(self, task, log_path):
        environment = os.environ.copy()
        environment.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        environment.setdefault("MPLCONFIGDIR", str(self.build_dir / "matplotlib"))
        command = [self.python, *task.args]
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=self.project_root, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", shell=False)
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    self._event("log", task, line=line.rstrip(), message=line.rstrip())
            code = process.wait()
        if code:
            raise WorkflowError(f"{task.label} failed (exit {code}); see {log_path.relative_to(self.project_root)}")

    def run(self, task_id="all", fresh=False) -> dict:
        if not isinstance(task_id, str) or (task_id != "all" and task_id not in self.tasks):
            raise WorkflowError("Unknown workflow task")
        if not isinstance(fresh, bool):
            raise WorkflowError("fresh must be true or false")
        if not self._lock.acquire(blocking=False):
            raise WorkflowError("A workflow is already running")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        started = time.monotonic()
        completed = set()
        visiting = set()
        results = []
        logs = self.build_dir / "logs" / run_id
        logs.mkdir(parents=True, exist_ok=True)
        try:
            def visit(name):
                if name in completed:
                    return
                if name in visiting:
                    raise WorkflowError("Workflow dependencies contain a cycle")
                visiting.add(name)
                task = self.tasks[name]
                for dependency in task.dependencies:
                    visit(dependency)
                visiting.remove(name)
                fingerprint, inputs = self._input_fingerprint(task)
                record = self._manifest["tasks"].get(name, {})
                present = all(self._path(path).is_file() for path in task.outputs)
                # Saved measurements can be adopted after the full design/raw
                # archive audit. Presentations need a recorded source match;
                # their first build regenerates rather than blessing an old view.
                should_reuse = (present and not (fresh and task.kind == "data")
                                and (task.kind == "data" or bool(record)))
                if should_reuse:
                    try:
                        self._validate(task)
                    except (OSError, ValueError, zipfile.BadZipFile, WorkflowError) as exc:
                        if task.kind == "data":
                            raise WorkflowError(f"Saved {task.label} failed validation: {exc}. Choose a fresh rebuild to replace it; prior data will be archived.") from exc
                        should_reuse = False
                if should_reuse and record and record.get("input_fingerprint") != fingerprint:
                    if task.kind == "data":
                        raise WorkflowError(f"Sources or upstream data for {task.label} changed. Choose a fresh rebuild to regenerate the numerical study.")
                    should_reuse = False
                old_inventory = record.get("outputs", [])
                owned = self._owned_files(task)
                inventory = self._inventory(owned, old_inventory) if should_reuse else []
                if should_reuse and record and [(p["path"], p["sha256"]) for p in inventory] != [(p["path"], p["sha256"]) for p in old_inventory]:
                    if task.kind == "data":
                        raise WorkflowError(f"Saved data for {task.label} changed after its recorded build. Choose a fresh rebuild after reviewing the changes.")
                    should_reuse = False
                moved = []
                self._event("stage_reused" if should_reuse else "stage_started", task,
                            message="Validated saved outputs" if should_reuse else "Generating required outputs")
                if not should_reuse:
                    if task.kind == "data":
                        moved = self._backup(task, run_id)
                    try:
                        self._execute(task, logs / (name + ".log"))
                        self._validate(task)
                    except Exception as exc:
                        self._restore(moved, run_id, task)
                        self._event("stage_failed", task, message=str(exc))
                        raise
                    inventory = self._inventory(self._owned_files(task))
                output_fingerprint = hashlib.sha256(json.dumps([(i["path"], i["sha256"]) for i in inventory]).encode()).hexdigest()
                provenance = record.get("provenance") if should_reuse and record else (
                    "imported_saved_outputs: original source revision unavailable; raw files and design audited at adoption"
                    if should_reuse else "generated_by_recorded_command")
                self._manifest["tasks"][name] = dict(status="complete", label=task.label, kind=task.kind,
                    input_fingerprint=fingerprint, inputs=inputs, output_fingerprint=output_fingerprint,
                    outputs=inventory, finished_utc=_utc(), provenance=provenance,
                    reused=should_reuse, backups=[str(b.relative_to(self.project_root)) for _, b in moved])
                self._write_manifest()
                completed.add(name)
                results.append(dict(task=name, reused=should_reuse, output_fingerprint=output_fingerprint))
                if not should_reuse:
                    self._event("stage_completed", task, message="Outputs generated and verified")

            targets = tuple(self.tasks) if task_id == "all" else (task_id,)
            for target in targets:
                visit(target)
            result = dict(task=task_id, fresh=fresh, run_id=run_id, status="complete", stages=results,
                          elapsed_s=time.monotonic() - started, manifest=str(self.manifest_path))
            self._manifest["last_run"] = result
            self._write_manifest()
            return result
        except Exception as exc:
            self._manifest["last_run"] = dict(task=task_id, fresh=fresh, run_id=run_id,
                                              status="failed", error=str(exc), stages=results)
            self._write_manifest()
            raise
        finally:
            self._lock.release()
