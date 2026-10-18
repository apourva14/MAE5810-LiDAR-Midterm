"""Portable navigation data: compatible CSV summaries and lossless offline replay.

Archives contain numeric/Unicode NumPy arrays only and are always loaded with
allow_pickle=False.  JSON snapshots embed the source configuration, sensor and
scene so moving the project does not require its former absolute paths.
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import json
import math
from pathlib import Path
import re

import numpy as np

from .. import api as ls
from .engine import OccupancyGrid, build_world

FORMAT_VERSION = 1
LOG_COLUMNS = (
    "t", "x", "y", "theta", "v", "omega", "true_clearance",
    "perceived_clearance", "n_returns", "n_obstacle_pts", "dist_goal",
)
SUMMARY_COLUMNS = (
    "scenario", "alpha_per_m", "Rmax_rho0.05", "Rmax_rho0.3",
    "status", "time_s", "path_m", "min_true_clearance_m",
    "min_perceived_clearance_m", "mean_returns",
)


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _snapshot_inputs(nav):
    snapshot = copy.deepcopy(nav)
    base = snapshot.get("_base_config_snapshot")
    source = snapshot.get("base_config")
    if base is None and source is not None:
        base = ls.load_config_file(source)
    if base is not None:
        snapshot["_base_config_snapshot"] = copy.deepcopy(base)
    if source is not None:
        snapshot.setdefault("_base_config_source", str(source))
        snapshot["base_config"] = Path(source).name
    return snapshot


def _scene_snapshot(scene):
    objects, exact_state = [], []
    for obj in scene.objects:
        spec = {"label": obj.label, "material": dataclasses.asdict(obj.material)}
        exact = {}
        if isinstance(obj, ls.Plane):
            spec.update(type="plane", point=obj.p.tolist(), normal=obj.n.tolist())
            exact["normal"] = obj.n.tolist()
        elif isinstance(obj, ls.OrientedBox):
            spec.update(type="box", center=obj.c.tolist(), size=(2 * obj.h).tolist(),
                        yaw_deg=float(np.degrees(np.arctan2(obj.R[1, 0], obj.R[0, 0]))))
            exact["rotation"] = obj.R.tolist()
        elif isinstance(obj, ls.Sphere):
            spec.update(type="sphere", center=obj.c.tolist(), radius=obj.r)
        elif isinstance(obj, ls.VerticalCylinder):
            spec.update(type="cylinder", base=obj.base.tolist(), radius=obj.r, height=obj.hgt)
        else:
            raise TypeError(f"Cannot persist unsupported scene primitive {type(obj).__name__}")
        objects.append(spec)
        exact_state.append(exact)
    return objects, exact_state


def _restore_scene(metadata):
    scene = ls.build_scene({"materials": {}, "objects": metadata["scene_objects"]})
    for obj, exact in zip(scene.objects, metadata["scene_exact_state"]):
        if "normal" in exact:
            obj.n = np.asarray(exact["normal"], dtype=float)
        if "rotation" in exact:
            obj.R = np.asarray(exact["rotation"], dtype=float)
    return scene


def _pack_ragged(arrays, prefix, values, width):
    offsets = np.zeros(len(values) + 1, dtype=np.int64)
    for i, value in enumerate(values):
        offsets[i + 1] = offsets[i] + len(value)
    arrays[prefix + "_offsets"] = offsets
    arrays[prefix] = (np.concatenate([np.asarray(v).reshape(-1, width) for v in values], axis=0)
                      if values else np.empty((0, width), dtype=float))


def _slice_ragged(archive, prefix, index):
    offsets = archive[prefix + "_offsets"]
    return archive[prefix][offsets[index]:offsets[index + 1]].copy()


def _serialize_result(result):
    grid = result["grid"]
    objects, exact = _scene_snapshot(result["scene"])
    frames = result.get("frames", [])
    history = result.get("history", [])
    headings = np.asarray(result["H"])
    arrays = {
        "log": np.asarray(result["log"]), "H": headings,
        "grid_log_odds": np.asarray(grid.l),
        "frame_psi": np.asarray([f["psi"] for f in frames], dtype=float),
        "frame_L": np.asarray([f["L"] for f in frames], dtype=float).reshape(len(frames), len(headings)),
    }
    _pack_ragged(arrays, "frame_obs", [f["obs"] for f in frames], 2)
    _pack_ragged(arrays, "frame_gnd", [f["gnd"] for f in frames], 2)
    metadata = {
        "format_version": FORMAT_VERSION, "name": result["name"],
        "status": result["status"], "sensor": dataclasses.asdict(result["cfg"]),
        "scene_objects": objects, "scene_exact_state": exact,
        "first_det": result["first_det"], "frame_count": len(frames),
        "history_count": len(history),
        "grid": {"resolution_m": grid.res, "extent": [grid.x0, grid.x1, grid.y0, grid.y1],
                 "l_occ": grid.l_occ, "l_free": grid.l_free, "l_clamp": grid.l_clamp},
        "grid_available": result.get("grid_available", True),
        "clearance_reference": "robot-centre distance; subtract robot radius for surface gap",
    }
    if history:
        point_fields = [f.name for f in dataclasses.fields(history[0]["pc"])
                        if f.name not in ("label_names", "range_image", "n_beams")]
        metadata["point_fields"] = point_fields
        metadata["label_names"] = list(history[0]["pc"].label_names)
        metadata["history_status"] = [f["status"] for f in history]
        offsets = np.zeros(len(history) + 1, dtype=np.int64)
        for i, frame in enumerate(history):
            if list(frame["pc"].label_names) != metadata["label_names"]:
                raise ValueError("A navigation run cannot change its scene label ordering")
            offsets[i + 1] = offsets[i] + frame["pc"].n
        arrays["point_offsets"] = offsets
        for field in point_fields:
            arrays["pc_" + field] = np.concatenate([getattr(f["pc"], field) for f in history], axis=0)
        arrays["pc_range_image"] = np.stack([f["pc"].range_image for f in history])
        arrays["pc_n_beams"] = np.asarray([f["pc"].n_beams for f in history], dtype=np.int64)
        arrays["history_k"] = np.asarray([f["k"] for f in history], dtype=np.int64)
        arrays["history_psi"] = np.asarray([f["psi"] for f in history], dtype=float)
        arrays["history_L_valid"] = np.asarray([f["L"] is not None for f in history], dtype=bool)
        arrays["history_L"] = np.asarray(
            [f["L"] if f["L"] is not None else np.full(len(headings), np.nan) for f in history])
        arrays["history_grid_log_odds"] = np.stack([f["grid_l"] for f in history])
        _pack_ragged(arrays, "history_obs", [f["obs"] for f in history], 2)
        _pack_ragged(arrays, "history_gnd", [f["gnd"] for f in history], 2)
    return arrays, metadata


def _summary_row(result, labels):
    log, cfg = np.asarray(result["log"]), result["cfg"]
    finite_perceived = log[np.isfinite(log[:, 7]), 7]
    return [
        result["name"], f"{cfg.extinction_per_m:.4f}",
        f"{ls.max_range_estimate(cfg, 0.05):.1f}", f"{ls.max_range_estimate(cfg, 0.3):.1f}",
        result["status"], f"{log[-1, 0]:.1f}",
        f"{np.sum(np.linalg.norm(np.diff(log[:, 1:3], axis=0), axis=1)):.2f}",
        f"{log[:, 6].min():.3f}",
        f"{finite_perceived.min() if finite_perceived.size else np.nan:.3f}",
        f"{log[:, 8].mean():.0f}",
    ] + [f"{result['first_det'].get(label, np.nan):.2f}" for label in labels]


def save_results(results, nav, directory):
    """Save exact run arrays/replay history, source snapshots, and legacy CSV files.

    Return the destination Path. No simulation, plotting, or animation is performed.
    """
    results = list(results)
    if not results:
        raise ValueError("No navigation runs were supplied")
    if any(np.asarray(r["log"]).size == 0 for r in results):
        raise ValueError("Cannot save a navigation run with an empty control log")
    slugs = [re.sub(r"[^A-Za-z0-9_-]+", "_", str(r["name"])).strip("_") or "run" for r in results]
    if len(set(slugs)) != len(slugs):
        raise ValueError("Navigation scenario names produce duplicate archive filenames")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot_inputs(nav)
    _write_json(directory / "input_config.json", snapshot)
    labels = list(results[0]["first_det"])
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(SUMMARY_COLUMNS) + ["first_det_" + label for label in labels])
        for result in results:
            writer.writerow(_summary_row(result, labels))
    manifest = {"format_version": FORMAT_VERSION, "runs": []}
    for result, slug in zip(results, slugs):
        arrays, metadata = _serialize_result(result)
        archive_name, metadata_name = f"history_{slug}.npz", f"run_{slug}.json"
        if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
            raise TypeError("Object arrays are not permitted in navigation archives")
        np.savez_compressed(directory / archive_name, **arrays)
        _write_json(directory / metadata_name, metadata)
        np.savetxt(directory / f"log_{slug}.csv", result["log"], delimiter=",", fmt="%.4f",
                   header=",".join(LOG_COLUMNS), comments="")
        manifest["runs"].append({"name": result["name"], "metadata": metadata_name, "archive": archive_name})
    _write_json(directory / "runs.json", manifest)
    return directory


def _load_run(directory, entry, nav):
    metadata = json.loads((directory / entry["metadata"]).read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported navigation archive version in {entry['metadata']}")
    with np.load(directory / entry["archive"], allow_pickle=False) as archive:
        grid = OccupancyGrid(metadata["grid"])
        grid.l = archive["grid_log_odds"].copy()
        frames = [
            dict(obs=_slice_ragged(archive, "frame_obs", i),
                 gnd=_slice_ragged(archive, "frame_gnd", i),
                 L=archive["frame_L"][i].copy(), psi=float(archive["frame_psi"][i]))
            for i in range(metadata["frame_count"])
        ]
        history = []
        for i in range(metadata["history_count"]):
            start, end = archive["point_offsets"][i:i + 2]
            fields = {field: archive["pc_" + field][start:end].copy() for field in metadata["point_fields"]}
            fields.update(label_names=list(metadata["label_names"]),
                          range_image=archive["pc_range_image"][i].copy(),
                          n_beams=int(archive["pc_n_beams"][i]))
            history.append(dict(
                k=int(archive["history_k"][i]), pc=ls.PointCloud(**fields),
                obs=_slice_ragged(archive, "history_obs", i),
                gnd=_slice_ragged(archive, "history_gnd", i),
                L=archive["history_L"][i].copy() if archive["history_L_valid"][i] else None,
                psi=float(archive["history_psi"][i]), grid_l=archive["history_grid_log_odds"][i].copy(),
                status=metadata["history_status"][i]))
        result = dict(
            name=metadata["name"], cfg=ls.LidarConfig(**metadata["sensor"]), scene=_restore_scene(metadata),
            log=archive["log"].copy(), H=archive["H"].copy(), grid=grid,
            first_det={key: np.nan if value is None else value for key, value in metadata["first_det"].items()},
            status=metadata["status"], frames=frames, history=history,
            replay_available=bool(history), grid_available=metadata["grid_available"],
            historical_csv_only=False, input_config=copy.deepcopy(nav), source_directory=str(directory))
    return result


def _load_legacy(directory, nav):
    if nav is None:
        raise ValueError("CSV-only navigation data require nav with a resolved base_config for static plotting")
    summary_path = directory / "summary.csv"
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = [{"scenario": path.stem[4:], "status": "unknown"} for path in sorted(directory.glob("log_*.csv"))]
    if not rows:
        raise FileNotFoundError(f"No navigation archives or CSV logs found in {directory}")
    results = []
    for row in rows:
        name = row["scenario"]
        path = directory / f"log_{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Navigation log missing for scenario {name!r}: {path}")
        log = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        scenario = next((s for s in nav["scenarios"] if s["name"] == name), None)
        if scenario is None:
            raise ValueError(f"No configuration for historical scenario {name!r}")
        cfg, scene = build_world(nav, scenario)
        first_det = {key[len("first_det_"):]: float(value) for key, value in row.items()
                     if key.startswith("first_det_")}
        results.append(dict(
            name=name, cfg=cfg, scene=scene, log=log,
            H=np.linspace(-np.pi, np.pi, nav["planner"]["n_headings"], endpoint=False),
            grid=OccupancyGrid(nav["grid"]), first_det=first_det,
            status=row.get("status", "unknown"), frames=[], history=[],
            replay_available=False, grid_available=False, historical_csv_only=True,
            input_config=copy.deepcopy(nav), source_directory=str(directory), summary=dict(row)))
    return results


def load_results(directory, nav=None):
    """Load full run archives, or historical CSV-only logs for static plotting.

    Return (results, resolved_navigation_config). Archives use embedded input
    snapshots. Older CSV-only data use the supplied matching configuration, or
    the project's default navigation configuration when no snapshot is present.
    """
    directory = Path(directory)
    snapshot_path = directory / "input_config.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.exists() else nav
    manifest_path = directory / "runs.json"
    if snapshot is None:
        from ..paths import DEFAULT_NAVIGATION_CONFIG
        config_path = Path(DEFAULT_NAVIGATION_CONFIG)
        snapshot = ls.load_config_file(config_path)
        snapshot["base_config"] = str((config_path.parent / snapshot["base_config"]).resolve())
    if not manifest_path.exists():
        return _load_legacy(directory, snapshot), snapshot
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported navigation manifest version in {manifest_path}")
    return [_load_run(directory, entry, snapshot) for entry in manifest["runs"]], snapshot


def require_frames(result):
    """Return animation frames, or explain why historical logs cannot animate."""
    frames = result.get("frames", [])
    if not frames:
        raise ValueError(
            f"Run {result.get('name', '')!r} has no saved per-scan animation frames. "
            "CSV-only logs support static plots; animation requires saved numerical history.")
    return frames


def require_history(result):
    """Return complete step history needed by first-person or map replay."""
    history = result.get("history", [])
    if not history:
        raise ValueError(
            f"Run {result.get('name', '')!r} has no saved full step history. "
            "CSV-only logs cannot reconstruct point clouds or occupancy-map evolution.")
    return history
