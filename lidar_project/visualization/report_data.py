"""Compact presentation data, assembled exclusively from saved study results.

This adapter never imports the sensor, planner, controller or map builder. The
interactive report replays recorded poses and first-seen measured endpoints;
it does not create measurements or rerun an experiment when a view changes.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source(path, project):
    path = Path(path)
    return {"path": path.relative_to(project).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def _display_array(array, decimals=5):
    """Round display coordinates only; all reported numeric metrics stay exact."""
    return np.round(np.asarray(array, float), decimals).tolist()


def _mapping_payload(directory, *, max_points=16_000):
    """Read measured representatives and their original first-observation steps.

    Sampling is deterministic in stored voxel order, without modifying or
    inventing any endpoint. Full map metrics are computed from all saved voxels
    and saved first-seen endpoint cells, independently of display sampling.
    """
    directory = Path(directory)
    metadata = _read(directory / "mapping.json")
    trajectory = _read(directory / "trajectory.json")
    with np.load(directory / "mapping.npz", allow_pickle=False) as source:
        xyz = source["map_xyz"]
        first = source["map_first_seen"]
        ground = source["ground_first_seen"]
        obstacle = source["obstacle_first_seen"]
    with np.load(directory / "trajectory.npz", allow_pickle=False) as source:
        state_t, control_t = source["states_t"], source["controls_t"]
        xy, theta = source["states_xy"], source["states_theta"]
        reference = source["reference_xy"]
        command_v, command_omega = source["controls_v"], source["controls_omega"]
        returns = source["controls_n_returns"]
        controller_points = source["controls_n_obstacle_points"]
        surface_gap = source["controls_swept_surface_gap"]

    if max_points < 1:
        raise ValueError("max_points must be positive")
    if len(state_t) != len(control_t) + 1:
        raise ValueError("Recorded mapping needs one initial/final state and each control state")
    if len(first) != len(xyz) or (len(first) and (first.min() < 0 or first.max() >= len(control_t))):
        raise ValueError("Measured endpoints have invalid first-observation indices")
    if not np.allclose(control_t, state_t[:-1], atol=1e-10, rtol=0):
        raise ValueError("Stored scan/control times do not align with recorded poses")

    indices = (np.arange(len(xyz)) if len(xyz) <= max_points else
               np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64))
    rounded = _display_array(xyz[indices])
    points = [point + [int(step)] for point, step in zip(rounded, first[indices])]
    curve = []
    for step, time_s in enumerate(control_t):
        cells = (((ground >= 0) & (ground <= step)) |
                 ((obstacle >= 0) & (obstacle <= step)))
        observed = int(np.count_nonzero(cells))
        curve.append({"step": step, "time_s": float(time_s),
                      "points": int(np.count_nonzero(first <= step)),
                      "observed_cells": observed,
                      "observed_fraction": observed / cells.size,
                      "returns": int(returns[step]),
                      "controller_points": int(controller_points[step])})

    dt = np.diff(state_t)
    expected_xy = xy[:-1] + dt[:, None] * command_v[:, None] * np.column_stack((np.cos(theta[:-1]), np.sin(theta[:-1])))
    expected_theta = theta[:-1] + dt * command_omega
    angle_residual = np.arctan2(np.sin(theta[1:] - expected_theta), np.cos(theta[1:] - expected_theta))
    robot = trajectory["config"]["robot"]
    motion_check = {
        "control_intervals": len(control_t),
        "max_recorded_euler_position_residual_m": float(np.max(np.linalg.norm(xy[1:] - expected_xy, axis=1), initial=0.)),
        "max_recorded_euler_heading_residual_rad": float(np.max(np.abs(angle_residual), initial=0.)),
        "speed_limit_mps": float(robot["v_max"]),
        "max_abs_commanded_speed_mps": float(np.max(np.abs(command_v), initial=0.)),
        "angular_speed_limit_rad_s": float(np.radians(robot["omega_max_deg_s"])),
        "max_abs_commanded_angular_speed_rad_s": float(np.max(np.abs(command_omega), initial=0.)),
        "scope": "Consistency of the two playback runs' saved states and commands only; no controller or trajectory rerun.",
    }
    return {
        "label": "Clear baseline" if trajectory["name"] == "baseline" else "Extinction 0.6 /m",
        "seed": trajectory["seed"], "status": trajectory["status"],
        "times": state_t.tolist(), "scan_times": control_t.tolist(),
        "trajectory": _display_array(np.column_stack((xy, np.zeros(len(xy))))),
        "theta_rad": theta.tolist(),
        "reference": _display_array(np.column_stack((reference, np.zeros(len(reference))))),
        "points": points, "full_map_points": len(xyz), "display_map_points": len(points),
        "full_map_curve": curve,
        "map_step_at_state": [min(step, len(control_t)-1) for step in range(len(state_t))],
        "extent_m": metadata["extent_m"], "voxel_size_m": metadata["voxel_size_m"],
        "resolution_m": metadata["resolution_m"], "stats": metadata,
        "robot_radius_m": robot["radius_m"], "robot_height_m": robot["height_m"],
        "mount_height_m": trajectory["config"]["mount_height_m"],
        "sensor": trajectory["sensor"], "world_objects": trajectory["world_objects"],
        "controls": {"v_mps": command_v.tolist(), "omega_rad_s": command_omega.tolist(),
                     "swept_surface_gap_m": surface_gap.tolist()},
        "motion_check": motion_check,
        "display_sampling": "Deterministic evenly spaced original voxel indices; coordinates rounded to 5 decimals for display. Counts and curves use all saved measurements.",
        "playback_semantics": "At state k, display only measured voxels with first_step <= map_step_at_state[k]. The terminal state has no extra scan and holds the last map. Never interpolate endpoints or scan times.",
    }


def load_report_data(project, *, max_map_points=16_000):
    """Return JSON-safe, offline report data from a project folder.

    Numerical summaries are copied without rounding. The only geometric
    reduction is explicitly labelled sampling of already measured map points.
    Missing inputs raise rather than silently fabricating illustrative data.
    """
    project = Path(project).resolve()
    data = project / "data"
    files = {
        "primary": data / "navigation_dependencies/summary.json",
        "robustness": data / "navigation_robustness/summary.json",
        "robustness_manifest": data / "navigation_robustness/manifest.json",
        "threshold": data / "navigation_threshold/summary.json",
        "threshold_design": data / "navigation_threshold/design.json",
        "fov": data / "navigation_validation/fov_los.json",
        "geometry": data / "evaluation/geometry/summary.json",
        "response": data / "evaluation/response/summary.json",
        "diagnostics": data / "navigation_robustness/failure_diagnostics.json",
    }
    raw = {name: _read(path) for name, path in files.items()}
    primary, robustness = raw["primary"], raw["robustness"]
    common_viewpoint = []
    cases = {item["id"]: item for item in primary["cases"]}
    for case_id in ("target_near", "baseline", "target_far"):
        path = data / "navigation_dependencies" / cases[case_id]["representative_replay"] / "traces.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            row = next(row for row in csv.DictReader(stream) if abs(float(row["t"]) - 6.) < 1e-9)
        eligible = int(row["eligible_echoes"])
        common_viewpoint.append({
            "case_id": case_id, "seed": 1, "time_s": float(row["t"]),
            "robot_xy_m": [float(row["robot_x_m"]), float(row["robot_y_m"])],
            "robot_theta_rad": float(row["robot_theta_rad"]),
            "eligible_echoes": eligible,
            "mean_eligible_range_m": float(row["eligible_range_sum_m"]) / eligible if eligible else None,
            "mean_eligible_power_w": float(row["eligible_power_sum_w"]) / eligible if eligible else None,
            "mean_eligible_incidence_deg": float(row["eligible_incidence_sum_deg"]) / eligible if eligible else None,
        })
        files[f"common_view_{case_id}"] = path
    mapping = {}
    for case_id in ("baseline", "fog"):
        directory = data / "navigation_dependencies/runs" / case_id / "seed_1"
        mapping[case_id] = _mapping_payload(directory, max_points=max_map_points)
        mapping[case_id]["source_run"] = directory.relative_to(project).as_posix()
        for filename in ("mapping.npz", "mapping.json", "trajectory.npz", "trajectory.json"):
            files[f"{case_id}_{filename}"] = directory / filename
    result = {
        "schema_version": 1,
        "primary": {"cases": primary["cases"], "settings": primary["settings"],
                    "definitions": primary["definitions"], "limitations": primary["limitations"],
                    "seeds": primary["seeds"], "runs_count": len(primary["runs"]),
                    "replays_count": len(primary["replays"]),
                    "common_viewpoint": common_viewpoint},
        "threshold": raw["threshold"],
        "robustness": {key: robustness[key] for key in (
            "complete", "runs", "planned_runs", "seeds", "representative_seed", "layout_count",
            "case_count", "conditions", "cases", "paired_comparisons", "interpretation")},
        "fov": raw["fov"], "mapping": mapping,
        "verification": {"geometry": raw["geometry"], "response": raw["response"],
                         "recorded_motion": {case: item["motion_check"] for case, item in mapping.items()}},
        "diagnostics": raw["diagnostics"],
        "provenance": {"files": [_source(path, project) for path in files.values()],
                       "semantics": "Presentation of saved study results only. 3D FOV rays/meshes are prepared explanatory geometry; mapping points are measured endpoints. No scans, route generation or experiment reruns occur during presentation.",
                       "display_coordinate_decimals": 5,
                       "map_point_limit_per_run": max_map_points},
    }
    result["robustness"]["layouts"] = [
        {key: row[key] for key in ("id", "label", "description")}
        for row in raw["robustness_manifest"]["layouts"]]
    result["robustness"]["definitions"] = raw["robustness_manifest"]["definitions"]
    result["robustness"]["limitations"] = raw["robustness_manifest"]["limitations"]
    # Reject non-finite JSON values before embedding: browser null means missing,
    # rather than a silently coerced zero or an invalid JavaScript number token.
    json.dumps(result, allow_nan=False)
    return result
