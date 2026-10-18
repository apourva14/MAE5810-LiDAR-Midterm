"""Paired, one-factor trajectory studies using saved numerical results.

The study reports every observed termination, including failures. Spatial route
adherence and clock-referenced position tracking are different quantities.
Neither illustrative coefficients nor a successful simulation certify safety.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import time

import numpy as np


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def point_polyline_distances(points, route):
    """Euclidean distances to a geometric polyline, independent of its timing."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    route = np.asarray(route, dtype=float).reshape(-1, 2)
    if not len(route):
        raise ValueError("A reference route must contain at least one point")
    if len(route) == 1:
        return np.linalg.norm(points - route[0], axis=1)
    delta = np.diff(route, axis=0)
    squared = np.sum(delta * delta, axis=1)
    relative = points[:, None, :] - route[None, :-1, :]
    fraction = np.sum(relative * delta[None, :, :], axis=2) / np.where(squared > 0, squared, 1)
    fraction = np.clip(fraction, 0, 1)
    residual = relative - fraction[:, :, None] * delta[None, :, :]
    return np.sqrt(np.min(np.sum(residual * residual, axis=2), axis=1))


def tracking_metrics(times, positions, reference):
    """State-sampled errors, including the final state; no time weighting."""
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if len(times) != len(positions) or not len(times):
        raise ValueError("Trajectory times and positions must contain the same nonzero number of states")
    reference_xy = np.asarray(reference["xy"], dtype=float)
    reference_t = np.asarray(reference["t"], dtype=float)
    if len(reference_xy) != len(reference_t) or not len(reference_t):
        raise ValueError("Reference time and position arrays must have equal nonzero length")
    if np.any(np.diff(reference_t) < 0):
        raise ValueError("Reference timestamps must be nondecreasing")
    spatial_error = point_polyline_distances(positions, reference_xy)
    scheduled = np.column_stack([np.interp(times, reference_t, reference_xy[:, axis]) for axis in range(2)])
    time_error = np.linalg.norm(positions - scheduled, axis=1)
    return dict(
        spatial_route_rmse_m=float(np.sqrt(np.mean(spatial_error ** 2))),
        spatial_route_max_m=float(spatial_error.max()),
        time_reference_rmse_m=float(np.sqrt(np.mean(time_error ** 2))),
        time_reference_max_m=float(time_error.max()),
    )


def evaluate_result(result, case_id, factor, seed, lidar_enabled=True):
    """Summarize a run without dropping its terminal state or failed outcome."""
    from ..trajectory import swept_surface_gap
    from .. import api

    states, controls = result["states"], result["controls"]
    times, positions = np.asarray(states["t"]), np.asarray(states["xy"])
    omega = np.asarray(controls["omega"], dtype=float)
    return_counts = np.asarray(controls["n_returns"], dtype=float)
    if len(times) != len(omega) + 1:
        raise ValueError("A run must contain K controls and K+1 states, including its true terminal state")
    if len(return_counts) != len(omega):
        raise ValueError("Each control step must have one recorded return count")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Recorded trajectory state timestamps must increase")
    active = result["config"]
    radius, height = active["robot"]["radius_m"], active["robot"]["height_m"]
    objects = result["world_objects"]
    if objects and isinstance(objects[0], dict):
        objects = api.build_scene({"objects": objects})
    gaps = [swept_surface_gap(positions[0], positions[0], objects, radius, height)]
    gaps.extend(swept_surface_gap(a, b, objects, radius, height) for a, b in zip(positions[:-1], positions[1:]))
    minimum_gap = float(min(gaps))
    status = str(result["status"])
    metric = dict(
        case_id=case_id, factor=factor, seed=int(seed), status=status,
        goal_completed=int(status == "goal"), collision=int(status == "collision"),
        timeout=int(status == "timeout"), state_count=len(times), control_steps=len(omega),
        elapsed_s=float(times[-1] - times[0]),
        path_length_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        swept_min_surface_gap_m=minimum_gap,
        initial_surface_gap_m=float(gaps[0]),
        overlap_observed=int(minimum_gap < 0.0),
        final_goal_distance_m=float(np.linalg.norm(positions[-1] - np.asarray(active["robot"]["goal"]))),
    )
    metric.update(tracking_metrics(times, positions, result["reference"]))
    metric.update(
        angular_rate_total_variation_rad_s=float(np.abs(np.diff(omega)).sum()),
        angular_rate_step_change_mean_rad_s=float(np.abs(np.diff(omega)).mean()) if len(omega) > 1 else 0.0,
        maximum_abs_commanded_angular_rate_rad_s=float(np.abs(omega).max()) if len(omega) else 0.0,
        mean_returns=float(return_counts.mean()) if len(return_counts) else 0.0,
        mean_returns_per_control_step=float(return_counts.mean()) if len(return_counts) else 0.0,
        mean_obstacle_points_per_control_step=float(np.mean(controls["n_obstacle_points"])) if len(omega) and "n_obstacle_points" in controls else 0.0,
        scans_performed=len(omega) if lidar_enabled else 0,
        lidar_enabled=int(lidar_enabled),
    )
    if "swept_surface_gap" in controls and len(omega):
        recorded = np.asarray(controls["swept_surface_gap"], dtype=float)
        independently_recomputed = np.asarray(gaps[1:])
        metric["swept_gap_recomputation_max_error_m"] = float(np.max(np.abs(recorded - independently_recomputed)))
    else:
        metric["swept_gap_recomputation_max_error_m"] = 0.0
    return metric


def factor_matrix():
    """Overrides are relative to one shared baseline and fixed reference route."""
    return [
        dict(case_id="baseline", factor="baseline",
             description="8 channels, 360 azimuths, one sub-ray; clear air, 0.6 m mount, 1.2 m/s speed limit, 0.25 m local margin",
             overrides={}),
        dict(case_id="dense_fog", factor="extinction",
             description="Air extinction coefficient 0.6 per m; all other settings unchanged",
             overrides={"sensor": {"air_extinction_per_m": 0.6}}),
        dict(case_id="sparse_sampling", factor="azimuth_density",
             description="90 azimuths per revolution instead of 360",
             overrides={"sensor": {"measurements_per_cycle": 90}}),
        dict(case_id="high_mount", factor="mount_height",
             description="Mounting height 1.6 m instead of 0.6 m",
             overrides={"mount_height_m": 1.6}),
        dict(case_id="faster_speed", factor="speed_limit",
             description="Speed limit 2.0 m/s instead of 1.2 m/s",
             overrides={"robot": {"v_max": 2.0}}),
        dict(case_id="smaller_margin", factor="local_safety_margin",
             description="Local controller margin 0.05 m instead of 0.25 m; planned route unchanged",
             overrides={"controller": {"safety_margin_m": 0.05}}),
        dict(case_id="lidar_disabled", factor="sensor_enabled",
             description="LiDAR disabled; retain the same route planned from the known reference map",
             overrides={"lidar_enabled": False}),
    ]


def _finite_mean(rows, key):
    values = [float(row[key]) for row in rows if row[key] is not None and np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else None


def summarize_cases(rows, factors):
    """Aggregate every run, including early collision and timeout terminations."""
    summaries = []
    for case in factors:
        selected = [row for row in rows if row["case_id"] == case["case_id"]]
        if not selected:
            continue
        gaps = [row["swept_min_surface_gap_m"] for row in selected]
        times = [row["elapsed_s"] for row in selected]
        statuses = {status: sum(row["status"] == status for row in selected)
                    for status in sorted({row["status"] for row in selected})}
        summaries.append(dict(
            case_id=case["case_id"], factor=case["factor"], description=case["description"],
            overrides=case["overrides"], runs=len(selected),
            goals=sum(row["goal_completed"] for row in selected),
            collisions=sum(row["collision"] for row in selected),
            timeouts=sum(row["timeout"] for row in selected), statuses=statuses,
            goal_fraction=sum(row["goal_completed"] for row in selected) / len(selected),
            mean_termination_time_s_all_runs=float(np.mean(times)),
            termination_time_range_s=[min(times), max(times)],
            mean_path_length_m=_finite_mean(selected, "path_length_m"),
            min_surface_gap_m=float(min(gaps)),
            surface_gap_range_m=[float(min(gaps)), float(max(gaps))],
            mean_spatial_route_rmse_m=_finite_mean(selected, "spatial_route_rmse_m"),
            max_spatial_route_error_m=max(row["spatial_route_max_m"] for row in selected),
            mean_time_reference_rmse_m=_finite_mean(selected, "time_reference_rmse_m"),
            mean_angular_rate_total_variation_rad_s=_finite_mean(selected, "angular_rate_total_variation_rad_s"),
            mean_returns_per_control_step=_finite_mean(selected, "mean_returns_per_control_step"),
        ))
    return summaries


def run_study(directory, config, seeds=(1, 2, 3)):
    """Run seven one-factor cases, save complete numerical runs, and return summary."""
    from ..trajectory import generate_reference, load_config, run_trajectory, save_result

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("A study requires nonempty, distinct seed labels")
    original = load_config(config) if isinstance(config, (str, Path)) else copy.deepcopy(config)
    baseline = copy.deepcopy(original)
    baseline.setdefault("sensor_overrides", {}).update(
        n_channels=8, measurements_per_cycle=360, n_beam_samples=1,
        air_extinction_per_m=2e-4, rain_mm_per_h=0.0)
    baseline["mount_height_m"] = 0.6
    baseline["robot"]["v_max"] = 1.2
    baseline["controller"]["safety_margin_m"] = 0.25
    factors = factor_matrix()
    reference = generate_reference(baseline)
    manifest = dict(
        study="trajectory_one_factor", seeds=list(seeds),
        baseline_config=baseline, supplied_config=original, factors=factors,
        pairing="The same seed labels are paired across cases. Changing beam counts, ray geometry or echo groups can change random-number consumption; perturbations are not assumed identical per ray.",
        definitions=dict(
            denominators="All runs, including failures, enter each case summary. Every run has K controls and K+1 states; final states enter time, path and tracking metrics.",
            spatial_route_rmse_m="Unweighted RMS over all recorded states of distance to the geometric reference polyline, irrespective of clock time.",
            time_reference_rmse_m="Unweighted RMS over all recorded states of distance to the reference position at that state's clock time; reference position is held at the end after its schedule finishes. The controller follows spatial progress; this cruise-time schedule is diagnostic, not a tracked clock setpoint or a turn-feasibility guarantee.",
            swept_min_surface_gap_m="Minimum over the initial pose and every swept straight integration segment of robot-centre-to-filled-obstacle-footprint distance minus robot radius; only obstacles intersecting the robot's vertical interval are considered. Negative values indicate overlap, not exact penetration depth.",
            angular_rate_total_variation_rad_s="Sum of absolute differences between successive commanded angular rates, in rad/s. Start/stop bookends are not added; this is not a physical smoothness or jerk metric.",
            mean_returns_per_control_step="Arithmetic mean of recorded full 3D return counts (including ground) over the K control updates; zero for a disabled sensor. Obstacle-only counts are a separate controller input. This is not target-conditioned detection probability.",
            failure_prefix="Path and tracking metrics for failures describe only the observed prefix; short termination time or low prefix error does not imply successful tracking.",
        ),
        limitations=[
            f"{len(seeds)} seed labels are descriptive examples, not a population-level reliability estimate.",
            "All sensor, material and environmental coefficients are illustrative, not calibrated.",
            "A reference route planned using the known map can already avoid known obstacles; LiDAR-disabled ablation distinguishes that contribution from sensed obstacle avoidance.",
            "Collision clearance uses a planar disc and static obstacle footprints with vertical overlap. The study does not certify real robot safety.",
        ])
    _write_json(directory / "manifest.json", manifest)
    _write_csv(directory / "cases.csv", [
        dict(case_id=case["case_id"], factor=case["factor"], description=case["description"],
             overrides_json=json.dumps(case["overrides"], sort_keys=True)) for case in factors])
    rows, started = [], time.perf_counter()
    for case in factors:
        scenario = {"name": case["case_id"], **copy.deepcopy(case["overrides"])}
        scenario.setdefault("lidar_enabled", True)
        for seed in seeds:
            began = time.perf_counter()
            result = run_trajectory(baseline, scenario, seed=seed, record_history=True, reference=reference)
            destination = directory / "runs" / case["case_id"] / f"seed_{seed}"
            save_result(result, destination)
            row = evaluate_result(result, case["case_id"], case["factor"], seed, scenario["lidar_enabled"])
            row["wall_time_s"] = time.perf_counter() - began
            row["run_directory"] = str(destination.relative_to(directory))
            rows.append(row)
            print(f"trajectory study: {case['case_id']} seed {seed}: {row['status']}, "
                  f"{row['elapsed_s']:.1f} s, minimum surface gap {row['swept_min_surface_gap_m']:.3f} m; "
                  f"wall time {row['wall_time_s']:.2f} s", flush=True)
        _write_csv(directory / "runs.csv", rows)
    summary = dict(
        study="trajectory_one_factor", runs=len(rows), seeds=list(seeds), case_count=len(factors),
        reference_length_m=float(np.linalg.norm(np.diff(np.asarray(reference["xy"]), axis=0), axis=1).sum()),
        reference_duration_s=float(np.asarray(reference["t"])[-1]),
        elapsed_wall_time_s=time.perf_counter() - started,
        cases=summarize_cases(rows, factors),
        interpretation="Descriptive paired simulations of an illustrative sensor/controller, including all failures; not safety certification or physical validation.")
    summary = _jsonable(summary)
    _write_json(directory / "summary.json", summary)
    return summary
