"""Repeated, paired navigation stress evaluation across three fixed layouts.

Every simulation is an inferential unit. Rays are never pooled to inflate the
navigation sample size. Only the first seed retains full scans and an endpoint
map; every run retains its complete states, controls and effective configuration.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from ..mapping import build_map, save_map
from ..trajectory import generate_reference, load_config, run_trajectory, save_result
from .environments import environment_cases
from .trajectory import evaluate_result


METRIC_KEYS = ("elapsed_s", "path_length_m", "swept_min_surface_gap_m",
               "spatial_route_rmse_m", "mean_returns_per_control_step")


def robustness_cases():
    """A predeclared graded sweep and matched sensor ablations."""
    cases = [dict(id="baseline", label="Baseline", factor="baseline", overrides={},
                  comparison_baseline="baseline")]
    for coefficient, suffix in ((.1, "010"), (.3, "030"), (.6, "060"), (1.2, "120"), (3., "300"), (10., "1000")):
        cases.append(dict(id=f"fog_{suffix}", label=f"Extinction {coefficient:g} /m",
                          factor="optical_stress", overrides={"sensor": {"air_extinction_per_m": coefficient}},
                          comparison_baseline="baseline"))
    cases.extend([
        dict(id="mount_high", label="Mount 1.6 m", factor="sensor_position",
             overrides={"mount_height_m": 1.6}, comparison_baseline="baseline"),
        dict(id="sampling_sparse", label="90 azimuth samples", factor="sampling",
             overrides={"sensor": {"measurements_per_cycle": 90}}, comparison_baseline="baseline"),
        dict(id="fov_front", label="Front 120-degree FOV", factor="field_of_view",
             overrides={"sensor": {"hfov_start_deg": -60., "hfov_end_deg": 60.}},
             comparison_baseline="baseline"),
        dict(id="fov_side", label="Side 120-degree FOV", factor="sensor_orientation",
             overrides={"sensor": {"hfov_start_deg": -60., "hfov_end_deg": 60.},
                        "sensor_mount": {"yaw_deg": 90.}}, comparison_baseline="fov_front"),
        dict(id="lidar_disabled", label="LiDAR disabled", factor="sensor_ablation",
             overrides={"lidar_enabled": False}, comparison_baseline="baseline"),
    ])
    return cases


def wilson_interval(successes, trials, z=1.959963984540054):
    """Two-sided 95% binomial score interval, counting independent runs."""
    if not isinstance(trials, (int, np.integer)) or trials < 1:
        raise ValueError("Wilson interval requires a positive integer trial count")
    if not isinstance(successes, (int, np.integer)) or not 0 <= successes <= trials:
        raise ValueError("Successes must be an integer between zero and trials")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("The normal critical value must be positive and finite")
    fraction, denominator = successes / trials, 1 + z * z / trials
    center = (fraction + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def describe(values):
    """An empty success-conditioned metric remains undefined, never zero."""
    values = np.asarray(list(values), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Summary metrics must be finite")
    if not len(values):
        return dict(n=0, mean=None, median=None, q25=None, q75=None, min=None, max=None)
    return dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
                min=float(np.min(values)), max=float(np.max(values)))


def aggregate_runs(rows):
    """Keep termination-prefix and success-conditioned scores visibly separate."""
    if not rows:
        raise ValueError("Cannot aggregate an empty run group")
    goals = sum(int(row["goal_completed"]) for row in rows)
    successes = [row for row in rows if row["goal_completed"]]
    metrics = lambda selected: {key: describe(row[key] for row in selected) for key in METRIC_KEYS}
    return dict(runs=len(rows), goals=goals,
                collisions=sum(int(row["collision"]) for row in rows),
                timeouts=sum(int(row["timeout"]) for row in rows),
                statuses={status: sum(row["status"] == status for row in rows)
                          for status in sorted({row["status"] for row in rows})},
                goal_fraction=goals / len(rows), goal_fraction_wilson95=wilson_interval(goals, len(rows)),
                all_run_prefix=metrics(rows), successful_runs=metrics(successes))


def paired_differences(rows, baseline_rows):
    """Pair seed labels within one layout, without assuming ray-identical noise."""
    first = {int(row["seed"]): row for row in rows}
    baseline = {int(row["seed"]): row for row in baseline_rows}
    if len(first) != len(rows) or len(baseline) != len(baseline_rows) or first.keys() != baseline.keys():
        raise ValueError("A paired comparison requires identical, distinct seed labels")
    pairs = []
    for seed in sorted(first):
        active, reference = first[seed], baseline[seed]
        both_goal = bool(active["goal_completed"] and reference["goal_completed"])
        row = dict(seed=seed, case_goal=int(active["goal_completed"]), baseline_goal=int(reference["goal_completed"]),
                   goal_difference=int(active["goal_completed"]) - int(reference["goal_completed"]),
                   both_goal=both_goal)
        row.update({"delta_" + key: float(active[key] - reference[key]) for key in METRIC_KEYS})
        pairs.append(row)
    return dict(pairs=len(pairs), case_only_goal=sum(row["goal_difference"] == 1 for row in pairs),
                baseline_only_goal=sum(row["goal_difference"] == -1 for row in pairs),
                both_goal=sum(row["both_goal"] for row in pairs),
                neither_goal=sum(not row["case_goal"] and not row["baseline_goal"] for row in pairs),
                goal_fraction_difference=float(np.mean([row["goal_difference"] for row in pairs])),
                all_run_prefix_deltas={key: describe(row["delta_" + key] for row in pairs) for key in METRIC_KEYS},
                both_success_deltas={key: describe(row["delta_" + key] for row in pairs if row["both_goal"])
                                     for key in METRIC_KEYS}, per_seed=pairs)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _csv(path, rows):
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _config_hash(config):
    return hashlib.sha256(json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _build_summary(rows, layouts, cases, seeds, representative, started, complete):
    aggregated, paired = [], []
    for layout in layouts:
        for case in cases:
            selected = [row for row in rows if row["layout_id"] == layout["id"] and row["case_id"] == case["id"]]
            if not selected:
                continue
            aggregation = aggregate_runs(selected)
            aggregated.append(dict(layout_id=layout["id"], layout_label=layout["label"],
                                   case_id=case["id"], label=case["label"], factor=case["factor"],
                                   comparison_baseline=case["comparison_baseline"], overrides=case["overrides"],
                                   **aggregation, representative=representative.get((layout["id"], case["id"]))))
            baseline = [row for row in rows if row["layout_id"] == layout["id"]
                        and row["case_id"] == case["comparison_baseline"]]
            if case["id"] != case["comparison_baseline"] and {r["seed"] for r in baseline} == {r["seed"] for r in selected}:
                paired.append(dict(layout_id=layout["id"], case_id=case["id"],
                                   comparison_baseline=case["comparison_baseline"],
                                   **paired_differences(selected, baseline)))
    return dict(format_version=1, study="navigation_robustness", complete=complete, runs=len(rows),
                planned_runs=len(layouts) * len(cases) * len(seeds), seeds=list(seeds),
                representative_seed=seeds[0], layout_count=len(layouts), case_count=len(cases),
                layouts=[{key: layout[key] for key in ("id", "label", "description")} for layout in layouts],
                conditions=cases, cases=aggregated, run_metrics=rows, paired_comparisons=paired,
                elapsed_wall_time_s=time.perf_counter() - started,
                interpretation="Repeated sensor-noise trials within three pre-existing deterministic layouts; confidence intervals describe stochastic outcomes conditional on each layout and this controller, not population-level real-world reliability.")


def run_study(directory, config=None, seeds=range(1001, 1021), *, case_ids=None, layout_ids=None):
    """Run fixed-layout paired trials; optional subsets support focused smoke runs.

    Full maps are representative examples only, not 20-run map coverage estimates.
    Every numerical trajectory is retained, including all collision/timeout runs.
    A manifest mismatch prevents accidentally resuming data with changed settings.
    """
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty, distinct seed labels")
    original = load_config(config) if config is None or isinstance(config, (str, Path)) else copy.deepcopy(config)
    layouts, cases = environment_cases(original), robustness_cases()
    for requested, available, name in ((case_ids, cases, "case"), (layout_ids, layouts, "layout")):
        if requested is not None and (not requested or set(requested) - {item["id"] for item in available}):
            raise ValueError(f"Unknown or empty {name} subset")
    if case_ids is not None:
        cases = [case for case in cases if case["id"] in case_ids]
    if layout_ids is not None:
        layouts = [layout for layout in layouts if layout["id"] in layout_ids]
    manifest = dict(format_version=1, study="navigation_robustness", seeds=list(seeds), cases=cases,
                    representative_seed=seeds[0], supplied_config=original,
                    layouts=[dict(id=layout["id"], label=layout["label"], description=layout["description"],
                                  config=layout["config"]) for layout in layouts],
                    pairing="Noise-seed labels are paired within layouts. Geometry and beam-count changes alter RNG consumption; paired labels are not identical per-ray perturbations.",
                    definitions=dict(
                        independent_unit="One complete closed-loop run. Success intervals use run counts, never rays or time steps.",
                        success="Engine goal termination: within configured goal tolerance, without prior collision, before configured step budget.",
                        wilson="Per-layout per-condition two-sided Wilson 95% binomial score interval with z=1.959963984540054. Intervals are pointwise and conditional on the fixed scene and model.",
                        clearance="Minimum exact swept-disc surface gap over initial state and every integration segment; negative means overlap.",
                        all_run_prefix="Each failed run contributes its complete observed prefix. A short prefix is not an efficient completed route.",
                        successful_runs="Time/path summaries condition on goal completion; undefined if no successful runs.",
                        pairing="Differences are case minus the stated matched baseline at the same layout and noise-seed label. Time/path interpretation uses both-success pairs.",
                        representative_map="Only the first seed of each case retains full measured scans and a 0.12 m voxel/0.2 m endpoint-cell map. Observed fraction is an illustrative measured-endpoint statistic, not map accuracy or free space."),
                    limitations=[
                        "Three existing layouts are fixed test scenarios, not randomized environments or a statistically sampled scene population.",
                        "Robot pose is exact; static objects, snapshot scans, and a planar unicycle/disc model are assumed.",
                        "Sensor coefficients and extinction values are illustrative; no physical calibration or atmospheric backscatter is modeled.",
                        "Controller, robot, planner and per-layout partial-map reference are frozen across conditions. No scene was adjusted to force a failure.",
                        "Intervals and comparisons are exploratory and pointwise; no familywise error control or real-world safety claim is made.",
                    ])
    signature = _config_hash(manifest)
    manifest["configuration_sha256"] = signature
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("configuration_sha256") != signature:
            raise ValueError("Existing robustness output has a different configuration; choose a separate directory")
    else:
        _json(manifest_path, manifest)
    _csv(directory / "cases.csv", [dict(case_id=case["id"], label=case["label"], factor=case["factor"],
                                        comparison_baseline=case["comparison_baseline"],
                                        overrides_json=json.dumps(case["overrides"], sort_keys=True)) for case in cases])
    rows, representatives, started = [], {}, time.perf_counter()
    for layout in layouts:
        baseline = layout["config"]
        reference = generate_reference(baseline)
        reference_dir = directory / "references"
        reference_dir.mkdir(exist_ok=True)
        np.savez_compressed(reference_dir / f"{layout['id']}.npz",
                            **{key: value for key, value in reference.items() if isinstance(value, np.ndarray)})
        _json(reference_dir / f"{layout['id']}.json",
              {key: value for key, value in reference.items() if not isinstance(value, np.ndarray)})
        for case in cases:
            scenario = {"name": case["id"], **copy.deepcopy(case["overrides"])}
            scenario.setdefault("lidar_enabled", True)
            for seed in seeds:
                relative = Path("runs") / layout["id"] / case["id"] / f"seed_{seed}"
                destination = directory / relative
                metric_path = destination / "metrics.json"
                if metric_path.exists():
                    row = json.loads(metric_path.read_text(encoding="utf-8"))
                    if row.get("configuration_sha256") != signature:
                        raise ValueError("Saved run signature differs from the current manifest")
                else:
                    began = time.perf_counter()
                    record = seed == seeds[0]
                    result = run_trajectory(baseline, scenario, seed=seed, record_history=record, reference=reference)
                    save_result(result, destination)
                    row = evaluate_result(result, case["id"], case["factor"], seed, scenario["lidar_enabled"])
                    row.update(layout_id=layout["id"], layout_label=layout["label"], label=case["label"],
                               run_directory=relative.as_posix(), comparison_baseline=case["comparison_baseline"],
                               representative_full_scans=record, configuration_sha256=signature)
                    if record:
                        mapping = build_map(result, voxel_size_m=.12, resolution_m=.2)
                        save_map(mapping, destination)
                        observed = (mapping["ground_first_seen"] >= 0) | (mapping["obstacle_first_seen"] >= 0)
                        representative = dict(seed=seed, run_directory=relative.as_posix(),
                                              observed_fraction=float(observed.mean()),
                                              observed_area_m2=float(observed.sum() * .2 ** 2),
                                              map_voxels=len(mapping["map_xyz"]),
                                              status=result["status"])
                        _json(destination / "representative_map.json", representative)
                    row["wall_time_s"] = time.perf_counter() - began
                    _json(metric_path, row)
                    print(f"robustness {layout['id']}/{case['id']} seed {seed}: {row['status']}, "
                          f"{row['elapsed_s']:.1f} s, gap {row['swept_min_surface_gap_m']:.3f} m; "
                          f"wall {row['wall_time_s']:.2f} s", flush=True)
                rows.append(row)
                if seed == seeds[0]:
                    representatives[(layout["id"], case["id"])] = json.loads(
                        (destination / "representative_map.json").read_text(encoding="utf-8"))
            summary = _build_summary(rows, layouts, cases, seeds, representatives, started, complete=False)
            _json(directory / "summary.json", summary)
            _csv(directory / "runs.csv", rows)
    summary = _build_summary(rows, layouts, cases, seeds, representatives, started, complete=True)
    _json(directory / "summary.json", summary)
    _csv(directory / "runs.csv", rows)
    return summary


def _slow_episodes(result, speed_threshold_mps=.05, minimum_steps=5):
    """Contiguous low commanded-speed intervals from saved control timestamps."""
    controls, states = result["controls"], result["states"]
    slow = np.asarray(controls["v"]) < speed_threshold_mps
    padded = np.r_[False, slow, False].astype(np.int8)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    episodes = []
    for first, stop in zip(starts, ends):
        if stop - first < minimum_steps:
            continue
        episodes.append(dict(first_control_step=int(first), stop_control_step_exclusive=int(stop),
                             start_time_s=float(states["t"][first]), end_time_s=float(states["t"][stop]),
                             duration_s=float(states["t"][stop] - states["t"][first]),
                             control_steps=int(stop-first), start_xy_m=states["xy"][first].tolist(),
                             end_xy_m=states["xy"][stop].tolist(),
                             net_translation_m=float(np.linalg.norm(states["xy"][stop]-states["xy"][first])),
                             mean_returns=float(np.mean(controls["n_returns"][first:stop])),
                             mean_controller_points=float(np.mean(controls["n_obstacle_points"][first:stop])),
                             mean_abs_angular_rate_rad_s=float(np.mean(np.abs(controls["omega"][first:stop])))))
    return episodes


def summarize_failure_modes(directory):
    """Audit illustrative failures from completed saved runs, without scanning.

    The only fresh geometric calculations are deterministic nearest-hit ray
    audits at recorded high-mount poses, repeated with baseline mounting height
    at the identical robot poses. This counterfactual does not rerun navigation,
    sample receiver noise, or change the canonical trial outcomes.
    """
    from .. import api
    from ..trajectory import load_result
    from ..trajectory.engine import mounted_sensor_pose
    from .navigation_dependencies import emitted_rays

    directory = Path(directory).expanduser().resolve()
    summary_file = directory / "summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError("Failure diagnostics require a completed robustness study")
    lookup = {(row["layout_id"], row["case_id"], row["seed"]): row for row in summary["run_metrics"]}

    def saved(layout, condition, seed):
        key = (layout, condition, seed)
        if key not in lookup:
            raise ValueError(f"Missing diagnostic run {key}")
        row = lookup[key]
        destination = directory / row["run_directory"]
        result = load_result(destination)
        return row, result, dict(run_directory=row["run_directory"], seed=seed, layout_id=layout,
                                 case_id=condition, trajectory_sha256=hashlib.sha256(
                                     (destination / "trajectory.npz").read_bytes()).hexdigest())

    stalls = []
    selections = (("open", "fog_120", 1005), ("open", "fog_1000", 1001),
                  ("narrow", "baseline", 1012))
    for layout, condition, seed in selections:
        row, result, provenance = saved(layout, condition, seed)
        controls = result["controls"]
        stalls.append(dict(**provenance, status=row["status"], elapsed_s=row["elapsed_s"],
                           final_goal_distance_m=row["final_goal_distance_m"],
                           swept_min_surface_gap_m=row["swept_min_surface_gap_m"],
                           path_length_m=row["path_length_m"], final_xy_m=result["states"]["xy"][-1].tolist(),
                           final_reference_progress_m=float(result["states"]["progress_s"][-1]),
                           reference_length_m=float(result["reference"]["s"][-1]),
                           last_ten_mean_returns=float(np.mean(controls["n_returns"][-10:])),
                           last_ten_mean_controller_points=float(np.mean(controls["n_obstacle_points"][-10:])),
                           low_speed_episodes=_slow_episodes(result)))

    row, result, provenance = saved("open", "mount_high", 1001)
    config = result["config"]
    cfg = api.build_sensor(config["_effective_sensor"])
    scene = api.build_scene({"objects": copy.deepcopy(result["world_objects"])})
    label = "unexpected_barrel"
    target_id = scene.labels.index(label)
    baseline_height = next(item["config"]["mount_height_m"] for item in manifest["layouts"] if item["id"] == "open")
    k_total = len(result["controls"]["t"])
    snapshots = []
    for step in range(max(0, k_total-4), k_total):
        record = dict(step=step, time_s=float(result["states"]["t"][step]),
                      robot_xy_m=result["states"]["xy"][step].tolist(),
                      robot_theta_rad=float(result["states"]["theta"][step]),
                      recorded_all_returns=int(result["controls"]["n_returns"][step]),
                      recorded_all_controller_points=int(result["controls"]["n_obstacle_points"][step]))
        for name, height in (("actual_high_mount", config["mount_height_m"]), ("same_pose_baseline_mount", baseline_height)):
            geometric_config = copy.deepcopy(config)
            geometric_config["mount_height_m"] = height
            pose = mounted_sensor_pose(geometric_config, result["states"]["xy"][step], result["states"]["theta"][step])
            origin, direction = emitted_rays(cfg, pose)
            distance, nearest_id, _ = scene.cast(origin, direction)
            valid = (nearest_id == target_id) & (distance >= cfg.min_range_m) & (distance <= cfg.max_range_m)
            points = origin[valid] + distance[valid, None] * direction[valid]
            relevant = ((points[:,2] >= config["controller"]["z_ground_m"])
                        & (points[:,2] <= config["robot"]["height_m"]))
            record[name] = dict(mount_height_m=float(height), nearest_range_eligible_barrel_rays=int(valid.sum()),
                                barrel_rays_within_controller_height_band=int(relevant.sum()),
                                minimum_true_barrel_hit_height_m=float(points[:,2].min()) if len(points) else None)
        snapshots.append(record)
    cluttered_high = next(case for case in summary["cases"]
                          if case["layout_id"] == "cluttered" and case["case_id"] == "mount_high")
    cluttered_base = next(case for case in summary["cases"]
                          if case["layout_id"] == "cluttered" and case["case_id"] == "baseline")
    successful_clutter = [r for r in summary["run_metrics"] if r["layout_id"] == "cluttered"
                         and r["case_id"] == "mount_high" and r["goal_completed"]]
    payload = dict(format_version=1, study="navigation_robustness_failure_diagnostics",
                   manifest_configuration_sha256=manifest["configuration_sha256"],
                   summary_sha256=hashlib.sha256(summary_file.read_bytes()).hexdigest(),
                   method="Read completed saved states and controls; deterministic nearest-hit casts at recorded poses only. No scanner calls, noise draws, replanning, controller changes or outcome reruns.",
                   low_speed_definition=dict(commanded_speed_below_mps=.05, minimum_contiguous_control_steps=5,
                                             intervals="[first_control_step, stop_control_step_exclusive); end time is the resulting state timestamp"),
                   stalled_run_examples=stalls,
                   high_mount_near_field=dict(**provenance, target_label=label, status=row["status"],
                                              sensor_vertical_fov_deg=[float(cfg.vfov_lower_deg),float(cfg.vfov_upper_deg)],
                                              controller_point_height_band_m=[float(config["controller"]["z_ground_m"]),
                                                                              float(config["robot"]["height_m"])],
                                              snapshots=snapshots,
                                              interpretation="The sensor can geometrically hit the barrel while all those hits lie above the controller's height band. The lower-mount counterfactual uses identical recorded XY/yaw poses and deterministic geometry, not simulated noisy returns or a new navigation trial."),
                   cluttered_high_mount=dict(goals=cluttered_high["goals"], runs=cluttered_high["runs"],
                                             collisions=cluttered_high["collisions"],
                                             all_run_gap_statistics=cluttered_high["all_run_prefix"]["swept_min_surface_gap_m"],
                                             successful_run_gap_statistics=describe(r["swept_min_surface_gap_m"] for r in successful_clutter),
                                             baseline_gap_statistics=cluttered_base["all_run_prefix"]["swept_min_surface_gap_m"]),
                   limitations=[
                       "Examples were selected after inspecting outcomes to explain failures; they do not supply new independent test trials or causal success-rate estimates.",
                       "Low commanded speed and heading oscillation support controller stagnation; recorded logs do not prove a unique internal cause for every turn choice.",
                       "Geometric height-band counts omit receiver noise. They explain the sensing/filter mechanism, while stored trial outcomes remain the primary navigation evidence.",
                       "The nonmonotonic 19/20 versus 20/20 completion examples have strongly overlapping pointwise intervals and do not establish that increased attenuation improves reliability.",
                   ])
    _json(directory / "failure_diagnostics.json", payload)
    return payload
