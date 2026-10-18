"""Compare sensed trajectories and endpoint maps in three static environments."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np

from ..trajectory import load_config, generate_reference, run_trajectory, save_result
from ..mapping import build_map, save_map
from .trajectory import evaluate_result


def _box(label, x, y, sx, sy, height, material="concrete", yaw=0):
    return dict(type="box", label=label, center=[x, y, height / 2],
                size=[sx, sy, height], yaw_deg=yaw, material=material)


def _cylinder(label, x, y, radius, height, material="wood"):
    return dict(type="cylinder", label=label, base=[x, y, 0], radius=radius,
                height=height, material=material)


def environment_cases(config):
    """Keep sensing/control fixed while varying map topology and held-out objects."""
    base = copy.deepcopy(config)
    base.setdefault("sensor_overrides", {}).update(
        n_channels=8, measurements_per_cycle=360, n_beam_samples=1,
        air_extinction_per_m=2e-4, rain_mm_per_h=0.0)
    base["mount_height_m"] = 0.6
    base["robot"].update(start=[0.0, 0.0], goal=[18.0, 0.0], v_max=1.2,
                         max_steps=180)
    base["planner"]["extent"] = [-2.0, 21.0, -6.0, 7.0]
    base["controller"]["safety_margin_m"] = 0.25
    base["lidar_enabled"] = True
    wall = _box("known_wall", 6, -.8, 2, 3.8, 1.2)
    barrel = _cylinder("unexpected_barrel", 11, 1.1, .8, 1.2, "concrete")
    definitions = [
        dict(id="open", label="Open scene",
             description="One known wall and one obstacle omitted from the planning map.",
             known_objects=[wall], unexpected_objects=[barrel]),
        dict(id="cluttered", label="Cluttered scene",
             description="Twelve varied-height boxes, pillars, a sphere and held-out obstacles create more occlusion and local detours.",
             known_objects=[
                 wall,
                 _box("crate_south", 10, -2.3, 1.6, 1.4, 1.4, "wood"),
                 _box("stack_east", 15, -.8, 1.1, 2.2, 1.6),
                 _box("pallet_north", 14.5, 4.3, 2.2, 1.0, .6, "wood", -15),
                 _cylinder("pillar_west", 3, 3.5, .3, 2.6),
                 _cylinder("pillar_midnorth", 8.8, 4.1, .4, 2.2),
                 _cylinder("pillar_eastnorth", 17.2, 3.4, .3, 2.5),
                 _box("rack_southwest", 4, -4, 2.8, 1, 2.0),
                 dict(type="sphere", label="metal_sphere", center=[10, 5, .6],
                      radius=.6, material="aluminium"),
             ], unexpected_objects=[
                 barrel,
                 _box("unexpected_bin", 13.2, 2.9, .7, .8, .75, "car_paint_black"),
                 _cylinder("unexpected_low_barrel", 8.7, -.7, .4, .9),
             ]),
        dict(id="narrow", label="Narrow passage",
             description="Tall corridor walls and alternating barriers restrict the route and occlude two held-out obstacles.",
             known_objects=[
                 _box("north_corridor_wall", 9, 2.35, 16, .3, 2.5),
                 _box("south_corridor_wall", 9, -2.35, 16, .3, 2.5),
                 _box("north_chicane", 6, 1.15, .5, 2.1, 2.2),
                 _box("south_chicane", 12, -1.15, .5, 2.1, 2.2),
             ], unexpected_objects=[
                 _cylinder("hidden_cylinder", 9, -.1, .35, .8, "concrete"),
                 _box("hidden_crate", 15.5, -.5, .7, .7, .9, "wood"),
             ]),
    ]
    cases = []
    for definition in definitions:
        active = copy.deepcopy(base)
        active["world"] = {key: copy.deepcopy(definition[key])
                           for key in ("known_objects", "unexpected_objects")}
        active["scenarios"] = [{"name": definition["id"]}]
        cases.append({**definition, "config": active})
    return cases


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(directory, config=None, seeds=(1, 2, 3)):
    """Save full scans, maps and outcomes; no visualization is imported."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seed labels for the environment comparison")
    config = load_config(config) if config is None or isinstance(config, (str, Path)) else config
    rows, cases = [], []
    for case in environment_cases(config):
        active = case["config"]
        reference = generate_reference(active)
        case_rows = []
        for seed in seeds:
            result = run_trajectory(active, case["id"], seed=seed, reference=reference)
            relative = Path("runs") / case["id"] / f"seed_{seed}"
            target = directory / relative
            save_result(result, target)
            mapping = build_map(result, voxel_size_m=.12, resolution_m=.2)
            save_map(mapping, target)
            row = evaluate_result(result, case["id"], "environment", seed)
            observed = (mapping["ground_first_seen"] >= 0) | (mapping["obstacle_first_seen"] >= 0)
            row.update(
                label=case["label"], run_directory=relative.as_posix(),
                observed_fraction=float(observed.mean()),
                observed_area_m2=float(observed.sum() * .2 ** 2),
                map_voxels=len(mapping["map_xyz"]),
                surface_voxels=int(np.count_nonzero(mapping["map_xyz"][:, 2] > .12)),
                n_obstacles=len(active["world"]["known_objects"]) + len(active["world"]["unexpected_objects"]),
                n_known_obstacles=len(active["world"]["known_objects"]),
                n_unmapped_obstacles=len(active["world"]["unexpected_objects"]),
            )
            rows.append(row)
            case_rows.append(row)
            print(f"{case['id']} seed {seed}: {row['status']}, {row['elapsed_s']:.1f} s, "
                  f"gap {row['swept_min_surface_gap_m']:.3f} m, "
                  f"observed {100 * row['observed_fraction']:.1f}%", flush=True)
        representative = case_rows[0]
        aggregate = dict(
            runs=len(case_rows), goals=sum(r["goal_completed"] for r in case_rows),
            collisions=sum(r["collision"] for r in case_rows),
            timeouts=sum(r["timeout"] for r in case_rows),
            mean_elapsed_s=float(np.mean([r["elapsed_s"] for r in case_rows])),
            mean_path_length_m=float(np.mean([r["path_length_m"] for r in case_rows])),
            worst_surface_gap_m=min(r["swept_min_surface_gap_m"] for r in case_rows),
            mean_observed_fraction=float(np.mean([r["observed_fraction"] for r in case_rows])),
        )
        cases.append(dict(id=case["id"], label=case["label"], description=case["description"],
                          run_directory=representative["run_directory"], metrics=representative,
                          aggregate=aggregate))
    summary = _jsonable(dict(
        format_version=1, cases=cases, runs=rows, seeds=list(seeds),
        settings=dict(channels=8, azimuths=360, subrays=1, mount_height_m=.6,
                      v_max_mps=1.2, dt_s=float(active["robot"]["dt"]),
                      time_budget_s=float(active["robot"]["dt"] * active["robot"]["max_steps"]),
                      voxel_size_m=.12, grid_resolution_m=.2,
                      extent=[-2, 21, -6, 7]),
        assumptions=[
            "Measured world points use the simulator's exact robot pose; this is known-pose mapping, not SLAM.",
            "A separate incomplete known map plans each environment's reference route. Held-out obstacles require sensed avoidance.",
            "The growing map uses measured endpoints only. Ground observations do not certify traversability; unseen space and obstacle interiors remain unknown.",
            "Observed area is the fraction of endpoint-observed XY cells within the same bounded region, not map accuracy or free-space coverage.",
            "All outcomes enter comparisons. A failed run's area, time and path describe only its observed prefix.",
            "Environment topology and occlusion change together; this is a task comparison, not an isolated causal test of obstacle count.",
        ],
    ))
    (directory / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _csv(directory / "runs.csv", rows)
    _csv(directory / "cases.csv", [dict(id=c["id"], label=c["label"], description=c["description"],
                                        run_directory=c["run_directory"], **c["aggregate"]) for c in cases])
    return summary
