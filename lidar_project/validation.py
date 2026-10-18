"""Numerical checks against analytical references; independent of rendering."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import replace

import numpy as np

from .config import CONFIG_TEMPLATE, ConfigError, _strip_jsonc, build_all
from .model.geometry import Plane, Sphere, OrientedBox, Scene
from .model.materials import MATERIALS
from .model.sensor import (
    LidarConfig, LidarSensor, SensorPose, received_power, max_range_estimate,
)


def selftest() -> bool:
    ok_all = True

    def report(name, ok, detail):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name:34s} {detail}")

    base = LidarConfig()

    # 1. Ground plane: range = h / sin(depression)
    h = 1.8
    cfg = replace(base, n_channels=5, vfov_lower_deg=-20, vfov_upper_deg=-6, measurements_per_cycle=1800)
    pc = LidarSensor(cfg, SensorPose(position=(0, 0, h)), seed=3).scan(Scene([Plane((0, 0, 0), (0, 0, 1), "asphalt")]))
    errs = []
    for r in range(5):
        k = pc.ring == r
        exp = h / np.sin(-np.radians(pc.elevation_deg[k]))
        errs.append(np.mean(pc.range_m[k] - exp))
    report("ground plane range vs h/sin(el)", max(abs(e) for e in errs) < 0.01,
           f"max |mean error| = {max(abs(e) for e in errs) * 1000:.2f} mm over 5 rings, {pc.n} pts")

    # 2. Sphere: points must lie on the surface
    c0, rad = np.array([10.0, 0, 1.8]), 1.0
    cfg = replace(base, n_channels=41, vfov_lower_deg=-15, vfov_upper_deg=15, hfov_start_deg=-15, hfov_end_deg=15,
                  measurements_per_cycle=3600)
    pc = LidarSensor(cfg, SensorPose(position=(0, 0, 1.8)), seed=4).scan(Scene([Sphere(c0, rad, "concrete")]))
    res = np.linalg.norm(pc.xyz_world - c0, axis=1) - rad
    report("sphere surface residual", abs(np.median(res)) < 0.005 and np.percentile(np.abs(res), 90) < 0.05,
           f"median {np.median(res) * 1000:+.2f} mm, 90th pct |res| {np.percentile(np.abs(res), 90) * 1000:.1f} mm, {pc.n} pts")

    # 3. Inverse-square law + atmosphere on a wall facing the beam (noise off -> exact)
    cfg = replace(base, n_channels=1, vfov_upper_deg=0.0, hfov_start_deg=-0.5, hfov_end_deg=0.5,
                  measurements_per_cycle=3600, noise_enabled=False)
    worst = 0.0
    for D in (5, 10, 20, 40, 80):
        pc = LidarSensor(cfg, SensorPose(position=(0, 0, 0))).scan(Scene([Plane((D, 0, 0), (-1, 0, 0), "concrete")]))
        k = np.argmin(np.abs(pc.azimuth_deg))
        th = received_power(cfg, 0.30, 1.0, pc.range_true_m[k])
        worst = max(worst, abs(pc.power_w[k] / th - 1))
    report("inverse-square + Beer-Lambert", worst < 0.005, f"max deviation from theory {worst * 100:.3f} % (5..80 m)")

    # 4. Lambert cosine law
    worst = 0.0
    for phi in (0, 30, 60, 75):
        n = (-np.cos(np.radians(phi)), np.sin(np.radians(phi)), 0.0)
        pc = LidarSensor(cfg, SensorPose()).scan(Scene([Plane((10, 0, 0), n, "concrete")]))
        k = np.argmin(np.abs(pc.azimuth_deg))
        th = received_power(cfg, 0.30, np.cos(np.radians(phi)), pc.range_true_m[k])
        worst = max(worst, abs(pc.power_w[k] / th - 1))
    report("Lambert cosine law", worst < 0.01, f"max deviation from theory {worst * 100:.3f} % (0..75 deg)")

    # 5. Detection range: analytic R_max vs Monte-Carlo detection probability
    cfg = replace(base, n_channels=1, vfov_upper_deg=0.0, hfov_start_deg=-1, hfov_end_deg=1, measurements_per_cycle=36000)
    rmax = max_range_estimate(cfg, 0.10)
    probs = {}
    for f in (0.8, 1.0, 1.2):
        hits = trials = 0
        sensor = LidarSensor(cfg, SensorPose(), seed=7)
        for _ in range(5):
            pc = sensor.scan(Scene([Plane((f * rmax, 0, 0), (-1, 0, 0), replace(MATERIALS["asphalt"]))]))
            hits += pc.n
            trials += pc.n_beams
        probs[f] = hits / trials
    ok = probs[0.8] > 0.97 and 0.40 < probs[1.0] < 0.60 and probs[1.2] < 0.05
    report("detection range (rho=0.1)", ok,
           f"R_max={rmax:.1f} m; P_det at 0.8/1.0/1.2 R_max = {probs[0.8]:.3f}/{probs[1.0]:.3f}/{probs[1.2]:.3f}")

    # 6. Pulse gate / mixed pixels: two faces meeting at an edge.
    #    gap < c*tau/2 (0.75 m): sub-ray echoes merge -> intermediate "smeared" ranges appear.
    #    gap >> c*tau/2        : echoes stay separate -> every return is one face or the other.
    cfg = replace(base, n_channels=1, vfov_upper_deg=0.0, hfov_start_deg=-1, hfov_end_deg=1, measurements_per_cycle=36000,
                  noise_enabled=False)

    def step(gap):
        sc = Scene([OrientedBox((10.5, -5, 0), (1, 10, 10), 0, "concrete", "A"),
                    OrientedBox((10.5 + gap, 5, 0), (1, 10, 10), 0, "concrete", "B")])
        return LidarSensor(cfg, SensorPose()).scan(sc).range_m

    r_small, r_big = step(0.3), step(5.0)
    mid_small = np.sum((r_small > 10.01) & (r_small < 10.29))
    mid_big = np.sum((r_big > 10.1) & (r_big < 14.9))
    report("pulse-gate merging at an edge", mid_small >= 3 and mid_big == 0 and r_small.min() >= 10 - 1e-6 and r_small.max() <= (10.3 + 1e-6) / np.cos(np.radians(1.0)),
           f"gap 0.3 m -> {mid_small} merged echoes in between; gap 5 m -> {mid_big}")

    # 7. the shipped template parses and equals the code defaults
    try:
        c, p, sc, sim, out = build_all(json.loads(_strip_jsonc(CONFIG_TEMPLATE)))
        same = dataclasses.asdict(c) == dataclasses.asdict(LidarConfig()) and len(sc.objects) == 7
        report("config template == code defaults", same, f"{len(sc.objects)} objects, seed={sim['seed']}, outputs={out['save']}")
    except Exception as e:  # noqa: BLE001
        report("config template == code defaults", False, repr(e))

    # 8. bad configs are rejected with a pointer to the key
    base_raw = json.loads(_strip_jsonc(CONFIG_TEMPLATE))
    msgs = []
    for path_, val in (("sensor.n_chanels", 16), ("sensor.return_mode", "best"), ("sensor.optical_efficiency", 1.5),
                       ("scene.objects.0.material", "asphlat")):
        raw = json.loads(json.dumps(base_raw))
        d = raw
        parts = path_.split(".")
        for q in parts[:-1]:
            d = d[int(q)] if q.isdigit() else d[q]
        d[int(parts[-1]) if parts[-1].isdigit() else parts[-1]] = val
        try:
            build_all(raw)
            msgs.append(None)
        except ConfigError as e:
            msgs.append(str(e))
    report("config validation", all(m is not None for m in msgs) and "n_channels" in msgs[0] and "asphalt" in msgs[3],
           f"{sum(m is not None for m in msgs)}/4 bad configs rejected; e.g. {msgs[0]}")

    print("ALL TESTS PASSED" if ok_all else "SOME TESTS FAILED")
    return ok_all
