"""Numerical LiDAR-in-the-loop navigation, independent of presentation.

The controller and scoring follow the original navigation implementation.  Full
step history additionally retains numerical observations for offline replay;
recording it does not consume random numbers or alter the controller.
"""
from __future__ import annotations

import copy

import numpy as np

from .. import api as ls


def wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def footprint_distance(obj, xy, z_hi):
    """Horizontal distance from point xy to the obstacle footprint; inf if the obstacle is entirely
    above the robot (z > z_hi) or is the ground plane."""
    xy = np.asarray(xy, float)
    if isinstance(obj, ls.OrientedBox):
        if obj.c[2] - obj.h[2] > z_hi:
            return np.inf
        q = obj.R[:2, :2].T @ (xy - obj.c[:2])
        return float(np.linalg.norm(np.maximum(np.abs(q) - obj.h[:2], 0.0)))
    if isinstance(obj, ls.VerticalCylinder):
        if obj.base[2] > z_hi:
            return np.inf
        return max(float(np.linalg.norm(xy - obj.base[:2])) - obj.r, 0.0)
    if isinstance(obj, ls.Sphere):
        if obj.c[2] - obj.r > z_hi:
            return np.inf
        return max(float(np.linalg.norm(xy - obj.c[:2])) - obj.r, 0.0)
    return np.inf


def split_returns(pc, z_ground, z_top):
    """Height-based segmentation of a sweep: (obstacle xy, ground xy, obstacle labels for scoring)."""
    z = pc.xyz_world[:, 2]
    obs = (z >= z_ground) & (z <= z_top)
    gnd = z < z_ground
    return pc.xyz_world[obs, :2], pc.xyz_world[gnd, :2], pc.label_id[obs]


def free_path_lengths(pts, xy, headings, r_inf, lookahead):
    """Distance a disc of radius r_inf can translate from xy along each heading before touching a point."""
    L = np.full(len(headings), float(lookahead))
    if len(pts) == 0:
        return L
    q = pts - xy                                                     # (M,2)
    u = np.stack([np.cos(headings), np.sin(headings)], 1)            # (H,2)
    along = q @ u.T                                                  # (M,H)
    perp = np.abs(q[:, 0, None] * u[None, :, 1] - q[:, 1, None] * u[None, :, 0])
    hit = (perp < r_inf) & (along > -r_inf)
    contact = along - np.sqrt(np.maximum(r_inf ** 2 - perp ** 2, 0.0))
    contact = np.where(hit, contact, np.inf)
    return np.clip(contact.min(axis=0), 0.0, lookahead)


class OccupancyGrid:
    """Log-odds occupancy grid, l <- l + l_inv(z) (binary Bayes filter with a static-world prior 0.5)."""

    def __init__(self, g):
        self.res = g["resolution_m"]
        self.x0, self.x1, self.y0, self.y1 = g["extent"]
        self.nx = int(round((self.x1 - self.x0) / self.res))
        self.ny = int(round((self.y1 - self.y0) / self.res))
        self.l = np.zeros((self.ny, self.nx))
        self.l_occ, self.l_free, self.l_clamp = g["l_occ"], g["l_free"], g["l_clamp"]

    def _cells(self, xy):
        ix = np.floor((xy[:, 0] - self.x0) / self.res).astype(int)
        iy = np.floor((xy[:, 1] - self.y0) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return np.unique(iy[ok] * self.nx + ix[ok])

    def update(self, origin, obs, gnd, rng, max_rays=3000):
        ends = np.vstack([obs, gnd])
        if len(ends) > max_rays:
            ends = ends[rng.choice(len(ends), max_rays, replace=False)]
        free = [gnd]
        if len(ends):
            d = np.linalg.norm(ends - origin, axis=1)
            n = int(np.ceil(d.max() / (0.5 * self.res)))
            s = np.linspace(0.0, 1.0, n, endpoint=False)[None, :, None]
            ray = origin + s * (ends - origin)[:, None, :]               # samples up to (not incl.) the end
            keep = (s[..., 0] * d[:, None]) < (d[:, None] - self.res)
            free.append(ray[keep])
        occ = self._cells(obs) if len(obs) else np.array([], int)
        fr = np.setdiff1d(self._cells(np.vstack(free)), occ)
        flat = self.l.ravel()
        flat[fr] += self.l_free
        flat[occ] += self.l_occ
        np.clip(self.l, -self.l_clamp, self.l_clamp, out=self.l)

    def prob(self):
        return 1.0 / (1.0 + np.exp(-self.l))


def build_world(nav, scen):
    base = nav.get("_base_config_snapshot")
    if base is None:
        base = ls.load_config_file(nav["base_config"])
    raw = copy.deepcopy(base)
    raw["sensor"] = {**raw.get("sensor", {}), **nav.get("sensor_overrides", {}), **scen.get("sensor", {})}
    raw["scene"]["objects"] = list(raw["scene"]["objects"]) + list(nav.get("extra_objects", []))
    cfg, _, scene, _, _ = ls.build_all(raw)
    return cfg, scene


def run(nav, scen, verbose=True, on_step=None):
    """Simulate one scenario. on_step(state) is called after every control step (live viewer);
    returning False from it stops the run."""
    cfg, scene = build_world(nav, scen)
    rb, pl = nav["robot"], nav["planner"]
    seed = nav["output"].get("seed", 1)
    sensor = ls.LidarSensor(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    grid = OccupancyGrid(nav["grid"])

    x, y = map(float, rb["start"])
    th = np.radians(rb["start_heading_deg"])
    goal = np.asarray(rb["goal"], float)
    r_inf = rb["radius_m"] + pl["safety_margin_m"]
    dt, wmax = rb["dt"], np.radians(rb["omega_max_deg_s"])
    H = np.linspace(-np.pi, np.pi, pl["n_headings"], endpoint=False)
    v = w = 0.0
    obstacle_ids = [i for i, o in enumerate(scene.objects) if not isinstance(o, ls.Plane)]
    first_det = {scene.labels[i]: np.nan for i in obstacle_ids}
    log, frames, history = [], [], []
    status = "timeout"

    for k in range(rb["max_steps"]):
        # ---- 1. sense (pose and motion during the sweep come from the robot state) ----
        sensor.pose = ls.SensorPose(position=(x, y, nav["mount_height_m"]), yaw_deg=np.degrees(th),
                                    velocity=(v * np.cos(th), v * np.sin(th), 0.0), yaw_rate_deg_s=np.degrees(w))
        pc = sensor.scan(scene)
        # ---- 2. perceive ----
        obs, gnd, obs_lab = split_returns(pc, pl["z_ground_m"], rb["height_m"])
        grid.update(np.array([x, y]), obs, gnd, rng)
        # ---- scoring only: true clearance and first-detection ranges ----
        d_true = {scene.labels[i]: footprint_distance(scene.objects[i], (x, y), rb["height_m"]) for i in obstacle_ids}
        for i in obstacle_ids:
            lab = scene.labels[i]
            if np.isnan(first_det[lab]) and np.count_nonzero(obs_lab == i) >= 3:
                first_det[lab] = d_true[lab]
        true_clear = min(d_true.values())
        perc_clear = float(np.min(np.linalg.norm(obs - [x, y], axis=1))) if len(obs) else np.inf
        # ---- 3. plan ----
        to_goal = goal - [x, y]
        dist_goal = float(np.linalg.norm(to_goal))
        if dist_goal < rb["goal_tolerance_m"]:
            status = "goal"
            log.append((k * dt, x, y, th, 0.0, 0.0, true_clear, perc_clear, pc.n, len(obs), dist_goal))
            history.append(dict(k=k, pc=pc, obs=obs, gnd=gnd, L=None, psi=th,
                                grid_l=grid.l.copy(), status=status))
            if on_step:
                on_step(dict(k=k, log=log, pc=pc, obs=obs, gnd=gnd, L=None, psi=th, grid=grid, cfg=cfg,
                             scene=scene, H=H, status=status))
            break
        psi_goal = np.arctan2(to_goal[1], to_goal[0])
        L = free_path_lengths(obs, np.array([x, y]), H, r_inf, pl["lookahead_m"])
        need = min(pl["d_stop_m"] + 0.5, dist_goal)
        J = (pl["w_goal"] * np.abs(wrap(H - psi_goal)) + pl["w_turn"] * np.abs(wrap(H - th))
             - pl["w_free"] * L / pl["lookahead_m"])
        J = np.where(L >= need, J, np.inf)
        best = int(np.argmin(J)) if np.isfinite(J).any() else int(np.argmax(L))
        psi = H[best]
        w = float(np.clip(pl["k_omega"] * wrap(psi - th), -wmax, wmax))
        L_now = free_path_lengths(obs, np.array([x, y]), np.array([th]), r_inf, pl["lookahead_m"])[0]
        scale = np.clip((L_now - pl["d_stop_m"]) / (pl["d_slow_m"] - pl["d_stop_m"]), 0.0, 1.0)
        v = float(rb["v_max"] * scale * max(np.cos(wrap(psi - th)), 0.0))
        v = min(v, max(L_now - pl["d_stop_m"], 0.0) / dt, dist_goal / dt)   # safety bound (Theorem 3)
        log.append((k * dt, x, y, th, v, w, true_clear, perc_clear, pc.n, len(obs), dist_goal))
        frames.append(dict(obs=obs, gnd=gnd[::4], L=L, psi=psi))
        history.append(dict(k=k, pc=pc, obs=obs, gnd=gnd, L=L, psi=psi,
                            grid_l=grid.l.copy(), status="running"))
        if on_step and on_step(dict(k=k, log=log, pc=pc, obs=obs, gnd=gnd, L=L, psi=psi, grid=grid, cfg=cfg,
                                    scene=scene, H=H, status="running")) is False:
            status = "stopped"
            break
        if true_clear < rb["radius_m"]:
            status = "collision"
            break
        # ---- 4. move: unicycle, forward Euler (translate, then rotate) ----
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th = float(wrap(th + w * dt))

    log = np.array(log)
    if history:
        history[-1]["status"] = status
    res = dict(name=scen["name"], cfg=cfg, scene=scene, log=log, frames=frames, grid=grid,
               first_det=first_det, status=status, H=H, history=history,
               replay_available=bool(history), grid_available=True)
    if verbose:
        steps = len(log)
        path = float(np.sum(np.linalg.norm(np.diff(log[:, 1:3], axis=0), axis=1)))
        print(f"[{scen['name']:>10}] {status:9s} steps={steps:3d}  t={log[-1, 0]:5.1f} s  path={path:5.1f} m  "
              f"min true clearance={log[:, 6].min() - rb['radius_m']:.2f} m (surface gap)  "
              f"alpha={cfg.extinction_per_m:.3f} 1/m  returns/scan={log[:, 8].mean():.0f}")
    return res
