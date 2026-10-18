"""Pinhole scene rendering and projection for visual overlays only."""
from __future__ import annotations
import numpy as np
from .. import api as ls
from . import theme


class FirstPerson:
    """Pinhole camera at the sensor, looking along the robot heading; rendered by ray casting."""

    def __init__(self, scene, eye_h, w=240, h=135, hfov_deg=90.0, pitch_deg=-6.0):
        self.scene, self.eye_h, self.w, self.h = (scene, eye_h, w, h)
        self.f = 0.5 * w / np.tan(np.radians(hfov_deg) / 2)
        self.hfov = np.radians(hfov_deg)
        u = (w / 2 - (np.arange(w) + 0.5)) / self.f
        v = (h / 2 - (np.arange(h) + 0.5)) / self.f
        U, V = np.meshgrid(u, v)
        p = np.radians(pitch_deg)
        self.d_cam = np.stack(
            [np.cos(p) - V * np.sin(p), U, np.sin(p) + V * np.cos(p)], -1
        ).reshape(-1, 3)
        self.d_cam /= np.linalg.norm(self.d_cam, axis=1, keepdims=True)
        self.pitch = p
        self.albedo = np.array(
            [
                theme.THEME["camera"]["albedo"].get(o.material.name, (0.6, 0.6, 0.6))
                for o in scene.objects
            ]
            + [(0, 0, 0)]
        )
        self.is_ground = np.array(
            [isinstance(o, ls.Plane) for o in scene.objects] + [False]
        )
        self.sun = np.array([0.4, 0.3, 0.87]) / np.linalg.norm([0.4, 0.3, 0.87])

    def _rot(self, th):
        c, s = (np.cos(th), np.sin(th))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def render(self, x, y, th, alpha):
        R = self._rot(th)
        d = self.d_cam @ R.T
        o = np.tile([x, y, self.eye_h], (len(d), 1))
        t, idx, n = self.scene.cast(o, d)
        hit = idx >= 0
        i = np.where(hit, idx, len(self.scene.objects))
        col = self.albedo[i].copy()
        p = o + np.where(hit, t, 0)[:, None] * d
        g = hit & self.is_ground[i]
        tile = (np.floor(p[:, 0]) + np.floor(p[:, 1])) % 2 == 0
        col[g] *= np.where(tile[g], 1.0, 0.82)[:, None]
        shade = 0.35 + 0.65 * np.clip(np.einsum("ij,j->i", n, self.sun), 0, 1)
        col = col * np.where(hit, shade, 1.0)[:, None]
        col[~hit] = (
            np.asarray(theme.THEME["camera"]["sky"])
            - 0.25 * np.clip(d[~hit, 2], 0, 1)[:, None]
        )
        tr = np.exp(-alpha * np.where(hit, t, 300.0))
        col = (
            col * tr[:, None]
            + np.asarray(theme.THEME["camera"]["fog"]) * (1 - tr)[:, None]
        )
        depth = np.where(hit, t, np.inf)
        return (
            np.clip(col, 0, 1).reshape(self.h, self.w, 3),
            depth.reshape(self.h, self.w),
        )

    def project(self, pts, x, y, th, depth):
        """World points -> pixel coordinates (only points in view and not hidden behind the render)."""
        R = self._rot(th)
        rel = (pts - [x, y, self.eye_h]) @ R
        cp, sp = (np.cos(self.pitch), np.sin(self.pitch))
        fwd = rel[:, 0] * cp + rel[:, 2] * sp
        up = -rel[:, 0] * sp + rel[:, 2] * cp
        ok = fwd > 0.3
        col = self.w / 2 - self.f * rel[:, 1] / np.where(ok, fwd, 1)
        row = self.h / 2 - self.f * up / np.where(ok, fwd, 1)
        ok &= (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        ci, ri = (col[ok].astype(int), row[ok].astype(int))
        dist = np.linalg.norm(rel[ok], axis=1)
        vis = dist <= depth[ri, ci] * 1.02 + 0.15
        sel = np.flatnonzero(ok)[vis]
        return (col[sel], row[sel], np.linalg.norm(rel[sel], axis=1))
