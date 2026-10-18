"""Display geometry adapters; physical primitives contain intersections only."""
from __future__ import annotations
import numpy as np
from .. import api as ls


def _sphere_mesh(obj, nu=32, nv=16):
    th, ph = np.meshgrid(
        np.linspace(0, np.pi, nv + 1),
        np.linspace(0, 2 * np.pi, nu, endpoint=False),
        indexing="ij",
    )
    v = obj.c + obj.r * np.stack(
        [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1
    ).reshape(-1, 3)
    f = []
    for i in range(nv):
        for j in range(nu):
            a, b = (i * nu + j, i * nu + (j + 1) % nu)
            c, e = ((i + 1) * nu + j, (i + 1) * nu + (j + 1) % nu)
            f += [[a, b, e], [a, e, c]]
    return (v, np.array(f))


def _box_mesh(obj):
    corners = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        float,
    )
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ]
    return (corners * obj.h @ obj.R.T + obj.c, np.array(faces))


def _cylinder_mesh(obj, n=24):
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([obj.r * np.cos(ph), obj.r * np.sin(ph)], 1)
    top = obj.base + [0, 0, obj.hgt]
    v = np.vstack(
        [
            np.c_[ring, np.zeros(n)] + obj.base,
            np.c_[ring, np.full(n, obj.hgt)] + obj.base,
            [obj.base],
            [top],
        ]
    )
    f = []
    for j in range(n):
        k = (j + 1) % n
        f += [
            [j, k, n + k],
            [j, n + k, n + j],
            [2 * n, k, j],
            [2 * n + 1, n + j, n + k],
        ]
    return (v, np.array(f))


def mesh_for(obj):
    """Return (vertices, triangle indices) for a finite primitive, otherwise None."""
    if isinstance(obj, ls.Sphere):
        return _sphere_mesh(obj)
    if isinstance(obj, ls.OrientedBox):
        return _box_mesh(obj)
    if isinstance(obj, ls.VerticalCylinder):
        return _cylinder_mesh(obj)
    return None


def footprint_outline(obj):
    """Closed 2D polygon of an obstacle footprint, for drawing."""
    if isinstance(obj, ls.OrientedBox):
        h = obj.h[:2]
        loc = np.array(
            [[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]], [-h[0], -h[1]]]
        )
        return loc @ obj.R[:2, :2].T + obj.c[:2]
    if isinstance(obj, (ls.VerticalCylinder, ls.Sphere)):
        c = obj.base[:2] if isinstance(obj, ls.VerticalCylinder) else obj.c[:2]
        a = np.linspace(0, 2 * np.pi, 40)
        return c + obj.r * np.stack([np.cos(a), np.sin(a)], 1)
    return None
