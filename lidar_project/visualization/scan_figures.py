"""Publication-style figures from an existing LiDAR PointCloud; never runs a scan.

The public functions import matplotlib only when called.  Pass sensor_position
explicitly for a moving sensor; a static pose can otherwise be recovered from
the paired local/world coordinates when a rigid transformation is identifiable.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from . import theme
from .scene_geometry import mesh_for

_PALETTE = theme.PALETTE
_STYLE = theme.MATPLOTLIB


def _names(pc: Any) -> list[str]:
    names = [str(s) for s in getattr(pc, "label_names", [])]
    ids = np.asarray(getattr(pc, "label_id", []), dtype=int)
    required = int(ids.max()) + 1 if ids.size else 0
    return names + [f"object_{i}" for i in range(len(names), required)]


def _human(name: str) -> str:
    known = {
        "wall_concrete": "Concrete wall",
        "wall_white": "White wall",
        "retro_box": "Reflective box",
    }
    if name in known:
        return known[name]
    return name.replace("_", " ").replace("-", " ").strip().capitalize()


def _is_ground(name: str) -> bool:
    tokens = name.lower().replace("-", "_").split("_")
    return any((t in {"ground", "floor", "road"} for t in tokens))


def _colors(names: list[str]) -> list[str]:
    result, j = ([], 0)
    for name in names:
        if _is_ground(name):
            result.append(theme.color("ground"))
        else:
            result.append(_PALETTE[j % len(_PALETTE)])
            j += 1
    return result


def _cloud(pc: Any):
    xyz = np.asarray(getattr(pc, "xyz_world", np.empty((0, 3))), float).reshape(-1, 3)
    ids = np.asarray(getattr(pc, "label_id", []), int)
    if len(ids) != len(xyz):
        ids = np.full(len(xyz), -1, dtype=int)
    good = np.all(np.isfinite(xyz), axis=1)
    return (xyz[good], ids[good])


def _sample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    return points[np.linspace(0, len(points) - 1, limit, dtype=int)]


def _sensor_position(pc: Any, supplied: Any):
    if supplied is not None:
        position = np.asarray(supplied, float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(
                "sensor_position must contain three finite world coordinates"
            )
        return position
    local = np.asarray(getattr(pc, "xyz_sensor", np.empty((0, 3))), float).reshape(
        -1, 3
    )
    world = np.asarray(getattr(pc, "xyz_world", np.empty((0, 3))), float).reshape(-1, 3)
    if len(local) != len(world) or len(local) < 3:
        return None
    valid = np.all(np.isfinite(local), axis=1) & np.all(np.isfinite(world), axis=1)
    local, world = (local[valid], world[valid])
    if len(local) < 3:
        return None
    take = np.linspace(0, len(local) - 1, min(len(local), 2048), dtype=int)
    a, b = (local[take], world[take])
    ac, bc = (a - a.mean(0), b - b.mean(0))
    if np.linalg.matrix_rank(ac) < 2:
        return None
    u, _, vt = np.linalg.svd(ac.T @ bc)
    fix = np.eye(3)
    fix[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ fix @ vt
    origin = b.mean(0) - a.mean(0) @ rotation
    residual = np.linalg.norm(a @ rotation + origin - b, axis=1)
    tolerance = 1e-07 * max(1.0, float(np.max(np.linalg.norm(bc, axis=1))))
    return origin if float(residual.max()) <= tolerance else None


def _window(xyz, ids, names, scene, sensor):
    """Cover every non-ground return and finite object, with ground as context."""
    ground_ids = [i for i, n in enumerate(names) if _is_ground(n)]
    foreground = xyz[~np.isin(ids, ground_ids)]
    pieces = [foreground] if len(foreground) else []
    for obj in getattr(scene, "objects", []) if scene is not None else []:
        if _is_ground(str(getattr(obj, "label", ""))):
            continue
        mesh = mesh_for(obj)
        if mesh is not None:
            vertices = np.asarray(mesh[0], float)
            if vertices.ndim == 2 and vertices.shape[1] == 3:
                vertices = vertices[np.all(np.isfinite(vertices), axis=1)]
                if len(vertices):
                    pieces.append(vertices)
    if sensor is not None:
        pieces.append(np.asarray(sensor)[None, :])
    if not pieces and len(xyz):
        pieces.append(xyz)
    context = (
        np.concatenate(pieces)
        if pieces
        else np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]])
    )
    lo, hi = (context.min(0), context.max(0))
    span = hi - lo
    padding = np.maximum(0.07 * span, np.array([0.5, 0.5, 0.25]))
    return (lo - padding, hi + padding)


def _legend_handles(names, colors, present, sensor):
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=5,
            color=colors[i],
            label=_human(names[i]),
        )
        for i in present
    ]
    if sensor is not None:
        handles.append(
            Line2D(
                [],
                [],
                marker="^",
                linestyle="",
                markersize=7,
                color=theme.color("#111111"),
                label="LiDAR position",
            )
        )
    return handles


def _draw_plan(ax, pc, scene=None, sensor_position=None, compact=False):
    xyz, ids = _cloud(pc)
    names, sensor = (_names(pc), _sensor_position(pc, sensor_position))
    colors = _colors(names)
    lo, hi = _window(xyz, ids, names, scene, sensor)
    present = []
    for i, name in enumerate(names):
        points = xyz[ids == i]
        if not len(points):
            continue
        present.append(i)
        ground = _is_ground(name)
        if ground:
            points = points[
                np.all((points[:, :2] >= lo[:2]) & (points[:, :2] <= hi[:2]), axis=1)
            ]
        points = _sample(points, 7000 if ground else 20000)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=1.3 if ground else 3.0,
            c=colors[i],
            alpha=0.35 if ground else 0.8,
            linewidths=0,
            rasterized=True,
            zorder=1 if ground else 3,
        )
    for obj in getattr(scene, "objects", []) if scene is not None else []:
        if type(obj).__name__ != "OrientedBox":
            continue
        corners = np.array(
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]]
        )
        corners = corners * np.asarray(obj.h) @ np.asarray(obj.R).T + np.asarray(obj.c)
        label = str(getattr(obj, "label", ""))
        color = colors[names.index(label)] if label in names else theme.color("#555555")
        ax.plot(corners[:, 0], corners[:, 1], "--", lw=1.1, color=color, zorder=4)
    if sensor is not None:
        ax.scatter(
            [sensor[0]],
            [sensor[1]],
            marker="^",
            s=65,
            c=theme.color("#111111"),
            edgecolors=theme.color("white"),
            linewidths=0.6,
            zorder=6,
        )
    ax.set(
        xlim=(lo[0], hi[0]),
        ylim=(lo[1], hi[1]),
        xlabel="World x (m)",
        ylabel="World y (m)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=theme.color("#E6E8EB"), lw=0.6, zorder=0)
    notes = []
    if any((_is_ground(n) for n in names)):
        notes.append("Ground displayed within object extent")
    if scene is not None and any(
        (type(o).__name__ == "OrientedBox" for o in getattr(scene, "objects", []))
    ):
        notes.append("Dashed outlines: configured boxes")
    ax.set_title(
        "Scene plan view", loc="left", pad=(31 if compact else 41) if notes else 8
    )
    handles = _legend_handles(names, colors, present, sensor)
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncol=(3 if compact else 4) if len(handles) > 3 else len(handles),
            frameon=False,
            fontsize=8 if compact else 11,
            columnspacing=1.1,
            handletextpad=0.4,
        )
    if notes:
        ax.text(
            0.0,
            1.015,
            "\n".join(notes),
            transform=ax.transAxes,
            fontsize=8 if compact else 11,
            va="bottom",
            color=theme.color("#666666"),
        )
    if not len(xyz):
        ax.text(
            0.5,
            0.5,
            "No detected returns",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=theme.color("#555555"),
        )


def _draw_objects(ax, pc, scene=None, sensor_position=None, compact=False):
    xyz, ids = _cloud(pc)
    names, sensor = (_names(pc), _sensor_position(pc, sensor_position))
    colors = _colors(names)
    lo, hi = _window(xyz, ids, names, None, sensor)
    present = []
    for i, name in enumerate(names):
        points = xyz[ids == i]
        if not len(points) or _is_ground(name):
            continue
        present.append(i)
        points = _sample(points, 16000)
        ax.scatter(
            *points.T,
            s=2.0,
            c=colors[i],
            alpha=0.85,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
    if sensor is not None:
        ax.scatter(
            *sensor[:, None],
            marker="^",
            s=50,
            c=theme.color("#111111"),
            depthshade=False,
        )
    if not present:
        ax.text2D(
            0.5,
            0.5,
            "No non-ground returns",
            transform=ax.transAxes,
            ha="center",
            color=theme.color("#555555"),
        )
    ax.set(
        xlim=(lo[0], hi[0]),
        ylim=(lo[1], hi[1]),
        zlim=(lo[2], hi[2]),
        xlabel="World x (m)",
        ylabel="World y (m)",
        zlabel="World height z (m)",
    )
    ax.set_box_aspect(np.maximum(hi - lo, 0.01))
    ax.view_init(elev=24, azim=-125)
    from matplotlib.ticker import MaxNLocator

    ax.zaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    ax.set_title("3D object returns (ground omitted)", loc="left", pad=10)
    ax.tick_params(labelsize=8 if compact else 11, pad=1)
    ax.xaxis.labelpad = 6
    ax.yaxis.labelpad = 6
    ax.zaxis.labelpad = 6
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis._axinfo["grid"]["color"] = (0.88, 0.89, 0.9, 1.0)
    handles = _legend_handles(names, colors, present, sensor)
    if handles and (not compact):
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            frameon=False,
            fontsize=11,
        )


def _axis_from_samples(length, sample_index, angles):
    """Recover the regular grid from returned samples without assuming its FOV."""
    sample_index, angles = (np.asarray(sample_index), np.asarray(angles, float))
    good = np.isfinite(angles)
    indices = np.unique(sample_index[good])
    if len(indices) == 0:
        return (np.arange(length, dtype=float), False)
    centers = np.array([np.median(angles[good & (sample_index == i)]) for i in indices])
    if length == 1:
        return (centers[:1], True)
    if len(indices) < 2:
        return (np.arange(length, dtype=float), False)
    step = float(np.median(np.diff(centers) / np.diff(indices)))
    offset = float(np.median(centers - step * indices))
    expected = offset + step * sample_index[good]
    if not np.allclose(expected, angles[good], atol=1e-05, rtol=1e-07):
        return (np.arange(length, dtype=float), False)
    return (offset + step * np.arange(length), True)


def _range_data(pc, cfg):
    image = np.asarray(getattr(pc, "range_image", np.empty((0, 0))), float)
    if image.ndim != 2 or min(image.shape, default=0) == 0:
        return (np.empty((0, 0)), np.array([]), np.array([]), False, False)
    nr, nc = image.shape
    if cfg is not None:
        az, el = (
            np.asarray(cfg.azimuths_deg(), float),
            np.asarray(cfg.elevations_deg(), float),
        )
        if len(az) == nc and len(el) == nr:
            return (image, az, el, True, True)
    cols, rows = np.nonzero(np.isfinite(image.T))
    azimuth = np.asarray(getattr(pc, "azimuth_deg", []), float)
    elevation = np.asarray(getattr(pc, "elevation_deg", []), float)
    if len(cols) == len(azimuth) == len(elevation):
        az, ax_ok = _axis_from_samples(nc, cols, azimuth)
        el, ey_ok = _axis_from_samples(nr, rows, elevation)
        return (image, az, el, ax_ok, ey_ok)
    return (image, np.arange(nc), np.arange(nr), False, False)


def _edges(centers):
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mids = (centers[1:] + centers[:-1]) / 2
    return np.r_[
        centers[0] - (mids[0] - centers[0]),
        mids,
        centers[-1] + (centers[-1] - mids[-1]),
    ]


def _draw_range(ax, pc, cfg=None, compact=False):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    image, az, el, az_known, el_known = _range_data(pc, cfg)
    if image.size == 0:
        ax.text(
            0.5,
            0.5,
            "No angular range image available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return
    finite = image[np.isfinite(image)]
    upper = float(finite.max()) if finite.size else 1.0
    upper = max(upper, 1e-09)
    cmap = plt.get_cmap(theme.colormap("range")).copy()
    cmap.set_bad(theme.color("missing"))
    xe, ye = (_edges(az), _edges(el))
    im = ax.imshow(
        np.ma.masked_invalid(image),
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        extent=(xe[0], xe[-1], ye[0], ye[-1]),
        cmap=cmap,
        vmin=0,
        vmax=upper,
        rasterized=True,
    )
    ax.set_title("Angular range image", loc="left")
    ax.set_xlabel("Azimuth (deg)" if az_known else "Azimuth sample index")
    ax.set_ylabel("Elevation (deg)" if el_known else "Channel index")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    if len(el) == 1:
        ax.set_yticks(el)
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    cb = ax.figure.colorbar(im, ax=ax, pad=0.025, fraction=0.035, aspect=22)
    cb.set_label("Measured range (m)")
    cb.ax.tick_params(labelsize=8 if compact else 11)
    ax.text(
        1,
        1.04,
        "Gray = no return",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8 if compact else 11,
        color=theme.color("#555555"),
    )
    if not finite.size:
        ax.text(
            0.5,
            0.5,
            "No detected returns",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=theme.color("#555555"),
        )


def _save(fig, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=260, bbox_inches="tight", pad_inches=0.1)
    return str(target)


def save_overview(pc, path, title="", *, scene=None, cfg=None, sensor_position=None):
    """Save a two-row overview, compatible with plot_cloud(pc, path, title).

    Top panels show object returns in 3D and in plan; the bottom range image
    spans the full width.  Pass cfg for exact angle axes even on an empty scan.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(11.5, 8.2), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=(1.45, 1), hspace=0.1, wspace=0.08)
        _draw_objects(
            fig.add_subplot(grid[0, 0], projection="3d"),
            pc,
            scene,
            sensor_position,
            compact=True,
        )
        _draw_plan(
            fig.add_subplot(grid[0, 1]), pc, scene, sensor_position, compact=True
        )
        _draw_range(fig.add_subplot(grid[1, :]), pc, cfg, compact=True)
        if title:
            fig.suptitle(title, fontsize=13)
        result = _save(fig, path)
        plt.close(fig)
        return result


def save_report_figures(pc, scene, cfg, directory, *, sensor_position=None) -> dict:
    """Write separate PNG/PDF figures from existing data, without simulation.

    Returns {figure_name: {"png": absolute_or_requested_path, "pdf": path}}.
    Names are plan_view, object_cloud, range_image, and return_counts.
    Counts are numbers of retained simulated returns, not detection accuracy.
    """
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {}

    def finish(name, fig):
        outputs[name] = {
            ext: _save(fig, directory / f"{name}.{ext}") for ext in ("png", "pdf")
        }
        plt.close(fig)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
        _draw_plan(ax, pc, scene, sensor_position)
        finish("plan_view", fig)
        fig = plt.figure(figsize=(7, 3.7), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")
        _draw_objects(ax, pc, scene, sensor_position)
        finish("object_cloud", fig)
        fig, ax = plt.subplots(figsize=(7, 2.7), layout="constrained")
        _draw_range(ax, pc, cfg)
        finish("range_image", fig)
        names = _names(pc)
        colors = _colors(names)
        ids = np.asarray(getattr(pc, "label_id", []), int)
        counts = np.array([np.count_nonzero(ids == i) for i in range(len(names))])
        fig, ax = plt.subplots(
            figsize=(7, max(2.7, 0.3 * len(names) + 1.1)), layout="constrained"
        )
        if names:
            y = np.arange(len(names))
            ax.barh(y, counts, color=colors, height=0.65, edgecolor="none")
            ax.set_yticks(y, [_human(n) for n in names])
            ax.invert_yaxis()
            upper = max(1, int(counts.max()))
            ax.set_xlim(0, 1.16 * upper)
            for i, count in enumerate(counts):
                ax.text(
                    count + upper * 0.015, i, f"{count:,}", va="center", fontsize=11
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No object labels available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_yticks([])
        ax.set_title("Retained returns by object", loc="left")
        ax.set_xlabel("Number of returns in this scan")
        ax.xaxis.grid(True, color=theme.color("#E6E8EB"), lw=0.6)
        ax.set_axisbelow(True)
        finish("return_counts", fig)
    return outputs
