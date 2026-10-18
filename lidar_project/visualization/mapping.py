"""Saved-data 3D motion and measurement-map comparison views.

Only archived trajectory states, full 3D scans, and measurement-map arrays are
read. Rendering never invokes sensing, planning, or map construction.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from . import theme
from .scene_geometry import mesh_for


def _palette(i):
    return theme.PALETTE[i % len(theme.PALETTE)]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _take(n, limit):
    return np.arange(n) if n <= limit else np.linspace(0, n - 1, limit, dtype=int)


def _columns(points, intensity=None):
    points = np.asarray(points, float).reshape(-1, 3)
    out = {axis: np.round(points[:, j], 4) for j, axis in enumerate(("x", "y", "z"))}
    if intensity is not None:
        out["intensity"] = np.round(np.asarray(intensity, float), 5)
    return out


def _load_cases(directory):
    from ..trajectory.storage import load_result
    from ..mapping import load_map

    comparison = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    cases = []
    for spec in comparison["cases"]:
        run_directory = Path(spec["run_directory"])
        if not run_directory.is_absolute():
            run_directory = directory / run_directory
        result, mapping = load_result(run_directory), load_map(run_directory)
        if not result.get("scans_available", False):
            raise ValueError(
                f"{spec['label']} has no retained full 3D scans; a current scan view cannot be reconstructed from XY obstacle points"
            )
        cases.append({"spec": spec, "result": result, "mapping": mapping})
    if not cases:
        raise ValueError("comparison.json contains no environments")
    return comparison, cases


def _truth_meshes(result):
    from .. import api as ls

    objects = []
    unknown = {
        obj["label"] for obj in result["config"]["world"].get("unexpected_objects", [])
    }
    for obj in ls.build_scene(
        {"materials": {}, "objects": result["world_objects"]}
    ).objects:
        mesh = mesh_for(obj)
        if mesh is None:
            continue
        vertices, faces = mesh
        objects.append(
            {
                "name": obj.label,
                "unknown": obj.label in unknown,
                "vertices": np.round(vertices, 4),
                "faces": faces,
            }
        )
    return objects


def _payload(comparison, cases):
    payload = {
        "cases": [],
        "time": [],
        "colors": {
            "background": theme.color("white"),
            "text": theme.color("figure_text"),
            "grid": theme.color("figure_grid"),
            "unknown": theme.color("missing"),
            "low": _palette(0),
            "surface": _palette(1),
            "path": theme.color("figure_text"),
            "scan": _palette(3),
            "robot": theme.color("figure_sensor"),
            "sensor": _palette(0),
            "truth": theme.color("ground"),
            "unexpected": _palette(1),
        },
        "font": theme.THEME["web"]["font_family"],
    }
    times, z_values = [], []
    for item in cases:
        spec, result, mapping = item["spec"], item["result"], item["mapping"]
        metadata = mapping["metadata"]
        points = np.asarray(mapping["map_xyz"])
        selection = _take(len(points), 30000)
        cloud = _columns(
            points[selection], np.asarray(mapping["map_intensity"])[selection]
        )
        cloud["step"] = np.asarray(mapping["map_first_seen"])[selection]
        scans = []
        for scan in result["scans"]:
            xyz = np.asarray(scan["xyz_world"])
            roi = metadata.get("roi_extent_m", metadata["extent_m"])
            inside = (
                (xyz[:, 0] >= roi[0])
                & (xyz[:, 0] < roi[1])
                & (xyz[:, 1] >= roi[2])
                & (xyz[:, 1] < roi[3])
            )
            candidates = np.flatnonzero(inside)
            keep = candidates[_take(len(candidates), 800)]
            scans.append(_columns(xyz[keep], np.asarray(scan["intensity"])[keep]))
        robot = result["config"]["robot"]
        payload["cases"].append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "description": spec.get("description", ""),
                "metrics": spec["metrics"],
                "aggregate": spec.get("aggregate", {}),
                "states": result["states"],
                "scans": scans,
                "cloud": cloud,
                "scan_counts": [len(scan["xyz_world"]) for scan in result["scans"]],
                "point_counts": np.cumsum(
                    np.bincount(
                        np.asarray(mapping["map_first_seen"], int),
                        minlength=len(result["scans"]),
                    )
                ),
                "ground_first": np.asarray(mapping["ground_first_seen"]),
                "surface_first": np.asarray(mapping["obstacle_first_seen"]),
                "extent": metadata["extent_m"],
                "resolution": metadata["resolution_m"],
                "ground_threshold": metadata["ground_threshold_m"],
                "roi_extent": metadata.get("roi_extent_m", metadata["extent_m"]),
                "map_cells": int(np.prod(metadata["grid_shape"])),
                "map_points": int(len(points)),
                "reference": result["reference"]["xy"],
                "meshes": _truth_meshes(result),
                "radius": robot["radius_m"],
                "envelope_height": robot["height_m"],
                "body_height": min(0.3, robot["height_m"]),
                "sensor_height": result["config"]["mount_height_m"],
                "status": result["status"],
                "seed": result["seed"],
            }
        )
        times.append(np.asarray(result["states"]["t"], float))
        if len(points):
            z_values.append(points[:, 2])
    payload["time"] = np.unique(np.round(np.concatenate(times), 8))
    z = np.concatenate(z_values) if z_values else np.array([0.0, 1.0])
    payload["z_limits"] = [min(-0.08, float(z.min())), max(1.2, float(z.max()))]
    from plotly.colors import get_colorscale

    payload["height_scale"] = get_colorscale(theme.colormap("range"))
    return payload


def _viewer(comparison, cases, output):
    from plotly.offline import get_plotlyjs

    data = json.dumps(
        _jsonable(_payload(comparison, cases)),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    replacements = {
        "__STYLE__": theme.stylesheet("mapping.css"),
        "__PLOTLY__": get_plotlyjs(),
        "__DATA__": data,
        "__LOGIC__": theme.asset("static/mapping.js"),
    }
    page = re.sub(
        r"__[A-Z_]+__",
        lambda match: replacements[match.group(0)],
        theme.asset("templates/mapping.html"),
    )
    path = output / "mapping_interactive.html"
    path.write_text(page, encoding="utf-8")
    return {"html": str(path.resolve())}


def _grid_colors():
    from matplotlib.colors import ListedColormap, BoundaryNorm

    return ListedColormap(
        [theme.color("missing"), _palette(0), _palette(1)]
    ), BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)


def _representative_label(cases):
    seeds = sorted({item["result"]["seed"] for item in cases})
    return f"Seed {seeds[0]}" if len(seeds) == 1 else "Representative runs"


def _comparison(cases, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from ..mapping import observed_grid

    fig, axes = plt.subplots(
        1, len(cases), figsize=(11.5, 3.7), layout="constrained", squeeze=False
    )
    cmap, norm = _grid_colors()
    for ax, item in zip(axes[0], cases):
        spec, result, mapping = item["spec"], item["result"], item["mapping"]
        extent = mapping["metadata"]["extent_m"]
        ax.imshow(
            observed_grid(mapping, None),
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        reference, actual = np.asarray(result["reference"]["xy"]), np.asarray(
            result["states"]["xy"]
        )
        ax.plot(
            reference[:, 0], reference[:, 1], "--", color=theme.color("white"), lw=1.2
        )
        ax.plot(actual[:, 0], actual[:, 1], color=theme.color("figure_text"), lw=1.3)
        ax.plot(*actual[0], "o", ms=4, color=theme.color("figure_sensor"))
        ax.plot(*reference[-1], "*", ms=7, color=theme.color("figure_sensor"))
        ax.set_title(spec["label"], loc="left", fontsize=11)
        ax.set(xlabel="World x (m)", ylabel="World y (m)")
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        ax.tick_params(labelsize=8.5)
        metrics = spec["metrics"]
        status = {
            "goal": "Goal reached",
            "collision": "Collision",
            "timeout": "Time limit",
        }.get(result["status"], result["status"])
        text = (
            f"Seed {result['seed']}: {status}; end {metrics['elapsed_s']:.1f} s; path {metrics['path_length_m']:.1f} m\n"
            f"Minimum gap {metrics['swept_min_surface_gap_m']:.2f} m; observed cells {100*metrics['observed_fraction']:.1f}%"
        )
        ax.text(
            0,
            -0.24,
            text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            linespacing=1.6,
        )
    handles = [
        Patch(color=theme.color("missing"), label="Unknown"),
        Patch(color=_palette(0), label="Observed low surfaces"),
        Patch(color=_palette(1), label="Observed elevated surfaces"),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=9
    )
    fig.suptitle(
        f"Final observed endpoint maps · {_representative_label(cases)}\n"
        "White dashed: planned route · Black solid: executed path · Circle: start · Star: goal",
        fontsize=10,
    )
    path = output / "mapping_comparison.png"
    fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return {"png": str(path.resolve())}


def _comparison_3d(cases, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig = plt.figure(figsize=(11.5, 4), layout="constrained")
    all_z = [
        np.asarray(item["mapping"]["map_xyz"])[:, 2]
        for item in cases
        if len(item["mapping"]["map_xyz"])
    ]
    values = np.concatenate(all_z) if all_z else np.array([0.0, 1.0])
    lower, upper = min(0.0, float(values.min())), max(1.2, float(values.max()))
    axes, artist = [], None
    for i, item in enumerate(cases):
        ax = fig.add_subplot(1, len(cases), i + 1, projection="3d")
        axes.append(ax)
        points = np.asarray(item["mapping"]["map_xyz"])
        points = points[_take(len(points), 30000)]
        if len(points):
            artist = ax.scatter(
                *points.T,
                c=points[:, 2],
                cmap=theme.colormap("range"),
                vmin=lower,
                vmax=upper,
                s=0.6,
                linewidths=0,
                depthshade=False,
                rasterized=True,
            )
        state = item["result"]["states"]
        xy = np.asarray(state["xy"])
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            np.full(len(xy), 0.04),
            color=theme.color("figure_text"),
            lw=1.1,
        )
        extent = item["mapping"]["metadata"]["extent_m"]
        ax.set(
            xlim=extent[:2],
            ylim=extent[2:],
            zlim=(lower, upper),
            xlabel="x (m)",
            ylabel="y (m)",
            zlabel="z (m)",
        )
        ax.set_box_aspect([extent[1] - extent[0], extent[3] - extent[2], upper - lower])
        ax.view_init(elev=30, azim=-115)
        ax.set_title(item["spec"]["label"], fontsize=11)
        ax.tick_params(labelsize=8, pad=1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.set_major_locator(MaxNLocator(nbins=3))
    if artist is not None:
        fig.colorbar(
            artist,
            ax=axes,
            label="Measured world height z (m)",
            fraction=0.014,
            pad=0.015,
            shrink=0.6,
        )
    fig.suptitle(
        f"{_representative_label(cases)} final measured 3D voxel maps", fontsize=12
    )
    path = output / "mapping_3d_comparison.png"
    fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return {"png": str(path.resolve())}


def render_mapping(data_dir, output_dir, *, interactive=True):
    """Render comparable measured maps from stored environment runs.

    Produces static PNG comparisons, a report figure, and optional interactive HTML.
    Observation grids contain endpoint evidence only, without inferred free or
    traversable cells. HTML provides synchronized 3D playback at every saved state.
    """
    import matplotlib.pyplot as plt
    from .mapping_report import render_mapping_report

    directory, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    comparison, cases = _load_cases(directory)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(theme.MATPLOTLIB):
        outputs = {
            "comparison": _comparison(cases, output),
            "map_3d": _comparison_3d(cases, output),
            "report": render_mapping_report(directory, output),
        }
        if interactive:
            outputs["viewer"] = _viewer(comparison, cases, output)
    return outputs
