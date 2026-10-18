"""Static, animated, and standalone browser views of saved trajectory runs.

Rendering never invokes a scan, route search, or controller. The trajectory
package's loader reconstructs only stored arrays and input snapshots.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import theme
from .scene_geometry import footprint_outline


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _palette(index: int) -> str:
    return theme.PALETTE[index % len(theme.PALETTE)]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _select_run(directory: Path, run_name=None):
    from ..trajectory import load_result

    rows = _rows(directory / "runs.csv")
    if rows:
        candidates = [
            row for row in rows if run_name is None or row["case_id"] == run_name
        ]
        if not candidates:
            raise ValueError(
                f"Unknown trajectory run {run_name!r}; choose a case_id from runs.csv"
            )
        baseline = [
            row
            for row in candidates
            if row["case_id"] == "baseline" or row.get("factor") == "baseline"
        ]
        selected = min(
            baseline or candidates, key=lambda row: (row["case_id"], int(row["seed"]))
        )
        run_dir = directory / "runs" / selected["case_id"] / f"seed_{selected['seed']}"
    else:
        run_dir = directory / run_name if run_name else directory
    return load_result(run_dir), rows


def _geometry(result):
    from .. import api as ls

    config = result["config"]
    unexpected = {obj["label"] for obj in config["world"].get("unexpected_objects", [])}
    scene = ls.build_scene({"materials": {}, "objects": result["world_objects"]})
    objects = []
    for obj in scene.objects:
        polygon = footprint_outline(obj)
        if polygon is not None:
            objects.append(
                {
                    "label": obj.label,
                    "unknown": obj.label in unexpected,
                    "xy": np.asarray(polygon, float),
                }
            )
    return objects


def _robot(result):
    return result["config"]["robot"]


def _bounds(result, objects, *, compact=False):
    pieces = [np.asarray(result["reference"]["xy"]), np.asarray(result["states"]["xy"])]
    pieces.extend(obj["xy"] for obj in objects)
    points = np.concatenate(pieces)
    low, high = points.min(0), points.max(0)
    padding = (
        np.maximum(0.02 * (high - low), [0.35, 0.15])
        if compact
        else np.maximum(0.05 * (high - low), 0.7)
    )
    return low - padding, high + padding


def _draw_map(ax, result, objects, *, compact=False, show_execution=True):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    low, high = _bounds(result, objects, compact=compact)
    for obj in objects:
        polygon = obj["xy"]
        unknown = obj["unknown"]
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            color=_palette(1) if unknown else theme.color("ground"),
            alpha=0.18 if unknown else 0.55,
            zorder=2,
        )
        ax.plot(
            polygon[:, 0],
            polygon[:, 1],
            color=_palette(1) if unknown else theme.color("figure_muted"),
            lw=1.25 if unknown else 0.8,
            zorder=3,
        )
    reference = np.asarray(result["reference"]["xy"])
    actual = np.asarray(result["states"]["xy"])
    ax.plot(reference[:, 0], reference[:, 1], "--", color=_palette(0), lw=1.2, zorder=4)
    if show_execution:
        ax.plot(
            actual[:, 0],
            actual[:, 1],
            "-",
            color=theme.color("figure_text"),
            lw=1.5,
            zorder=5,
        )
    start, goal = np.asarray(reference[0]), np.asarray(reference[-1])
    ax.plot(*start, "o", color=theme.color("figure_sensor"), ms=4.5, zorder=7)
    ax.plot(*goal, "*", color=theme.color("figure_sensor"), ms=8, zorder=7)
    ax.set(
        xlim=(low[0], high[0]),
        ylim=(low[1], high[1]),
        xlabel="World x (m)",
        ylabel="World y (m)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=theme.color("figure_grid"), lw=0.5, zorder=0)
    ax.tick_params(labelsize=8 if compact else 9)
    ax.xaxis.label.set_size(8.5 if compact else 9.5)
    ax.yaxis.label.set_size(8.5 if compact else 9.5)
    handles = [
        Line2D(
            [],
            [],
            linestyle="--",
            color=_palette(0),
            lw=1.2,
            label="Known-map reference",
        ),
        Line2D([], [], color=theme.color("figure_text"), lw=1.5, label="Executed path"),
        Patch(
            facecolor=theme.color("ground"),
            edgecolor=theme.color("figure_muted"),
            alpha=0.55,
            label="Prior-map obstacle (truth)",
        ),
        Patch(
            facecolor=_palette(1),
            edgecolor=_palette(1),
            alpha=0.3,
            label="Unmapped obstacle (truth)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=theme.color("figure_sensor"),
            ms=4,
            label="Start",
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            color=theme.color("figure_sensor"),
            ms=7,
            label="Goal",
        ),
    ]
    if not compact:
        for obj in objects:
            if obj["unknown"]:
                center = obj["xy"][:-1].mean(0)
                ax.annotate(
                    "Unmapped obstacle",
                    center,
                    xytext=(8, 16),
                    textcoords="offset points",
                    fontsize=8,
                    color=_palette(1),
                    arrowprops={"arrowstyle": "-", "color": _palette(1), "lw": 0.8},
                )
    return handles


def _save(fig, directory: Path, name: str):
    import matplotlib.pyplot as plt

    paths = {}
    for suffix in ("png", "pdf"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.07)
        paths[suffix] = str(path.resolve())
    plt.close(fig)
    return paths


def _case_label(case, factor):
    known = {
        "baseline": "Baseline",
        "fog": "Dense fog",
        "sparse_azimuth": "Sparse azimuth sampling",
        "high_mount": "Higher sensor mount",
        "faster": "Higher speed",
        "small_margin": "Smaller safety margin",
        "lidar_disabled": "LiDAR feedback disabled",
        "dense_fog": "Dense fog",
        "sparse_sampling": "Sparse azimuth sampling",
        "faster_speed": "Higher speed",
        "smaller_margin": "Smaller safety margin",
    }
    return known.get(case, known.get(factor, case.replace("_", " ").capitalize()))


def _overview(result, rows, objects, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.5, 3.9),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.2, 1]},
    )
    handles = _draw_map(axes[0], result, objects)
    axes[0].set_title(
        "(a) Reference and closed-loop execution", loc="left", fontsize=11
    )
    if rows:
        cases = list(dict.fromkeys(row["case_id"] for row in rows))
        for i, case in enumerate(cases):
            selected = [row for row in rows if row["case_id"] == case]
            gaps = np.asarray(
                [float(row["swept_min_surface_gap_m"]) for row in selected]
            )
            completed = sum(
                str(row["goal_completed"]).lower() in ("1", "true") for row in selected
            )
            center = float(gaps.mean())
            axes[1].barh(
                i,
                center,
                height=0.56,
                color=_palette(0) if completed == len(selected) else _palette(1),
                alpha=0.75,
            )
            axes[1].errorbar(
                center,
                i,
                xerr=[[max(0, center - gaps.min())], [max(0, gaps.max() - center)]],
                fmt="none",
                ecolor=theme.color("figure_text"),
                elinewidth=0.9,
                capsize=2,
            )
            mean_time = np.mean([float(row["elapsed_s"]) for row in selected])
            axes[1].text(
                1.02,
                i,
                f"{mean_time:.1f} s; {completed}/{len(selected)} goals",
                transform=axes[1].get_yaxis_transform(),
                va="center",
                fontsize=8,
            )
        axes[1].set_yticks(
            np.arange(len(cases)),
            [
                _case_label(
                    case, next(row["factor"] for row in rows if row["case_id"] == case)
                )
                for case in cases
            ],
            fontsize=8.5,
        )
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Minimum surface gap (m)", fontsize=9.5)
        axes[1].axvline(0, color=theme.color("collision"), linestyle="--", lw=0.8)
        axes[1].grid(True, axis="x", color=theme.color("figure_grid"), lw=0.5)
        axes[1].set_axisbelow(True)
        axes[1].set_title(
            "(b) One-factor comparisons\nMean and min-max across paired seeds",
            loc="left",
            fontsize=10.5,
            linespacing=1.6,
        )
        axes[1].text(
            1.02,
            1.01,
            "End time; goals",
            transform=axes[1].transAxes,
            fontsize=8,
            color=theme.color("figure_muted"),
        )
    else:
        states, controls = result["states"], result["controls"]
        axes[1].plot(
            states["t"],
            states["surface_gap"],
            color=theme.color("figure_text"),
            label="True surface gap",
        )
        axes[1].plot(
            controls["t"],
            controls["perceived_clearance"],
            ".",
            color=_palette(1),
            ms=2,
            label="Nearest-return gap proxy",
        )
        axes[1].set(xlabel="Time (s)", ylabel="Distance (m)")
        axes[1].set_title("(b) Executed clearance", loc="left", fontsize=11)
        axes[1].legend(frameon=False, fontsize=8)
    fig.legend(
        handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8
    )
    return _save(fig, output, "trajectory_overview")


def _report(result, objects, output):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 2.6), layout="constrained")
    handles = _draw_map(ax, result, objects, compact=True)
    handles = handles[:4]
    for handle, label in zip(
        handles, ("Reference", "Executed", "Known obstacle", "Unmapped obstacle")
    ):
        handle.set_label(label)
    reference = np.asarray(result["reference"]["xy"])
    for point, label in ((reference[0], "Start"), (reference[-1], "Goal")):
        ax.annotate(
            label,
            point,
            xytext=(0, -11),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.5,
        )
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        frameon=True,
        facecolor=theme.color("white"),
        edgecolor=theme.color("figure_grid"),
        framealpha=0.95,
        fontsize=7.7,
        handlelength=1.6,
        columnspacing=1.0,
    )
    return _save(fig, output, "trajectory_report")


def _display_points(points, limit=1200):
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) > limit:
        points = points[np.linspace(0, len(points) - 1, limit, dtype=int)]
    return points


def _motion(result, objects, output, every, fps):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, FancyArrowPatch

    states, controls = result["states"], result["controls"]
    n = len(states["t"])
    stride = max(1, int(every), int(np.ceil(n / 90)))
    indices = sorted(set(list(range(0, n, stride)) + [n - 1]))
    fig, ax = plt.subplots(figsize=(9, 3.2), layout="constrained")
    handles = _draw_map(ax, result, objects, show_execution=False)
    (line,) = ax.plot([], [], "-", color=theme.color("figure_text"), lw=1.8, zorder=6)
    observations = ax.scatter(
        [], [], s=5, color=theme.color("obstacle"), alpha=0.8, linewidths=0, zorder=5
    )
    body = Circle(
        np.asarray(states["xy"])[0],
        _robot(result)["radius_m"],
        fill=False,
        edgecolor=theme.color("figure_sensor"),
        linewidth=1.3,
        zorder=8,
    )
    ax.add_patch(body)
    heading = FancyArrowPatch(
        (0, 0),
        (1, 0),
        arrowstyle="-|>",
        mutation_scale=10,
        color=theme.color("figure_sensor"),
        linewidth=1.2,
        zorder=9,
    )
    ax.add_patch(heading)
    (target,) = ax.plot([], [], "x", ms=6, mew=1.2, color=_palette(2), zorder=9)
    from matplotlib.lines import Line2D

    handles += [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=".",
            color=theme.color("obstacle"),
            markersize=5,
            label="LiDAR obstacle returns",
        )
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8
    )

    def draw(frame):
        k = indices[frame]
        xy = np.asarray(states["xy"])
        x, y = xy[k]
        th = float(states["theta"][k])
        line.set_data(xy[: k + 1, 0], xy[: k + 1, 1])
        body.center = (x, y)
        heading.set_positions((x, y), (x + np.cos(th), y + np.sin(th)))
        if k < len(result["observations"]):
            points = _display_points(result["observations"][k], 1000)
            observations.set_offsets(points)
            lookahead = np.asarray(controls["target_xy"])[k]
            target.set_data([lookahead[0]], [lookahead[1]])
            v, omega = float(controls["v"][k]), float(controls["omega"][k])
            info = f"command v = {v:.2f} m/s; turn rate = {np.degrees(omega):.1f} deg/s"
        else:
            observations.set_offsets(np.empty((0, 2)))
            target.set_data([], [])
            info = "final state; no additional scan"
        ax.set_title(
            f"t = {float(states['t'][k]):.1f} s; {info}; true gap = {float(states['surface_gap'][k]):.2f} m",
            fontsize=9.5,
            loc="left",
        )
        return line, body, heading, observations, target

    anim = FuncAnimation(
        fig, draw, frames=len(indices), interval=1000 / max(1, fps), blit=False
    )
    path = output / "trajectory_motion.gif"
    anim.save(path, writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)
    return {"gif": str(path.resolve())}


def _viewer(result, objects, output):
    import re
    from plotly.offline import get_plotlyjs

    states, controls = result["states"], result["controls"]
    low, high = _bounds(result, objects)
    data = {
        "name": result.get("name", "Trajectory run"),
        "status": result["status"],
        "seed": result.get("seed"),
        "states": states,
        "controls": controls,
        "observations": [
            np.round(_display_points(points), 4) for points in result["observations"]
        ],
        "reference": result["reference"]["xy"],
        "objects": objects,
        "extent": [low[0], high[0], low[1], high[1]],
        "radius": _robot(result)["radius_m"],
        "colors": {
            "reference": _palette(0),
            "actual": theme.color("figure_text"),
            "unknown": _palette(1),
            "known": theme.color("ground"),
            "observations": theme.color("obstacle"),
            "target": _palette(2),
            "robot": theme.color("figure_sensor"),
            "speed": _palette(0),
            "turn": _palette(1),
            "grid": theme.color("figure_grid"),
            "known_line": theme.color("figure_muted"),
            "paper": theme.color("white"),
            "plot": theme.color("background"),
        },
        "font": theme.THEME["web"]["font_family"],
    }
    payload = json.dumps(_jsonable(data), ensure_ascii=False, allow_nan=False).replace(
        "</", "<\\/"
    )
    replacements = {
        "__STYLE__": theme.stylesheet("trajectory.css"),
        "__PLOTLY__": get_plotlyjs(),
        "__DATA__": payload,
        "__LOGIC__": theme.asset("static/trajectory.js"),
    }
    template = theme.asset("templates/trajectory.html")
    page = re.sub(
        r"__[A-Z_]+__",
        lambda match: replacements.get(match.group(0), match.group(0)),
        template,
    )
    path = output / "trajectory_interactive.html"
    path.write_text(page, encoding="utf-8")
    return {"html": str(path.resolve())}


def render_trajectory(
    data_dir, output_dir, *, run_name=None, gif=True, interactive=True, every=2, fps=8
):
    """Render stored trajectory data, optionally selecting one factor-study case.

    ``data_dir`` can be a saved individual run or a study directory containing
    ``runs.csv`` and ``runs/<case_id>/seed_<seed>/``. With a study, the baseline
    run with the lowest seed is selected by default. The GIF displays at most
    approximately 90 stored states; the HTML retains every state and command.
    Return keys: ``overview``, ``report``, and optional ``motion``/``viewer``.
    """
    import matplotlib.pyplot as plt

    data = Path(data_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    result, rows = _select_run(data, run_name)
    objects = _geometry(result)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(theme.MATPLOTLIB):
        outputs = {
            "overview": _overview(result, rows, objects, output),
            "report": _report(result, objects, output),
        }
        if gif:
            outputs["motion"] = _motion(result, objects, output, every, fps)
        if interactive:
            outputs["viewer"] = _viewer(result, objects, output)
    return outputs
