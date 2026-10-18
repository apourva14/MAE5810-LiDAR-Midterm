"""Scientific comparison figures from saved controlled-experiment data.

The renderer reads CSV/JSON/NPZ artifacts only. It never calls a sensor scan,
scene intersection, random sampler, or experiment runner. Public entry point:
``render_evaluation(data_dir, output_dir)``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import theme


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _sort(rows: list[dict[str, str]], key: str) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: float(row[key]))


def _manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def _colors() -> list[str]:
    palette = list(theme.PALETTE)
    return [palette[i % len(palette)] for i in range(max(7, len(palette)))]


def _axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=10.5, pad=9)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.tick_params(labelsize=9)
    ax.grid(True, color=theme.color("figure_grid"), linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, output: Path, name: str) -> dict[str, str]:
    import matplotlib.pyplot as plt

    paths = {}
    for suffix in ("png", "pdf"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.09)
        paths[suffix] = str(path.resolve())
    plt.close(fig)
    return paths


def _weather_order(rows):
    names = sorted({row["weather"] for row in rows})

    def rank(name):
        name = name.lower()
        return (
            0
            if "clear" in name
            else 1
            if "rain" in name
            else 2
            if "fog" in name
            else 3,
            name,
        )

    return sorted(names, key=rank)


def _weather_label(name):
    if "clear" in name.lower():
        return "Clear air"
    if "rain" in name.lower():
        return "Rain"
    if "fog" in name.lower():
        return "Fog"
    return name.replace("_", " ").capitalize()


def _response_laws(directory: Path, output: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = _rows(directory / "laws.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.95), layout="constrained")
    distance = [row for row in rows if row["case"] == "distance"]
    for i, rho in enumerate(sorted({float(row["rho"]) for row in distance})):
        selected = _sort(
            [row for row in distance if float(row["rho"]) == rho], "range_m"
        )
        x = _values(selected, "range_m")
        axes[0].plot(
            x,
            _values(selected, "predicted_power_w"),
            "-",
            color=_colors()[i],
            lw=1.2,
            label=rf"$\rho_d = {rho:.2f}$",
        )
        axes[0].plot(
            x,
            _values(selected, "measured_power_w"),
            "o",
            ms=3.4,
            markerfacecolor="white",
            color=_colors()[i],
            markeredgewidth=0.9,
        )
    _axes(axes[0], "(a) Range dependence", "Range r (m)", "Return power (W)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=7.7, loc="best", handlelength=1.4)

    selected = _sort(
        [row for row in rows if row["case"] == "incidence"], "incidence_deg"
    )
    x = _values(selected, "incidence_deg")
    predicted = _values(selected, "predicted_power_w")
    observed = _values(selected, "measured_power_w")
    axes[1].plot(
        x, predicted / predicted[0], "-", color=theme.color("figure_text"), lw=1.2
    )
    axes[1].plot(
        x,
        observed / predicted[0],
        "o",
        ms=4,
        markerfacecolor="white",
        color=_colors()[0],
    )
    _axes(
        axes[1],
        "(b) Diffuse incidence",
        "Incidence angle (deg)",
        "Relative power P/P(0)",
    )
    axes[1].set_ylim(0, 1.08)

    atmosphere = [row for row in rows if row["case"] == "extinction"]
    weather = _weather_order(atmosphere)
    clear = next(name for name in weather if "clear" in name.lower())
    baseline = {
        float(row["range_m"]): float(row["predicted_power_w"])
        for row in atmosphere
        if row["weather"] == clear
    }
    for i, name in enumerate(weather):
        selected = _sort(
            [row for row in atmosphere if row["weather"] == name], "range_m"
        )
        x = _values(selected, "range_m")
        denominator = np.asarray([baseline[value] for value in x])
        axes[2].plot(
            x,
            _values(selected, "predicted_power_w") / denominator,
            "-",
            color=_colors()[i],
            lw=1.2,
            label=_weather_label(name),
        )
        axes[2].plot(
            x,
            _values(selected, "measured_power_w") / denominator,
            "o",
            ms=3.3,
            markerfacecolor="white",
            color=_colors()[i],
            markeredgewidth=0.9,
        )
    _axes(axes[2], "(c) Atmospheric loss", "Range r (m)", "Power / clear-air power")
    axes[2].set_yscale("log")
    axes[2].legend(frameon=False, fontsize=7.7, loc="best", handlelength=1.4)
    handles = [
        Line2D(
            [], [], color=theme.color("figure_text"), lw=1.2, label="Analytic power"
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=theme.color("figure_text"),
            markersize=4,
            label="Simulated power",
        ),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=8
    )
    return _save(fig, output, "response_laws")


def _geometry_pose_fov(directory: Path, output: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    rows = _rows(directory / "geometry_pose.csv")
    fov = _rows(directory / "fov.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.75), layout="constrained")
    for ax, variable, xlabel, title in (
        (axes[0], "height", "Sensor height h (m)", "(a) Sensor position"),
        (axes[1], "pitch", "Mounting pitch (deg)", "(b) Sensor orientation"),
    ):
        selected = [row for row in rows if row["experiment"] == f"ground_{variable}"]
        xkey = "height_m" if variable == "height" else "pitch_deg"
        selected = _sort(selected, xkey)
        x = _values(selected, xkey)
        ax.plot(
            x,
            _values(selected, "predicted_range_m"),
            "-",
            lw=1.3,
            color=theme.color("figure_text"),
            label="Analytic range",
        )
        ax.plot(
            x,
            _values(selected, "measured_range_m"),
            "o",
            ms=4,
            markerfacecolor="white",
            color=_colors()[0],
            markeredgewidth=1,
            label="Simulated range",
        )
        _axes(ax, title, xlabel, "Range to ground (m)")
        ax.legend(frameon=False, fontsize=7.8, loc="best")
    ax = axes[2]
    for i, (mode, label) in enumerate(
        (("partial", "Partial window"), ("full360", "360-deg scan"))
    ):
        selected = _sort(
            [
                row
                for row in fov
                if row["motion_axis"] == "yaw"
                and row["horizontal_mode"] == mode
                and float(row["angle_deg"]) <= 60
            ],
            "angle_deg",
        )
        if not selected:
            continue
        x = _values(selected, "angle_deg")
        ax.plot(
            x,
            _values(selected, "predicted_count"),
            "-" if i == 0 else "--",
            color=_colors()[i],
            lw=1.2,
            label=label,
        )
        ax.plot(
            x,
            _values(selected, "measured_count"),
            linestyle="none",
            marker="o" if i == 0 else "s",
            color=_colors()[i],
            markerfacecolor="white",
            ms=3.5,
            markeredgewidth=0.9,
        )
    _axes(
        ax,
        "(c) Horizontal field of view",
        "Sensor yaw (deg)",
        "Target returns per scan",
    )
    ax.set_ylim(bottom=-0.03 * max(1, ax.get_ylim()[1]))
    ax.legend(frameon=False, fontsize=7.8, loc="best")
    return _save(fig, output, "geometry_pose_fov")


def _box_outline(corners: np.ndarray) -> np.ndarray | None:
    if corners.shape != (8, 3):
        return None
    return np.vstack([corners[:4, :2], corners[0, :2]])


def _nominal_window(
    ax, config: dict[str, Any], pose: dict[str, Any], radius: float
) -> None:
    """Draw the nominal horizontal window at the sensor height for the LOS scene."""
    from matplotlib.patches import Wedge

    # The controlled LOS study uses zero roll/pitch. A horizontal slice is then
    # bounded by the configured azimuth limits, independent of visibility.
    if not np.isclose(pose.get("roll_deg", 0), 0) or not np.isclose(
        pose.get("pitch_deg", 0), 0
    ):
        return
    lower = float(config["hfov_start_deg"]) + float(pose.get("yaw_deg", 0))
    upper = float(config["hfov_end_deg"]) + float(pose.get("yaw_deg", 0))
    origin = np.asarray(pose["position"], float)[:2]
    extent = min(float(config["max_range_m"]), radius)
    if upper - lower >= 360:
        upper = lower + 360
    patch = Wedge(
        origin,
        extent,
        lower,
        upper,
        width=max(0, extent - float(config["min_range_m"])),
        facecolor=_colors()[0],
        edgecolor="none",
        alpha=0.06,
        zorder=0,
    )
    ax.add_patch(patch)
    for angle in (lower, upper):
        a = np.radians(angle)
        end = origin + extent * np.asarray([np.cos(a), np.sin(a)])
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            "--",
            color=_colors()[0],
            linewidth=0.85,
            alpha=0.85,
            zorder=1,
        )


def _los_comparison(directory: Path, output: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from ..data import load_npz

    metadata = _manifest(directory)
    selected_cases = [
        ("los_clear", "(a) Unobstructed target"),
        ("los_opaque", "(b) Opaque occluder"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), layout="constrained")
    counts = {row["case"]: row for row in _rows(directory / "los.csv")}
    for ax, (slug, title) in zip(axes, selected_cases):
        case = metadata["cases"][slug]
        config, pose = case["config"], case["pose"]
        cloud = load_npz(directory / case["artifacts"]["scan"])
        with np.load(
            directory / case["artifacts"]["reference"], allow_pickle=False
        ) as source:
            reference = {key: np.array(source[key], copy=True) for key in source.files}
        origin = np.asarray(reference["sensor_origin"], float)
        target = _box_outline(reference["target_box_corners"])
        occluder = _box_outline(reference["occluder_box_corners"])
        target_range = np.asarray(reference["potential_target_range_m"], float)
        directions = np.asarray(reference["directions_world"], float)
        good = np.isfinite(target_range) & (target_range > 0)
        target_hits = origin + target_range[good, None] * directions[good]
        blocked = np.asarray(reference["blocked_target"], bool)[good]
        extent = max(
            1,
            float(
                np.max(
                    np.linalg.norm(
                        reference["target_box_corners"][:, :2] - origin[:2], axis=1
                    )
                )
            ),
        )
        _nominal_window(ax, config, pose, extent * 1.28)
        if len(target_hits):
            ax.scatter(
                target_hits[:, 0],
                target_hits[:, 1],
                s=5,
                color=theme.color("ground"),
                marker=".",
                alpha=0.55,
                zorder=2,
                rasterized=True,
            )
        # Subsample rays for legibility; the numerical cloud is never resampled.
        indices = np.flatnonzero(good)
        for i in indices[
            np.linspace(0, max(len(indices) - 1, 0), min(11, len(indices)), dtype=int)
        ]:
            endpoint = origin + target_range[i] * directions[i]
            ax.plot(
                [origin[0], endpoint[0]],
                [origin[1], endpoint[1]],
                "-",
                color=theme.color("ground"),
                lw=0.45,
                alpha=0.35,
                zorder=1,
            )
        if np.any(blocked):
            ax.scatter(
                target_hits[blocked, 0],
                target_hits[blocked, 1],
                s=10,
                marker="x",
                linewidths=0.55,
                color=_colors()[1],
                zorder=4,
                rasterized=True,
            )
        if target is not None:
            ax.fill(
                target[:, 0],
                target[:, 1],
                facecolor=_colors()[2],
                edgecolor=_colors()[2],
                alpha=0.12,
                linewidth=1.1,
                zorder=2,
            )
        if occluder is not None:
            ax.fill(
                occluder[:, 0],
                occluder[:, 1],
                facecolor=theme.color("object_fill"),
                edgecolor=theme.color("object_fill"),
                alpha=0.7,
                zorder=5,
            )
        names = list(cloud.label_names)
        for label_id, label in enumerate(names):
            if "ground" in label.lower():
                continue
            points = cloud.xyz_world[cloud.label_id == label_id]
            if not len(points):
                continue
            color = (
                _colors()[0]
                if "target" in label.lower()
                else theme.color("object_fill")
            )
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=5,
                color=color,
                alpha=0.8,
                zorder=6,
                rasterized=True,
            )
        ax.plot(
            origin[0],
            origin[1],
            "^",
            color=theme.color("figure_sensor"),
            ms=6,
            zorder=8,
        )
        ax.annotate(
            "LiDAR",
            origin[:2],
            xytext=(-8, -14),
            textcoords="offset points",
            fontsize=8,
            color=theme.color("figure_text"),
        )
        _axes(ax, title, "World x (m)", "World y (m)")
        ax.set_aspect("equal", adjustable="box")
        half_width = max(1.4, extent * 0.31)
        ax.set_xlim(
            origin[0] - 0.7, float(np.max(reference["target_box_corners"][:, 0])) + 1.1
        )
        ax.set_ylim(origin[1] - half_width, origin[1] + half_width)
        n = int(float(counts[slug]["target_returns"]))
        ax.text(
            0.03,
            0.95,
            f"Target returns: {n:,}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            color=theme.color("figure_text"),
        )
    handles = [
        Line2D(
            [],
            [],
            linestyle="--",
            color=_colors()[0],
            lw=1,
            label="Nominal azimuth limits",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color=_colors()[0],
            markersize=4,
            label="Target returns",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="x",
            color=_colors()[1],
            markersize=4,
            label="Occluded target rays",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.8,
        columnspacing=1.2,
    )
    return _save(fig, output, "los_comparison")


def _detection_uncertainty(directory: Path, output: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    detection = _rows(directory / "detection.csv")
    uncertainty = [
        row
        for row in _rows(directory / "uncertainty.csv")
        if int(row["detections"]) >= 20
    ]
    weather = _weather_order(detection)
    rho_values = sorted({float(row["rho"]) for row in detection})
    symbols = ("o", "s", "^")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), layout="constrained")
    reference = _sort(detection, "expected_snr")
    axes[0].plot(
        _values(reference, "expected_snr"),
        _values(reference, "predicted_p"),
        "-",
        color=theme.color("figure_text"),
        lw=1.4,
        label="Analytic probability",
        zorder=2,
    )
    for i, name in enumerate(weather):
        for j, rho in enumerate(rho_values):
            rows = _sort(
                [
                    row
                    for row in detection
                    if row["weather"] == name and float(row["rho"]) == rho
                ],
                "expected_snr",
            )
            x, y = _values(rows, "expected_snr"), _values(rows, "observed_p")
            errors = np.vstack(
                [
                    np.maximum(0, y - _values(rows, "ci_low")),
                    np.maximum(0, _values(rows, "ci_high") - y),
                ]
            )
            axes[0].errorbar(
                x,
                y,
                yerr=errors,
                fmt=symbols[j],
                linestyle="none",
                ms=3.5,
                markerfacecolor="white",
                markeredgewidth=0.75,
                color=_colors()[i],
                ecolor=_colors()[i],
                elinewidth=0.65,
                capsize=1.5,
                capthick=0.65,
                alpha=0.9,
                zorder=3,
            )
            rows = _sort(
                [
                    row
                    for row in uncertainty
                    if row["weather"] == name and float(row["rho"]) == rho
                ],
                "expected_snr",
            )
            x, y = _values(rows, "expected_snr"), 100 * _values(rows, "rmse_m")
            valid = np.isfinite(x) & np.isfinite(y)
            axes[1].plot(
                x[valid],
                y[valid],
                linestyle="none",
                marker=symbols[j],
                ms=3.8,
                markerfacecolor="white",
                markeredgewidth=0.8,
                color=_colors()[i],
                zorder=3,
            )
    reference = _sort(
        [
            row
            for row in uncertainty
            if np.isfinite(float(row["predicted_sigma_rms_m"]))
        ],
        "expected_snr",
    )
    axes[1].plot(
        _values(reference, "expected_snr"),
        100 * _values(reference, "predicted_sigma_rms_m"),
        "-",
        color=theme.color("figure_text"),
        lw=1.25,
        label="Conditional RMS prediction",
        zorder=2,
    )
    _axes(
        axes[0],
        "(a) Threshold detection",
        r"Expected power / noise RMS, $P_G/\sigma_P$",
        "Detection probability",
    )
    _axes(
        axes[1],
        "(b) Range error given detection",
        r"Expected power / noise RMS, $P_G/\sigma_P$",
        "Range-error RMSE (cm)",
    )
    axes[0].set_ylim(-0.035, 1.035)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right", handlelength=1.8)
    axes[1].legend(frameon=False, fontsize=8, loc="best", handlelength=1.8)
    axes[0].text(
        0.02,
        0.97,
        "Monte Carlo\n95% Wilson CI",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        color=theme.color("figure_muted"),
    )
    handles = [
        Line2D(
            [],
            [],
            color=_colors()[i],
            marker="o",
            markersize=3.5,
            lw=0,
            label=_weather_label(name),
        )
        for i, name in enumerate(weather)
    ]
    handles += [
        Line2D(
            [],
            [],
            color=theme.color("figure_text"),
            linestyle="none",
            marker=symbols[j],
            markerfacecolor="white",
            markersize=4,
            label=rf"$\rho_d = {rho:.2f}$",
        )
        for j, rho in enumerate(rho_values)
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        handlelength=1.1,
        columnspacing=1.4,
    )
    return _save(fig, output, "detection_uncertainty")


def _fov_volume(directory: Path, output: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.ticker import MaxNLocator
    from .. import api as ls
    from .scene_geometry import mesh_for

    metadata = _manifest(directory)
    with np.load(
        directory / metadata["frustum"]["artifact"], allow_pickle=False
    ) as archive:
        data = {key: np.array(archive[key], copy=True) for key in archive.files}
    fig = plt.figure(figsize=(11.5, 4.1), layout="constrained")
    all_vertices = data["vertices"]
    lo, hi = all_vertices.min(0) - 0.6, all_vertices.max(0) + 0.6
    for i, name in enumerate(data["case_names"].tolist()):
        ax = fig.add_subplot(1, len(data["case_names"]), i + 1, projection="3d")
        vertices = all_vertices[
            data["vertex_offsets"][i] : data["vertex_offsets"][i + 1]
        ]
        faces = data["faces"][data["face_offsets"][i] : data["face_offsets"][i + 1]]
        case = metadata["cases"][name]
        pose, config = case["pose"], case["config"]
        origin = data["sensor_origins"][i]
        ax.add_collection3d(
            Poly3DCollection(
                vertices[faces],
                facecolor=_colors()[i],
                edgecolor="none",
                alpha=0.035,
                zorder=1,
            )
        )
        # Extract exact angular shell boundaries from the saved vertices.
        rotation = (
            ls.rot_z(np.radians(pose["yaw_deg"]))
            @ ls.rot_y(np.radians(pose["pitch_deg"]))
            @ ls.rot_x(np.radians(pose["roll_deg"]))
        )
        local = (vertices - origin) @ rotation
        radius = np.linalg.norm(local, axis=1)
        az = np.arctan2(local[:, 1], local[:, 0])
        el = np.arctan2(local[:, 2], np.linalg.norm(local[:, :2], axis=1))
        outer = np.isclose(radius, float(data["display_range_m"]), atol=1e-7)
        for coordinate, ordering in ((az, el), (el, az)):
            for boundary in (coordinate[outer].min(), coordinate[outer].max()):
                ids = np.flatnonzero(
                    outer & np.isclose(coordinate, boundary, atol=1e-8)
                )
                ids = ids[np.argsort(ordering[ids])]
                ax.plot(*vertices[ids].T, color=_colors()[i], lw=1.2, zorder=3)
        for a in (az[outer].min(), az[outer].max()):
            for e in (el[outer].min(), el[outer].max()):
                ids = np.flatnonzero(
                    np.isclose(az, a, atol=1e-8) & np.isclose(el, e, atol=1e-8)
                )
                ids = ids[np.argsort(radius[ids])]
                ax.plot(*vertices[ids].T, color=_colors()[i], lw=1.1, zorder=3)
        for obj in ls.build_scene(case["scene"]).objects:
            mesh = mesh_for(obj)
            if mesh is not None:
                vertices_target, triangles_target = mesh
                ax.add_collection3d(
                    Poly3DCollection(
                        vertices_target[triangles_target],
                        facecolor=theme.color("ground"),
                        edgecolor="none",
                        alpha=0.55,
                    )
                )
        ax.scatter(
            *origin, marker="^", color=theme.color("figure_sensor"), s=30, zorder=5
        )
        ax.set(
            xlim=(lo[0], hi[0]),
            ylim=(lo[1], hi[1]),
            zlim=(lo[2], hi[2]),
            xlabel="World x (m)",
            ylabel="World y (m)",
            zlabel="World z (m)",
        )
        ax.set_title(
            f"({chr(97+i)}) Yaw {pose['yaw_deg']:g} deg; pitch {pose['pitch_deg']:g} deg",
            fontsize=11,
        )
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=24, azim=-118)
        ax.tick_params(labelsize=9, pad=1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(nbins=4))
            axis.pane.fill = False
    gate = float(np.max(data["configured_max_range_m"]))
    displayed = float(data["display_range_m"])
    first_config = metadata["cases"][str(data["case_names"][0])]["config"]
    bounds = (
        f"Nominal 3D FOV: azimuth [{first_config['hfov_start_deg']:g}, {first_config['hfov_end_deg']:g}] deg; "
        f"elevation [{first_config['vfov_lower_deg']:g}, {first_config['vfov_upper_deg']:g}] deg in sensor frame"
    )
    fig.suptitle(
        bounds
        + f"\nDisplay clipped at {displayed:g} m; configured range gate {gate:g} m",
        fontsize=11,
    )
    path = output / "fov_volume.png"
    fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return {"png": str(path.resolve())}


def render_evaluation(data_dir, output_dir) -> dict[str, dict[str, str]]:
    """Write four PNG/PDF comparison figures plus a 3D FOV PNG from saved data.

    ``data_dir`` must contain the ``geometry`` and ``response`` study folders.
    Returns ``{figure_name: {"png": absolute_path, "pdf": absolute_path}}``.
    """
    import matplotlib.pyplot as plt

    data = Path(data_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    required = (
        data / "geometry" / "manifest.json",
        data / "response" / "config_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "saved evaluation data is incomplete: " + ", ".join(missing)
        )
    output.mkdir(parents=True, exist_ok=True)
    # The project-wide theme remains the only style source; report figures need
    # modestly smaller labels so three scientific panels fit a single column.
    with plt.rc_context(theme.MATPLOTLIB):
        outputs = {
            "geometry_pose_fov": _geometry_pose_fov(data / "geometry", output),
            "los_comparison": _los_comparison(data / "geometry", output),
            "response_laws": _response_laws(data / "response", output),
            "detection_uncertainty": _detection_uncertainty(data / "response", output),
            "fov_volume": _fov_volume(data / "geometry", output),
        }
    return outputs
