"""Offline browser rendering of an already computed PointCloud."""
from __future__ import annotations
import json
import math
from typing import Sequence
import numpy as np
from .. import api as ls
from . import theme
from .scene_geometry import mesh_for


def plot_interactive(
    pc: ls.PointCloud,
    scene: ls.Scene,
    cfg: ls.LidarConfig,
    sensor_pos: Sequence[float],
    path: str,
    title: str = "",
) -> str:
    """Write an offline viewer with separate world-frame and angular views.

    Native controls filter measured returns and reference meshes without modifying or
    resampling the simulation data. Ground is optional and visually subdued.
    """
    import html as html_lib
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    from plotly.utils import PlotlyJSONEncoder

    names = list(pc.label_names)
    n_labels = len(names)
    friendly = {
        "ground": "Ground",
        "wall_concrete": "Concrete wall",
        "car": "Car",
        "wall_white": "White wall",
        "sphere": "Sphere",
        "pole": "Pole",
        "retro_box": "Reflective box",
    }
    display_names = [friendly.get(n, n.replace("_", " ").capitalize()) for n in names]
    is_ground = [n.lower() == "ground" for n in names]
    palette = theme.PALETTE
    colours, next_colour = ([], 0)
    for ground in is_ground:
        colours.append(
            theme.color("#B8BEC6") if ground else palette[next_colour % len(palette)]
        )
        if not ground:
            next_colour += 1
    label_scale = (
        [
            [boundary, colours[i]]
            for i in range(n_labels)
            for boundary in (i / n_labels, (i + 1) / n_labels)
        ]
        if n_labels
        else [[0, theme.color("#B8BEC6")], [1, theme.color("#B8BEC6")]]
    )

    def finite_max(values, fallback):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        return max(float(values.max()), fallback) if values.size else fallback

    max_range = finite_max(pc.range_m, 1.0) if pc.n else max(cfg.max_range_m, 1.0)
    snr_low = math.log10(max(cfg.snr_threshold, 1e-09))
    snr_high = max(math.log10(finite_max(pc.snr, 1.0)), snr_low + 0.1)
    modes = {
        "object": dict(
            label="Object identity (ground truth)",
            scale=label_scale,
            lo=-0.5,
            hi=max(n_labels - 0.5, 0.5),
            title="Object ID<br>(ground truth)",
            explanation="Colours identify the known scene objects. These labels come from the simulator, not an object detector.",
        ),
        "range": dict(
            label="Measured range (m)",
            field=0,
            scale=theme.colormap("range"),
            lo=0,
            hi=max_range,
            title="Range<br>(m)",
            explanation="Distance reported for each return, including simulated range noise.",
        ),
        "reflectivity": dict(
            label="Apparent reflectivity",
            field=1,
            scale=theme.colormap("reflectivity"),
            lo=0,
            hi=finite_max(pc.reflectivity, 1.0),
            title="Apparent<br>reflectivity",
            explanation="Range-compensated return strength. It still depends on incidence angle, atmosphere, specular response and noise; values may exceed 1.",
        ),
        "intensity": dict(
            label="Return intensity (0–1)",
            field=2,
            scale=theme.colormap("intensity"),
            lo=0,
            hi=1,
            title="Intensity<br>(0–1)",
            explanation="Received power divided by the configured saturation power, clipped to the interval 0–1.",
        ),
        "snr": dict(
            label="Signal-to-noise ratio (log scale)",
            field=3,
            scale=theme.colormap("snr"),
            lo=snr_low,
            hi=snr_high,
            title="log₁₀ SNR",
            explanation="Colour shows log₁₀ of measured return power divided by receiver noise RMS. Hover shows the untransformed ratio.",
        ),
        "incidence": dict(
            label="Incidence angle (°)",
            field=4,
            scale=theme.colormap("incidence"),
            lo=0,
            hi=90,
            title="Incidence<br>(°)",
            explanation="0° is normal incidence; 90° is grazing incidence. For a merged echo, this is the strongest contributing sub-ray's angle.",
        ),
    }
    cloud = go.Figure()
    return_meta = []
    for label_id in range(n_labels):
        selected = pc.label_id == label_id
        if not np.any(selected):
            continue
        hover = np.column_stack(
            [
                pc.range_m[selected],
                pc.reflectivity[selected],
                pc.intensity[selected],
                pc.snr[selected],
                pc.incidence_deg[selected],
                pc.ring[selected],
                pc.azimuth_deg[selected],
                pc.elevation_deg[selected],
                pc.label_id[selected],
            ]
        ).tolist()
        return_meta.append(
            dict(index=len(cloud.data), label=label_id, ground=is_ground[label_id])
        )
        cloud.add_trace(
            go.Scatter3d(
                x=pc.xyz_world[selected, 0].tolist(),
                y=pc.xyz_world[selected, 1].tolist(),
                z=pc.xyz_world[selected, 2].tolist(),
                mode="markers",
                name=display_names[label_id],
                customdata=hover,
                visible=not is_ground[label_id],
                showlegend=False,
                opacity=0.32 if is_ground[label_id] else 0.95,
                marker=dict(
                    size=1.4 if is_ground[label_id] else 2.3,
                    color=[label_id] * len(hover),
                    coloraxis="coloraxis",
                ),
                hovertemplate="<b>%{fullData.name}</b> · measured return<br>World x: %{x:.3f} m<br>World y: %{y:.3f} m<br>World z: %{z:.3f} m<br>Range: %{customdata[0]:.3f} m<br>Apparent reflectivity: %{customdata[1]:.3f}<br>Intensity: %{customdata[2]:.3f} (0–1)<br>Signal/noise RMS: %{customdata[3]:.2f}<br>Incidence: %{customdata[4]:.2f}°<br>Channel: %{customdata[5]:.0f}<br>Azimuth: %{customdata[6]:.2f}° · Elevation: %{customdata[7]:.2f}°<br>Object ID: %{customdata[8]:.0f} (ground truth)<extra></extra>",
            )
        )
    sp = np.asarray(sensor_pos, dtype=float)
    cloud.add_trace(
        go.Scatter3d(
            x=[float(sp[0])],
            y=[float(sp[1])],
            z=[float(sp[2])],
            mode="markers",
            name="LiDAR at scan start",
            showlegend=False,
            marker=dict(size=6, color=theme.color("#202B38"), symbol="diamond"),
            hovertemplate="<b>LiDAR at scan start</b><br>World x: %{x:.3f} m<br>World y: %{y:.3f} m<br>Height coordinate z: %{z:.3f} m<extra></extra>",
        )
    )
    mesh_meta = []
    for label_id, obj in enumerate(scene.objects):
        mesh = mesh_for(obj)
        if mesh is None:
            continue
        vertices, faces = mesh
        known = label_id < n_labels
        mesh_name = (
            display_names[label_id]
            if known
            else friendly.get(obj.label, obj.label.replace("_", " "))
        )
        mesh_meta.append(
            dict(
                index=len(cloud.data),
                label=label_id,
                ground=is_ground[label_id] if known else obj.label.lower() == "ground",
            )
        )
        cloud.add_trace(
            go.Mesh3d(
                x=vertices[:, 0].tolist(),
                y=vertices[:, 1].tolist(),
                z=vertices[:, 2].tolist(),
                i=faces[:, 0].tolist(),
                j=faces[:, 1].tolist(),
                k=faces[:, 2].tolist(),
                color=colours[label_id] if known else theme.color("#B8BEC6"),
                opacity=0.12,
                flatshading=True,
                name=f"{mesh_name} — reference geometry",
                visible=False,
                showscale=False,
                showlegend=False,
                hovertemplate="<b>%{fullData.name}</b><br>Known scene surface; not a measured return.<br>World x: %{x:.2f} m · y: %{y:.2f} m · z: %{z:.2f} m<extra></extra>",
            )
        )
    axis_style = dict(
        backgroundcolor=theme.color("#FFFFFF"),
        gridcolor=theme.color("#E2E7EC"),
        zerolinecolor=theme.color("#C5CED8"),
        showbackground=True,
        color=theme.color("#475569"),
        ticks="outside",
    )
    cloud.update_layout(
        autosize=True,
        paper_bgcolor=theme.color("#FFFFFF"),
        font=dict(
            family=theme.THEME["web"]["font_family"],
            color=theme.color("#243447"),
            size=12,
        ),
        margin=dict(l=65, r=24, t=18, b=24),
        showlegend=False,
        uirevision="lidar-view",
        scene=dict(
            aspectmode="data",
            dragmode="orbit",
            xaxis=dict(title="World x (m)", **axis_style),
            yaxis=dict(title="World y (m)", **axis_style),
            zaxis=dict(title="World z (m)", **axis_style),
            camera=dict(eye=dict(x=-1.7, y=-1.5, z=1.0)),
        ),
        coloraxis=dict(
            colorscale=label_scale,
            cmin=-0.5,
            cmax=max(n_labels - 0.5, 0.5),
            showscale=False,
            colorbar=dict(
                title=dict(text="Object ID<br>(ground truth)", font=dict(size=11)),
                tickmode="array",
                tickvals=list(range(n_labels)),
                ticktext=[f"{name} [{i}]" for i, name in enumerate(display_names)],
                tickfont=dict(size=10),
                x=1.02,
                y=0.5,
                len=0.78,
                thickness=12,
                outlinewidth=0,
            ),
        ),
    )
    if not pc.n:
        cloud.add_annotation(
            text="No detected returns in this scan.<br>Reference geometry can still be shown.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, color=theme.color("#64748B")),
        )
    azimuths, elevations = (cfg.azimuths_deg().tolist(), cfg.elevations_deg().tolist())
    range_values = np.asarray(pc.range_image)
    range_z = [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in range_values
    ]
    range_labels = np.full(range_values.shape, -1, dtype=int)
    if pc.n and range_values.ndim == 2 and range_values.shape[1]:
        columns = np.rint(
            (pc.azimuth_deg - cfg.hfov_start_deg) / (360.0 / cfg.measurements_per_cycle)
        ).astype(int)
        rows = pc.ring.astype(int)
        valid = (
            (rows >= 0)
            & (rows < range_values.shape[0])
            & (columns >= 0)
            & (columns < range_values.shape[1])
        )
        range_labels[rows[valid], columns[valid]] = pc.label_id[valid]
    angular = go.Figure(
        go.Heatmap(
            z=range_z,
            x=azimuths,
            y=elevations,
            customdata=range_labels.tolist(),
            colorscale=theme.colormap("range"),
            zmin=0,
            zmax=max_range,
            hoverongaps=False,
            colorbar=dict(
                title=dict(text="Range<br>(m)", font=dict(size=11)),
                thickness=12,
                outlinewidth=0,
                tickfont=dict(size=10),
            ),
            hovertemplate="Azimuth: %{x:.2f}°<br>Elevation: %{y:.2f}°<br>Measured range: %{z:.3f} m<br>Object ID: %{customdata:.0f} (ground truth)<extra></extra>",
        )
    )
    angular.update_layout(
        autosize=True,
        paper_bgcolor=theme.color("#FFFFFF"),
        plot_bgcolor=theme.color("#F1F4F7"),
        font=dict(
            family=theme.THEME["web"]["font_family"],
            color=theme.color("#243447"),
            size=12,
        ),
        margin=dict(l=68, r=82, t=12, b=58),
        uirevision="lidar-angular",
        xaxis=dict(
            title="Azimuth in sensor frame (°)",
            automargin=True,
            zeroline=False,
            gridcolor=theme.color("#E2E7EC"),
        ),
        yaxis=dict(
            title="Elevation in sensor frame (°)",
            automargin=True,
            zeroline=False,
            gridcolor=theme.color("#E2E7EC"),
        ),
    )
    if len(azimuths) == 1:
        angular.update_xaxes(
            tickvals=azimuths, range=[azimuths[0] - 0.5, azimuths[0] + 0.5]
        )
    if len(elevations) == 1:
        angular.update_yaxes(
            tickvals=elevations, range=[elevations[0] - 0.5, elevations[0] + 0.5]
        )

    def script_json(value):
        return (
            json.dumps(value, cls=PlotlyJSONEncoder, ensure_ascii=False)
            .replace("</", "<\\/")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )

    escape = html_lib.escape
    heading = title or "3D LiDAR — simulated measurements"
    colour_options = "".join(
        (
            f"""<option value="{key}">{escape(mode['label'])}</option>"""
            for key, mode in modes.items()
        )
    )
    object_options = '<option value="all">All non-ground objects</option>'
    for label_id, name in enumerate(display_names):
        if not is_ground[label_id]:
            object_options += (
                f'<option value="{label_id}">{escape(name)} · ID {label_id}</option>'
            )
    object_options += '<option value="none">Hide object returns</option>'
    legend_items = "".join(
        (
            f'<span class="legend-item" data-label="{i}" data-ground="{int(is_ground[i])}"><span class="legend-swatch" style="background:{colours[i]}"></span>{escape(name)} [{i}]</span>'
            for i, name in enumerate(display_names)
        )
    )
    no_ground = " disabled" if not any(is_ground) else ""
    page = theme.asset("templates/scan.html")
    page = page.replace("__STYLE__", theme.stylesheet("scan.css")).replace(
        "__LOGIC__", theme.asset("static/scan.js")
    )
    replacements = {
        "__TITLE__": escape(heading),
        "__SX__": f"{sp[0]:.2f}",
        "__SY__": f"{sp[1]:.2f}",
        "__SZ__": f"{sp[2]:.2f}",
        "__COLOUR_OPTIONS__": colour_options,
        "__OBJECT_OPTIONS__": object_options,
        "__NO_GROUND__": no_ground,
        "__LEGEND_ITEMS__": legend_items,
        "__PLOTLY__": get_plotlyjs(),
        "__CLOUD_SPEC__": script_json(cloud.to_plotly_json()),
        "__ANGULAR_SPEC__": script_json(angular.to_plotly_json()),
        "__MODES__": script_json(modes),
        "__RETURN_META__": script_json(return_meta),
        "__MESH_META__": script_json(mesh_meta),
        "__RANGE_LABELS__": script_json(range_labels.tolist()),
        "__TICKVALS__": script_json(list(range(n_labels))),
        "__TICKTEXT__": script_json(
            [f"{name} [{i}]" for i, name in enumerate(display_names)]
        ),
    }
    import re

    page = re.sub(
        "__[A-Z_]+__",
        lambda match: replacements.get(match.group(0), match.group(0)),
        page,
    )
    page = page.replace("�", "\\ufffd")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return path
