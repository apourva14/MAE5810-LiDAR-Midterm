# LiDAR midterm project

Physics-based 3D LiDAR simulation, fixed-scene measurement evaluations, and the
course report. Numerical simulation and presentation code are separate.

**Team Members for this project:**<br>
Arunkumar Venkatesan - av652@cornell.edu<br>
Xuyi Lian - xl2339@cornell.edu<br>
Apurva Singh - as4642@cornell.edu

## Read the report

[Open the live visual report](https://xuyilian.github.io/MAE5810-LiDAR-Midterm/),
or open `index.html` directly. The page works offline and presents saved
measurements, equations, 3D FOV/LOS geometry, trajectories, and growing maps.
The editable report and PDF are in `outputs/reports/`.

For a local browser address, run with Python 3.10 or newer:

```sh
python3 start.py
```

On macOS, `Start Project.command` launches the same page. Keep its terminal
window open. Use `--port 0` if the default port is occupied, or `--no-browser`
to open the printed address yourself. The server listens at `127.0.0.1`.

## Install generation dependencies

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[visualization,reports]"
```

Video export requires FFmpeg on your executable search path. On macOS:

```sh
brew install ffmpeg
```

PDF export requires LibreOffice and Times New Roman. A managed bundled renderer
is used when available. Otherwise, set the executable path before starting:

```sh
export LIDAR_SOFFICE="/Applications/LibreOffice.app/Contents/MacOS/soffice"
```

Word generation works separately if PDF export is unavailable. PDF export
checks the course page limit before replacing the published file.

## Generate deliverables

Start the local server and open `/builder` at its printed address. Choose
**Generate all deliverables**, or select an individual stage. Leave
**Recompute numerical studies** unchecked to reuse verified measurements;
missing inputs are generated automatically. A fresh numerical build archives
previous owned data in ignored `build/backups/`.

The GitHub checkout includes source, reports, media, summaries, and raw threshold
trials. Large scan and navigation histories regenerate from the supplied
configurations and fixed seeds. The offline HTML and saved videos play without
these raw arrays. Build records and logs remain in ignored `build/`.

The report uses nine target settings at matched viewpoints and Normal, Rain,
and Fog navigation in one fixed obstacle setup. The two evaluations share a
baseline; weather comparisons hold the scene, robot, sensor, controller, and
planned route fixed. Mapping uses the known simulated pose.

To reproduce numerical evidence directly:

```sh
python -m lidar_project evaluate
python -m lidar_project evaluate-navigation-dependencies --seeds 1 2 3 --cases baseline reflectance_dark reflectance_bright target_yaw_45 target_yaw_75 target_short target_tall target_near target_far rain fog
python -m lidar_project evaluate-navigation-threshold --trials 1000
```

Render and export the saved report evidence:

```sh
python tools/render_report_figures.py
python tools/export_report.py --use-saved-figures
python tools/build_visual_report.py
```

These presentation commands do not rerun the sensor or navigation. To export
only one document format, use `tools/export_report.py --format docx` or
`--format pdf`.

## Simulation videos

```sh
python tools/generate_environment_video.py
python tools/generate_sensor_state_video.py
```

Both generators save MP4 videos and source manifests in
`simulalted_environments/` (the requested folder spelling). They reuse recorded
scans when available and generate missing video inputs in a separate data
folder. The environment video compares Normal, Fog, and Rain at the same
simulation time. The sensor-state video compares the best and worst enabled
configurations among the tested settings, prioritizing goal completion,
collision avoidance, and clearance. Its manifest records the selection rule.

The videos display measured scans and accumulating maps alongside faint true
obstacle geometry. Recorded runs stop at their actual goal or collision time;
no later scans are invented. Weather models attenuation only. The coefficients
are illustrative and are not calibrated to a physical device.

## Files

| Deliverable | Location |
| --- | --- |
| Offline interactive report | `index.html` |
| Editable report and course PDF | `outputs/reports/` |
| Target and weather figures, trajectories, profiles, maps | `outputs/all_case_report/` |
| Fixed-scene 3D FOV/LOS geometry and figure | `outputs/weather_report/` |
| Threshold detection figure | `outputs/navigation_validation/report_threshold_detection.png` |
| Figure provenance | `outputs/report_figures_manifest.json` |
| Environment and sensor-state videos | `simulalted_environments/` |
| Submission archive and inventory | `outputs/submission/` |

```text
configs/                         Sensor and trajectory inputs
lidar_project/
  model/                         Geometry, materials, LiDAR measurement physics
  config.py, data.py, api.py      Validated inputs and point-cloud storage
  trajectory/                    Route planning, tracking, recorded runs
  navigation/control.py          Steering geometry used by trajectory tracking
  mapping/                       Measured-point accumulation and storage
  experiments/                   Report checks, fixed-scene studies, metrics
  visualization/                 Report figures, HTML assets, video renderer
  cli.py                         Numerical generation commands
  workflow.py, project_app.py     Deliverables builder and local server
tools/                           Report, video, and submission generators
tests/                           Numerical, storage, report, and workflow checks
data/                            Saved evidence and regenerated raw histories
outputs/                         Report figures and documents
simulalted_environments/          Generated videos and source manifests
build/                           Ignored caches, logs, and verification artifacts
```

Appearance changes belong in `lidar_project/visualization/`. The HTML report
uses `templates/report.html` and `static/report*`; video appearance is in
`simulation_video.py`. Neither renderer changes the sensor model.

## Verify and package

```sh
python -m unittest discover -s tests -v
python tools/package_project.py
```

The submission ZIP contains source, configurations, tests, the guide, reports,
figures, videos, CSV/JSON evidence, and threshold trials. Evidence is stored
under `evidence/data/` in the archive. Large navigation histories, caches, and
local logs are omitted; the documented generators rebuild the raw data.
`python run.py` provides the same numerical commands as `python -m lidar_project`.
