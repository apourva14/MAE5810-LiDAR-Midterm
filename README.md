# LiDAR midterm project

This is the single code folder for the 3D LiDAR simulation, navigation,
measurement evaluation, and course report. Numerical code and visualization
remain separate, so changing a figure or viewer does not rerun the sensor.

**Team Members for this project:**
Arunkumar Venkatesan - av652@cornell.edu
Xuyi Lian - xl2339@cornell.edu
Apurva Singh - as4642@cornell.edu

## Start here

[Read the live visual report](https://xuyilian.github.io/MAE5810-LiDAR-Midterm/).

Open **`index.html`** for the interactive visual report. It includes the report's
equations, rotatable 3D FOV/LOS diagram, recorded dependency comparisons,
threshold detection trials, navigation outcomes, and moving measured maps.
The page embeds its plotting library and data, works offline, and requires no
Python to read. Controls explore saved results or an explicitly labelled
analytical relationship; they do not generate deliverables or run experiments.

The GitHub checkout includes source, reports, media, numerical summaries, and
threshold trials. Large raw scan and navigation history arrays are excluded;
the documented simulation commands regenerate them. The embedded visual report
opens independently of these arrays.

For a local browser address, use Python 3.10 or newer:

```sh
python3 start.py
```

On macOS, `Start Project.command` launches the same page using `.venv` when
available. Keep its terminal window open. If the default port is occupied,
use `python3 start.py --port 0`. Add `--no-browser` to open the printed address
yourself. The server listens only on this computer at `127.0.0.1`.

## Reproduce or regenerate deliverables

To run simulations, render figures, or rebuild reports, create the environment once:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[visualization,reports]"
```

The optional deliverables builder is separate from the visual report. Start
the local server and open `http://127.0.0.1:8766/builder`:

```sh
python start.py
```

1. Choose **Generate all deliverables** to verify inputs and generate the
   report, figures, interactive media, and submission archive.
2. Leave **Recompute numerical studies** unchecked to reuse audited saved
   measurements. Missing inputs are generated automatically. The first build
   refreshes presentations; later builds reuse unchanged outputs.
3. Select a single step to generate it together with its prerequisites.
   Rendering steps leave numerical measurements unchanged.
4. To regenerate numerical results after changing a model or configuration,
   select **Recompute numerical studies**. Previous owned data is moved to
   `build/backups/`; failed replacement runs restore it. A complete fresh run
   includes 54,000 controlled sensor trials, 33,000 threshold trials, and 720
   navigation robustness runs, in addition to the smaller scene studies.

The page shows real build progress, errors, and links to completed files.
Build records and diagnostic logs are in ignored `build/`; they are not part
of maintained documentation or the submission archive.

After saved results or report equations change, refresh only the visual report:

```sh
python tools/build_visual_report.py
```

This presentation exporter reads existing data and the editable report without
running the sensor, planner, or navigation studies. Its HTML, CSS, and JavaScript
sources are in `lidar_project/visualization/templates/report.html` and
`lidar_project/visualization/static/report*`.

## Generated deliverables

| Deliverable | Location |
| --- | --- |
| Offline interactive visual report | `index.html` |
| Editable report and 5–10 page PDF | `outputs/reports/` |
| 3D FOV/LOS, threshold and robustness figures; evaluation viewer | `outputs/navigation_validation/` |
| Navigation dependency comparisons and moving 3D maps | `outputs/navigation_dependencies/` |
| Moving 3D map viewer for three environments | `outputs/mapping/` |
| Route figures, recorded animation and interactive viewer | `outputs/trajectory/` |
| Point cloud, scan figures and interactive 3D view | `outputs/scan/` |
| Controlled geometry and optical-response figures | `outputs/evaluation/` |
| Source, reports, media and evidence archive | `outputs/submission/Midterm_Project_Submission.zip` |
| Archive file inventory and SHA-256 hashes | `outputs/submission/deliverables_manifest.json` |

PDF export requires LibreOffice. The managed bundled renderer is used when
available. On another computer, set `LIDAR_SOFFICE` to the full executable
path before starting the builder; for example, on macOS:

```sh
export LIDAR_SOFFICE="/Applications/LibreOffice.app/Contents/MacOS/soffice"
python start.py
```

Install Times New Roman to reproduce the course formatting. Word generation
works separately if PDF export is unavailable. PDF export validates the page
limit and does not publish an invalid or placeholder PDF. The original course
workspace reports outside this code folder remain separate copies.

The submission ZIP contains the code, configurations, tests, README, report,
media, CSV/JSON evidence, and raw threshold trials. Review evidence is under
`evidence/data/` inside the archive. Large raw navigation scan/history archives
are omitted; after extracting, **Generate all deliverables** rebuilds the live
`data/` folder using the fixed study designs and seeds. Interactive media also
open offline without running the builder.

## Structure

```text
index.html                       Offline interactive visual report
builder.html                     Optional deliverables generation interface
start.py                         Starts the loopback-only Python server
Start Project.command            macOS launcher
configs/                         Sensor, navigation, and trajectory inputs
lidar_project/
  model/                         Materials, intersections, sensor physics
  config.py                      Configuration parsing and validation
  data.py                        Point clouds and scan storage
  navigation/                    Controller, occupancy grid, run storage
  trajectory/                    Reference planning, LiDAR tracking, run storage
  mapping/                       Accumulation of measured surfaces at known poses
  experiments/                   Controlled studies and analytical references
  visualization/                 All presentation code
    default_theme.json           Colors, fonts, colormaps, camera appearance
    scan_figures.py              Static scan and report figures
    scan_viewer.py               Offline point-cloud viewer
    navigation_figures.py        Navigation plots and GIFs
    trajectory.py                Trajectory study plots and stored-motion viewer
    first_person.py              Camera rendering and LiDAR overlay
    live.py                      Live and recorded-frame views
    browser.py                   Local browser transport
    templates/                   HTML layouts
    static/                      CSS and JavaScript
  cli.py                         Commands connecting the layers
  workflow.py                    Fixed tasks, dependencies, caching and rebuilds
  project_app.py                 Local HTTP interface to the fixed workflow
data/scan/                       Saved numerical scan and input snapshot
data/navigation/                 Navigation logs, summaries, replay archives
outputs/scan/                    Scan figures and interactive HTML
outputs/navigation/              Navigation figures, snapshots, GIFs
data/trajectory/                 Reference routes, executed states, factor study
outputs/trajectory/              Trajectory figures, animation, interactive viewer
data/mapping/                    Three environment comparisons and measured maps
outputs/mapping/                 Moving 3D map viewer and comparison figures
data/navigation_dependencies/    Matched-pose and closed-loop dependency studies
outputs/navigation_dependencies/ Dependency figures and moving 3D comparison
data/navigation_validation/      Prepared FOV/LOS ray geometry
data/navigation_threshold/       Independent threshold trials at saved poses
data/navigation_robustness/      Repeated closed-loop trials in fixed layouts
outputs/navigation_validation/  Validation figures and offline HTML
tools/build_model_notes.py       Mathematical model notes PDF builder
tools/build_report.py            Editable report from saved evaluation results
tools/build_visual_report.py     Offline visual report from saved data and equations
tools/export_report.py           Isolated PDF export and page-limit validation
tools/package_project.py         Verified source/report/media ZIP
outputs/reports/                 Generated Word/PDF reports
outputs/submission/              Submission ZIP and file manifest
build/                           Ignored build records, logs and data backups
tests/                           Numerical regression and storage checks
```

## Command-line use

The optional builder uses these same commands. For numerical simulation only, install
with `python -m pip install -e .`; renderers and reports need their extras.

## Run simulations

```sh
python -m lidar_project scan
python -m lidar_project scan --set sensor.rain_mm_per_h=25
python -m lidar_project navigate
python -m lidar_project navigate --scenario dense_fog
```

These commands write numerical data without loading visualization libraries.
Use `--data-dir PATH` to save a separate experiment instead of replacing a saved run.
Configuration references resolve relative to their configuration file.

## Render saved data or run a live view

```sh
python -m lidar_project render-scan
python -m lidar_project render-navigation
python -m lidar_project render-navigation --gifs
python -m lidar_project live --scenario clear
python -m lidar_project live --backend browser
python -m lidar_project replay --scenario clear --backend browser
```

Rendering commands read existing data without running the sensor or planner.
New navigation runs save complete numerical history for animations and replay.
The retained CSV-only navigation runs support static plots; their existing GIFs
remain in `outputs/navigation/`, but regenerating GIFs or replay requires a run
saved with numerical history. Both browser viewers work without a CDN connection.

## Change the visualization

Edit `lidar_project/visualization/default_theme.json` for default appearance, or
pass `--theme PATH.json` to a rendering, live, or replay command for partial theme
overrides. Layouts and interactions live in the visualization modules,
`templates/`, and `static/`. The numerical modules do not import this layer.

Use `--output-dir PATH` to preview a new style without replacing existing media.

## Controlled evaluation

```sh
python -m lidar_project evaluate
python -m lidar_project render-evaluation
```

The numerical command saves geometry, FOV, LOS, target-response, atmosphere,
and repeated-noise studies under `data/evaluation/`, including configurations,
seeds, analytical references, raw measurements, and CSV summaries. Use
`--part geometry` or `--part response` to run one study. The rendering command
reads those files and saves figures under `outputs/evaluation/`; it accepts
`--theme`, `--data-dir`, and `--output-dir` independently of simulation.

After installing the report extras and running the controlled sensor study,
navigation dependency study, and strengthened validation workflow below,
`python tools/build_report.py` generates the
editable course report. It creates compact report figures from saved data;
the interactive viewer and full comparison plots retain their own render command.

## Trajectory generation and evaluation

```sh
python -m lidar_project trajectory
python -m lidar_project evaluate-trajectory --seeds 1 2 3
python -m lidar_project render-trajectory --gifs
```

`configs/trajectory.jsonc` defines the robot, known planning map, physical scene,
sensor, and tracking settings. The reference route uses the known map; the
executed route uses LiDAR obstacle returns and an assumed known robot pose.
An additional obstacle is present in the physical scene but omitted from the
planning map. It tests whether sensing changes the motion around the reference.

The study changes one factor at a time and stores every run, including failures.
Metrics distinguish spatial route deviation from error against the timed
reference, and measure goal completion, collisions, travel time, path length,
minimum robot-surface gap, and control variation. Rendering reads saved data
without rerunning the planner or sensor. Use `--data-dir` and `--output-dir` to
keep alternative experiments and styles separate; `--theme` changes appearance.

## Map and compare environments in 3D

```sh
python -m lidar_project map-environments --seeds 1 2 3
python -m lidar_project render-mapping
```

The presets in `lidar_project/experiments/environments.py` define open,
cluttered, and narrow scenes with the same robot and sensor settings. Each
scene has an incomplete planning map and additional physical obstacles.
Recorded scans retain full 3D returns; the original controller still uses its
separate height-filtered obstacle points.

The viewer compares scenes at the same simulation time and supports playback,
rewinding, camera rotation, and optional scene geometry. The accumulating
voxel map contains measured surfaces using the simulator's known robot pose;
it does not estimate pose. The top-down map distinguishes unknown cells,
observed ground, and observed elevated surfaces. Observed endpoint area is
not a measure of free or traversable space. Both maps use the same displayed
region across scenes, and rewinding excludes future measurements.

Legacy trajectory archives without recorded 3D scans remain readable for their
original views; generating a measured 3D map requires a newly recorded run.

## Evaluate dependencies using navigation

```sh
python -m lidar_project evaluate-navigation-dependencies --seeds 1 2 3
python -m lidar_project render-navigation-dependencies
```

The shared cluttered task varies sensor mounting, FOV, occlusion, weather,
target features, sampling, and receiver noise. Matched-pose scans replay the
baseline robot poses to isolate measurement changes; closed-loop runs measure
the resulting navigation and map behavior with the same reference route.
Target diagnostics separate angular sampling, occlusion, range eligibility,
and threshold detection. All outcomes, including collision and timeout
prefixes, remain in the saved comparisons.

`sensor_mount` in the trajectory configuration accepts `offset_xy_m`,
`roll_deg`, `pitch_deg`, and `yaw_deg`; `mount_height_m` sets its height.
Renderers read saved numerical data and support independent `--data-dir`,
`--output-dir`, and `--theme` options.

## Strengthened navigation validation

Run the navigation dependency study above first, then prepare and evaluate:

```sh
python -m lidar_project prepare-navigation-coverage
python -m lidar_project evaluate-navigation-threshold --trials 1000
python -m lidar_project evaluate-navigation-robustness
python -m lidar_project diagnose-navigation-failures
python -m lidar_project render-navigation-validation
```

Coverage preparation saves deterministic FOV/LOS ray audits at an existing
navigation pose. Threshold trials reuse three original navigation rays and
vary illustrative atmospheric extinction to span the detector transition;
the receiver, target reflectance, and geometry stay fixed within each view.
Use `--source-dir` to choose another saved dependency study and `--data-dir`
to keep prepared geometry or numerical trial results separate.

Robustness evaluation repeats sensing conditions in the three fixed layouts.
Its default noise seeds are 1001 through 1020; `--seeds` accepts an explicit
list. Saved outcomes retain failures, pointwise success intervals, and paired
comparisons. These repetitions measure stochastic behavior within each fixed
scene, rather than sampling new environments.
Failure diagnostics read the completed default study's recorded trajectories
and audit the mounting-height effect at identical saved poses; they do not
rerun navigation or draw sensor noise. Custom seed subsets may omit the
selected diagnostic runs.

The renderer reads only saved data and creates report figures plus a
self-contained `navigation_validation.html`. It accepts `--diagram`,
`--threshold-dir`, `--robustness-dir`, `--output-dir`, and `--theme` independently
of the numerical studies.

## Checks and model notes

```sh
python -m lidar_project selftest
python -m unittest discover -s tests -v
python -m pip install -e ".[reports]"
python tools/build_model_notes.py
```

Build just the editable report or export its PDF:

```sh
python tools/export_report.py --format docx
python tools/export_report.py --format pdf
```

Generated reports are in this folder's `outputs/reports/`.
`python run.py` provides the same commands as `python -m lidar_project`.
