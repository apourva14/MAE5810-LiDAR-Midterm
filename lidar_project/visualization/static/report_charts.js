/* Offline report charts: saved observations and explicitly labelled theory. */
(function () {
  'use strict';
  const C = {blue: '#2579A8', orange: '#C8753F', teal: '#3D9C94', ink: '#243844', muted: '#61727B', grid: '#E6ECEF'};
  const CONFIG = {responsive: true, displaylogo: false, scrollZoom: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d']};
  let DATA;
  const rendered = new Set();
  const byId = id => document.getElementById(id);
  const finite = value => value !== null && value !== undefined && Number.isFinite(Number(value));
  const format = (value, digits = 2) => finite(value) ? Number(value).toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits}) : '—';
  const integer = value => format(value, 0);
  const percent = value => finite(value) ? format(Number(value) * 100, 1) + '%' : '—';
  const text = (id, value) => {if (byId(id)) byId(id).textContent = value;};
  const lines = (id, values) => {
    const host = byId(id);
    if (!host) return;
    host.replaceChildren(...values.map(value => {
      const line = document.createElement('span');
      line.style.display = 'block';
      line.textContent = value;
      return line;
    }));
  };
  const axis = (title, extra = {}) => ({title: {text: title, standoff: 12},
    automargin: true, gridcolor: C.grid, zerolinecolor: '#ACBBC2', linecolor: '#C6D1D6',
    tickfont: {size: 11}, ...extra});
  const layout = (xlabel, ylabel, extra = {}) => ({
    autosize: true, height: 320, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#FFFFFF',
    font: {family: 'Arial, sans-serif', size: 12, color: C.ink},
    margin: {l: 65, r: 24, t: 30, b: 66},
    xaxis: axis(xlabel), yaxis: axis(ylabel),
    legend: {orientation: 'h', x: 0, y: 1.16, font: {size: 11}},
    hoverlabel: {font: {size: 12}, bgcolor: '#FFFFFF'},
    ...extra
  });
  function plot(id, traces, spec) {
    const host = byId(id);
    if (!host) return Promise.resolve();
    rendered.add(id);
    return Plotly.react(host, traces, spec, CONFIG);
  }
  function optionList(id, values, selected) {
    const host = byId(id);
    if (!host) return;
    host.replaceChildren(...values.map(([value, label]) => {
      const option = document.createElement('option');
      option.value = String(value); option.textContent = label;
      return option;
    }));
    host.value = String(selected);
  }
  const errorBars = (centers, lows, highs) => ({type: 'data', symmetric: false,
    array: highs.map((high, i) => Math.max(0, high - centers[i])),
    arrayminus: lows.map((low, i) => Math.max(0, centers[i] - low)), thickness: 1.25, width: 4});
  const normalCDF = value => {
    const absolute = Math.abs(value), t = 1 / (1 + 0.2316419 * absolute);
    const tail = Math.exp(-absolute * absolute / 2) / Math.sqrt(2 * Math.PI) *
      t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return value < 0 ? tail : 1 - tail;
  };

  async function renderPower() {
    const get = (id, fallback) => byId(id) ? Number(byId(id).value) : fallback;
    const range = get('power-range', 5), rho = get('power-rho', .3);
    const kappa = get('power-kappa', .0002), angle = get('power-angle', 0);
    text('power-range-value', format(range, 1) + ' m');
    text('power-rho-value', format(rho, 2));
    text('power-kappa-value', format(kappa, 4) + ' m⁻¹');
    text('power-angle-value', format(angle, 0) + '°');
    const ratio = (r, reflectance, extinction, degrees) =>
      reflectance / .3 * Math.cos(degrees * Math.PI / 180) * (5 / r) ** 2 *
      Math.exp(-2 * (extinction * r - .0002 * 5));
    const xs = Array.from({length: 160}, (_, i) => 1 + 19 * i / 159);
    const value = ratio(range, rho, kappa, angle);
    const number = value >= .001 && value < 1000 ? format(value, 3) : value.toExponential(3);
    lines('power-readout', [
      'Model prediction: relative returned power = ' + number + ' × the reference.',
      'Reference: 5 m, diffuse weight 0.30, normal incidence, κ = 0.0002 m⁻¹. Fixed optics; eligible, visible, beam-filling diffuse echo.',
      'These sliders evaluate Equation (7). They do not create measurements or rerun a navigation experiment.'
    ]);
    return plot('power-model-plot', [
      {x: xs, y: xs.map(r => ratio(r, .3, .0002, 0)), name: 'Reference surface / air',
        mode: 'lines', line: {color: '#9AA9B1', dash: 'dash', width: 2},
        hovertemplate: 'Range %{x:.2f} m<br>Reference %{y:.4g} ×<extra></extra>'},
      {x: xs, y: xs.map(r => ratio(r, rho, kappa, angle)), name: 'Selected model parameters',
        mode: 'lines', line: {color: C.blue, width: 3},
        hovertemplate: 'Range %{x:.2f} m<br>Predicted power %{y:.4g} ×<extra></extra>'},
      {x: [range], y: [value], name: 'Selected range', mode: 'markers',
        marker: {color: C.orange, size: 10, line: {color: '#FFFFFF', width: 2}},
        hovertemplate: 'Range %{x:.2f} m<br>Predicted power %{y:.4g} ×<extra></extra>'}
    ], layout('Range (m)', 'Relative returned power (log scale)', {
      yaxis: axis('Relative returned power', {type: 'log'}),
      shapes: [{type: 'line', x0: range, x1: range, y0: 0, y1: 1, yref: 'paper',
        line: {color: C.orange, dash: 'dot', width: 1}}]
    }));
  }

  async function renderDependencies() {
    const cases = DATA.primary.cases;
    const selected = cases.find(c => c.id === (byId('dependency-case')?.value || 'mount_high')) || cases[0];
    const reference = cases.find(c => c.id === selected.comparison_baseline) || cases.find(c => c.id === 'baseline');
    const pair = [reference, selected];
    const labels = ['Target intersections', 'Visible target hits', 'In-range echoes', 'Detected returns'];
    const keys = ['target_opportunities', 'visible_target_echoes', 'eligible_echoes', 'detected_returns'];
    const stages = pair.map((item, index) => ({type: 'bar', name: item.label + (index ? ' (selected)' : ' (reference)'),
      x: labels, y: keys.map(key => item.replay[key]), marker: {color: index ? C.orange : C.blue},
      hovertemplate: '%{x}<br>%{y:,} target echoes<extra>%{fullData.name}</extra>'}));
    const power = pair.map((item, index) => ({type: 'bar', x: [index ? 'Selected' : 'Reference'],
      y: [finite(item.replay.mean_eligible_power_w) ? 1000 * item.replay.mean_eligible_power_w : null],
      name: item.label, legendgroup: String(index), showlegend: false,
      marker: {color: index ? C.orange : C.blue},
      hovertemplate: item.label + '<br>%{y:.6f} mW predicted eligible power<extra></extra>'}));
    const errors = pair.map((item, index) => ({type: 'bar', x: [index ? 'Selected' : 'Reference'],
      y: [finite(item.replay.range_rmse_m) ? 1000 * item.replay.range_rmse_m : null],
      xaxis: 'x2', yaxis: 'y2', name: item.label, showlegend: false,
      marker: {color: index ? C.orange : C.blue},
      hovertemplate: item.label + '<br>%{y:.3f} mm detected-return RMS<extra></extra>'}));
    const probability = selected.replay.observed_detection_probability;
    lines('dependency-summary', [selected.label + ': ' + selected.description,
      integer(selected.replay.eligible_echoes) + ' eligible target echoes → ' + integer(selected.replay.detected_returns) +
        ' detections; detected / eligible = ' + percent(probability) + '.',
      'Closed-loop task: ' + selected.aggregate.goals + '/' + selected.aggregate.runs + ' goals; worst swept surface gap ' +
        format(selected.aggregate.worst_surface_gap_m, 3) + ' m. This is the primary three-seed scene.']);
    let note = 'Replays use 271 identical baseline snapshots across three seeds. Stage counts isolate geometry and visibility before thresholding. Power is the noiseless prediction over eligible echoes; RMS includes detected target returns only. A missing conditional quantity is shown as —.';
    if (selected.id === 'fov_side') note += ' The reference is the matched front-facing 120° window, rotated 90° for the selected case.';
    if (selected.replay.eligible_echoes === 0) note += ' No eligible echoes: conditional detection probability, power, and range RMS are undefined.';
    text('dependency-note', note);
    return Promise.all([
      plot('dependency-stage-plot', stages, layout('', 'Target echoes / returns', {barmode: 'group',
        margin: {l: 68, r: 16, t: 50, b: 78}, xaxis: axis('', {tickangle: -12})})),
      plot('dependency-power-plot', [...power, ...errors], layout('', '', {
        margin: {l: 62, r: 62, t: 47, b: 50},
        xaxis: axis('', {domain: [0, .39], categoryorder: 'array', categoryarray: ['Reference', 'Selected']}),
        yaxis: axis('Eligible power (mW)', {domain: [0, 1], rangemode: 'tozero'}),
        xaxis2: axis('', {domain: [.61, 1], anchor: 'y2', categoryorder: 'array', categoryarray: ['Reference', 'Selected']}),
        yaxis2: axis('Detected-target RMS (mm)', {domain: [0, 1], anchor: 'x2', side: 'right', rangemode: 'tozero'}),
        annotations: [{xref: 'paper', yref: 'paper', x: .18, y: 1.12, showarrow: false, text: 'Predicted power'},
          {xref: 'paper', yref: 'paper', x: .84, y: 1.12, showarrow: false, text: 'Measured error'}]
      }))
    ]);
  }

  function renderDistance() {
    const rows = DATA.primary.common_viewpoint || [];
    return plot('distance-plot', [{type: 'scatter', mode: 'markers+lines',
      x: rows.map(r => r.mean_eligible_range_m), y: rows.map(r => r.mean_eligible_power_w * 1000),
      text: rows.map(r => r.label || ({target_near: 'Target at x = 9 m', baseline: 'Target at x = 11 m', target_far: 'Target at x = 13 m'}[r.case_id]) || r.case_id),
      marker: {color: C.teal, size: 10}, line: {color: C.teal, width: 2},
      customdata: rows.map(r => [r.eligible_echoes, r.time_s, r.seed]),
      hovertemplate: '%{text}<br>Mean range %{x:.3f} m<br>Eligible power %{y:.6f} mW<br>%{customdata[0]} eligible echoes; t=%{customdata[1]} s, seed %{customdata[2]}<extra></extra>'
    }], layout('Mean eligible range at one shared pose (m)', 'Mean eligible target power (mW)', {
      showlegend: false, height: 300,
      annotations: rows.map(r => ({x: r.mean_eligible_range_m, y: r.mean_eligible_power_w * 1000,
        text: ({target_near: 'x = 9 m', baseline: 'x = 11 m', target_far: 'x = 13 m'}[r.case_id]) || '',
        showarrow: true, arrowhead: 0, ay: -24, ax: 0, font: {size: 11}}))
    }));
  }

  async function renderThreshold() {
    const study = DATA.threshold, selection = byId('threshold-pose')?.value || 'all';
    const poses = study.views.filter(view => selection === 'all' || String(view.pose_id) === selection);
    const levels = [...new Set(study.settings.map(s => Math.round(s.mean_snr * 100) / 100))].sort((a, b) => a - b);
    const index = Math.max(0, Math.min(levels.length - 1, Number(byId('threshold-snr')?.value ?? 5)));
    const snr = levels[index];
    text('threshold-snr-value', 'Mean SNR ' + format(snr, 1));
    const xs = Array.from({length: 161}, (_, i) => 6 + 8 * i / 160);
    const traces = [{x: xs, y: xs.map(x => normalCDF(x - 10) * 100), mode: 'lines',
      name: 'Model Φ(SNR − 10)', line: {color: C.ink, width: 2},
      hovertemplate: 'Mean SNR %{x:.2f}<br>Predicted detection %{y:.2f}%<extra></extra>'}];
    const residuals = [], readout = [];
    for (const view of poses) {
      const rows = study.settings.filter(row => row.pose_id === view.pose_id).sort((a, b) => a.mean_snr - b.mean_snr);
      const color = [C.blue, C.orange, C.teal][view.pose_id % 3];
      const observed = rows.map(row => row.observed_pd * 100);
      const low = rows.map(row => row.wilson_low * 100), high = rows.map(row => row.wilson_high * 100);
      const offset = (view.pose_id - 1) * .045;
      const custom = rows.map(row => [row.mean_snr, row.detections, row.trials, row.expected_pd * 100,
        row.wilson_low * 100, row.wilson_high * 100, row.extinction_per_m]);
      traces.push({x: rows.map(row => row.mean_snr + offset), y: observed, mode: 'markers', name: view.label,
        marker: {color, size: rows.map(row => Math.abs(row.mean_snr - snr) < .01 ? 11 : 7),
          line: {color: '#FFFFFF', width: 1}}, error_y: {...errorBars(observed, low, high), color}, customdata: custom,
        hovertemplate: 'Mean SNR %{customdata[0]:.1f}<br>%{customdata[1]}/%{customdata[2]} detected<br>Observed %{y:.2f}%<br>Model %{customdata[3]:.2f}%<br>95% Wilson [%{customdata[4]:.2f}, %{customdata[5]:.2f}]%<br>κ=%{customdata[6]:.4f} m⁻¹<extra>%{fullData.name}</extra>'});
      const delta = rows.map(row => (row.observed_pd - row.expected_pd) * 100);
      residuals.push({x: rows.map(row => row.mean_snr + offset), y: delta, name: view.label, mode: 'markers',
        marker: {color, size: 7}, customdata: custom,
        error_y: {...errorBars(delta, rows.map(row => (row.wilson_low - row.expected_pd) * 100),
          rows.map(row => (row.wilson_high - row.expected_pd) * 100)), color},
        hovertemplate: 'Mean SNR %{customdata[0]:.1f}<br>Observed − predicted %{y:.3f} percentage points<extra>%{fullData.name}</extra>'});
      const chosen = rows.find(row => Math.abs(row.mean_snr - snr) < .01);
      if (chosen) readout.push(view.label + ': ' + integer(chosen.detections) + '/' + integer(chosen.trials) +
        ' detections (' + percent(chosen.observed_pd) + '); model ' + percent(chosen.expected_pd) +
        '; pointwise 95% Wilson [' + percent(chosen.wilson_low) + ', ' + percent(chosen.wilson_high) + '].');
    }
    readout.push('Every marker counts 1,000 independent actual scan trials, including misses. Horizontal offsets separate viewpoints; they do not change SNR. The shaded model transition is not a confidence interval.');
    lines('threshold-readout', readout);
    const shapes = [{type: 'rect', x0: 8.355146373, x1: 11.644853627, y0: 0, y1: 1, yref: 'paper',
      fillcolor: 'rgba(37,121,168,.07)', line: {width: 0}, layer: 'below'},
      {type: 'line', x0: snr, x1: snr, y0: 0, y1: 1, yref: 'paper', line: {color: C.orange, dash: 'dot', width: 1}}];
    return Promise.all([
      plot('threshold-plot', traces, layout('Mean echo power / noise RMS (SNR)', 'Detected / trials (%)', {
        xaxis: axis('Mean echo power / noise RMS (SNR)', {range: [5.8, 14.2]}),
        yaxis: axis('Detected / trials (%)', {range: [-3, 103]}), shapes})),
      plot('threshold-residual', residuals, layout('Mean echo power / noise RMS (SNR)', 'Observed − predicted (percentage points)', {
        height: 300, xaxis: axis('Mean echo power / noise RMS (SNR)', {range: [5.8, 14.2]}),
        shapes: [{type: 'line', x0: 5.8, x1: 14.2, y0: 0, y1: 0, line: {color: C.ink, width: 1, dash: 'dash'}}]
      }))
    ]);
  }

  const layoutColors = {open: C.blue, cluttered: C.orange, narrow: C.teal};
  const layoutNames = {open: 'Open', cluttered: 'Cluttered', narrow: 'Narrow'};
  const extinctionIds = ['baseline', 'fog_010', 'fog_030', 'fog_060', 'fog_120', 'fog_300', 'fog_1000'];
  const extinctionLabels = ['0.0002', '0.1', '0.3', '0.6', '1.2', '3', '10'];
  const ablationIds = ['baseline', 'mount_high', 'sampling_sparse', 'fov_front', 'fov_side', 'lidar_disabled'];
  const ablationLabels = ['Baseline', 'High mount', 'Sparse scan', 'Front FOV', 'Side FOV', 'Sensor off'];
  const conditionLabel = condition => condition?.label || condition?.id || condition?.case_id || 'Condition';
  function robustCell(layoutId, caseId) {
    return DATA.robustness.cases.find(row => row.layout_id === layoutId && row.case_id === caseId);
  }
  function robustRows() {
    const host = byId('robust-rows');
    if (!host) return;
    host.replaceChildren(...DATA.robustness.cases.map(row => {
      const tr = document.createElement('tr'), gap = row.all_run_prefix.swept_min_surface_gap_m;
      const success = row.successful_runs || {};
      const interval = row.goal_fraction_wilson95 || [null, null];
      const values = [layoutNames[row.layout_id] || row.layout_label, row.label,
        row.goals + '/' + row.collisions + '/' + row.timeouts,
        percent(row.goal_fraction) + ' [' + percent(interval[0]) + ', ' + percent(interval[1]) + ']',
        format(gap.median, 3) + ' [' + format(gap.q25, 3) + ', ' + format(gap.q75, 3) + ']',
        format(success.elapsed_s?.median, 2), format(success.path_length_m?.median, 2)];
      for (const value of values) {
        const cell = document.createElement('td'); cell.textContent = value; tr.append(cell);
      }
      return tr;
    }));
  }
  async function renderRobustness() {
    const selectedLayout = byId('robust-layout')?.value || 'all';
    const layouts = ['open', 'cluttered', 'narrow'].filter(value => selectedLayout === 'all' || value === selectedLayout);
    const selectedCase = byId('robust-case')?.value || 'mount_high';
    const goalTraces = [], gapTraces = [], ablationTraces = [], detail = [];
    for (const layoutId of layouts) {
      const color = layoutColors[layoutId], name = layoutNames[layoutId];
      const extinction = extinctionIds.map(id => robustCell(layoutId, id));
      const goals = extinction.map(row => row.goal_fraction * 100);
      goalTraces.push({x: extinctionLabels, y: goals, mode: 'lines+markers', name,
        marker: {color, size: 7}, line: {color, width: 2},
        error_y: {...errorBars(goals, extinction.map(row => row.goal_fraction_wilson95[0] * 100),
          extinction.map(row => row.goal_fraction_wilson95[1] * 100)), color},
        customdata: extinction.map(row => [row.goals, row.runs, row.collisions, row.timeouts,
          row.goal_fraction_wilson95[0] * 100, row.goal_fraction_wilson95[1] * 100]),
        hovertemplate: 'κ=%{x} m⁻¹<br>Goals %{customdata[0]}/%{customdata[1]} (%{y:.1f}%)<br>95% Wilson [%{customdata[4]:.1f}, %{customdata[5]:.1f}]%<br>Collisions %{customdata[2]}; timeouts %{customdata[3]}<extra>%{fullData.name}</extra>'});
      const gap = extinction.map(row => row.all_run_prefix.swept_min_surface_gap_m);
      gapTraces.push({x: extinctionLabels, y: gap.map(value => value.median), mode: 'lines+markers', name,
        marker: {color, size: 7}, line: {color, width: 2},
        error_y: {...errorBars(gap.map(value => value.median), gap.map(value => value.q25), gap.map(value => value.q75)), color},
        customdata: gap.map(value => [value.q25, value.q75, value.n]),
        hovertemplate: 'κ=%{x} m⁻¹<br>Median gap %{y:.3f} m<br>IQR [%{customdata[0]:.3f}, %{customdata[1]:.3f}] m<br>%{customdata[2]} runs, including failures<extra>%{fullData.name}</extra>'});
      const ablations = ablationIds.map(id => robustCell(layoutId, id));
      const rates = ablations.map(row => row.goal_fraction * 100);
      ablationTraces.push({type: 'bar', x: ablationLabels, y: rates, name, marker: {color},
        error_y: {...errorBars(rates, ablations.map(row => row.goal_fraction_wilson95[0] * 100),
          ablations.map(row => row.goal_fraction_wilson95[1] * 100)), color: C.ink},
        customdata: ablations.map(row => [row.goals, row.runs, row.collisions, row.timeouts]),
        hovertemplate: '%{x}<br>Goals %{customdata[0]}/%{customdata[1]}<br>Collisions %{customdata[2]}; timeouts %{customdata[3]}<extra>%{fullData.name}</extra>'});
      const row = robustCell(layoutId, selectedCase), metric = row.all_run_prefix.swept_min_surface_gap_m;
      const interval = row.goal_fraction_wilson95, success = row.successful_runs || {};
      detail.push(name + ' · ' + row.label + ': ' + row.goals + ' goals / ' + row.collisions + ' collisions / ' + row.timeouts +
        ' timeouts; goal fraction ' + percent(row.goal_fraction) + ' [95% Wilson ' + percent(interval[0]) + '–' + percent(interval[1]) + '].');
      detail.push('All-run gap median ' + format(metric.median, 3) + ' m [IQR ' + format(metric.q25, 3) + '–' + format(metric.q75, 3) +
        ']; successful-run median time ' + format(success.elapsed_s?.median, 2) + ' s; path ' + format(success.path_length_m?.median, 2) + ' m.');
    }
    detail.push('One whole run is the statistical unit (20 per layout and condition). Gap summaries retain failures; time and path condition on success. Intervals apply to these fixed layouts and model, and are pointwise. Non-monotonic samples do not establish non-monotonic reliability.');
    lines('robust-detail', detail);
    return Promise.all([
      plot('robust-goal', goalTraces, layout('Fixed extinction setting κ (m⁻¹)', 'Goal fraction (%)', {
        xaxis: axis('Fixed extinction setting κ (m⁻¹)', {type: 'category', categoryorder: 'array', categoryarray: extinctionLabels}),
        yaxis: axis('Goal fraction (%)', {range: [-3, 104]})})),
      plot('robust-gap', gapTraces, layout('Fixed extinction setting κ (m⁻¹)', 'Swept surface gap (m)', {
        xaxis: axis('Fixed extinction setting κ (m⁻¹)', {type: 'category', categoryorder: 'array', categoryarray: extinctionLabels}),
        shapes: [{type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 0, y1: 0, line: {color: C.orange, dash: 'dash', width: 1}}]})),
      plot('robust-ablation', ablationTraces, layout('', 'Goal fraction (%)', {barmode: 'group', height: 330,
        xaxis: axis('', {categoryorder: 'array', categoryarray: ablationLabels, tickangle: -15}),
        yaxis: axis('Goal fraction (%)', {range: [-3, 104]}), margin: {l: 65, r: 20, t: 32, b: 76}}))
    ]);
  }

  async function init(data) {
    DATA = data;
    optionList('dependency-case', data.primary.cases.map(row => [row.id, row.label]), 'mount_high');
    optionList('threshold-pose', [['all', 'All three viewpoints'], ...data.threshold.views.map(row => [row.pose_id, row.label])], 'all');
    const snr = byId('threshold-snr');
    if (snr) {snr.min = 0; snr.max = 10; snr.step = 1; snr.value = 5;}
    optionList('robust-layout', [['all', 'All three layouts'], ['open', 'Open'], ['cluttered', 'Cluttered'], ['narrow', 'Narrow']], 'all');
    optionList('robust-case', data.robustness.conditions.map(row => [row.id || row.case_id, conditionLabel(row)]), 'mount_high');
    for (const id of ['power-range', 'power-rho', 'power-kappa', 'power-angle']) byId(id)?.addEventListener('input', renderPower);
    byId('dependency-case')?.addEventListener('change', renderDependencies);
    byId('threshold-pose')?.addEventListener('change', renderThreshold);
    byId('threshold-snr')?.addEventListener('input', renderThreshold);
    byId('robust-layout')?.addEventListener('change', renderRobustness);
    byId('robust-case')?.addEventListener('change', renderRobustness);
    robustRows();
    await Promise.all([renderPower(), renderDependencies(), renderDistance(), renderThreshold(), renderRobustness()]);
  }
  function resize() {
    for (const id of rendered) {
      const host = byId(id);
      if (host && host.clientWidth && host.getClientRects().length) Plotly.Plots.resize(host);
    }
  }
  window.ReportCharts = {init, resize};
}());
