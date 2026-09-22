"""Builds assets/plotly/training-roofline.html, the interactive FSDP/TP roofline.

Plots forward-pass MLP compute time and communication time against batch size
on a TPU v5p 16x16x16, with a slider over power-of-two FSDP/TP splits. Times come
from the mixed FSDP/TP model in the training chapter (T_FSDP comms, T_TP comms
and T_math, with comms on the two axes fully overlapped). The page loads plotly
from a CDN and follows the parent page's light/dark theme.

Usage:
  uv run --with plotly python _scripts/training_roofline_plotly.py [-o OUT]
"""

import argparse
import json
import math
import pathlib

import plotly.graph_objects as go

# Model constants, matching the chapter: D=8192, F=32768, TPU v5p bf16 FLOPs/s,
# and ICI arithmetic intensity alpha = C / W_ici = 2550.
D_MODEL, D_FF = 8192, 32768
MESH_AXIS, MESH_DIMS = 16, 3
N_CHIPS = MESH_AXIS ** MESH_DIMS
PEAK_FLOPS = 4.59e14
W_ICI = PEAK_FLOPS / 2550
BATCH_SIZES = [10 ** (1 + 7 * i / 63) for i in range(64)]
MATH_PER_B = 4 * D_MODEL * D_FF / (N_CHIPS * PEAK_FLOPS)  # T_math / B.
INITIAL_FSDP = 512  # Crosses over near B = 4e5, as the chapter predicts.

FONT = ('-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
        '"Helvetica Neue", Arial, sans-serif')
COMPUTE_COLOR = '#1f2328'
COMMS_COLOR = '#b509ac'  # The site's theme color.
SHADE = 'rgba(181, 9, 172, 0.07)'

# Colors swapped in by the embedded script when the parent page is dark.
DARK_THEME = {
    'ink': '#e6e6e6', 'muted': '#9aa0a6', 'grid': 'rgba(255,255,255,0.08)',
    'compute': '#e6e6e6', 'comms': '#2698ba',
    'shade': 'rgba(38, 152, 186, 0.12)',
}
LIGHT_THEME = {
    'ink': '#1f2328', 'muted': '#5b6470', 'grid': 'rgba(0,0,0,0.07)',
    'compute': COMPUTE_COLOR, 'comms': COMMS_COLOR, 'shade': SHADE,
}


def mesh_axes(n: int) -> int:
  """Number of 16-wide mesh axes an n-way sharding spans (0 if unsharded)."""
  return math.ceil(math.log(n, MESH_AXIS) - 1e-9) if n > 1 else 0


def sharding_times(fsdp: int) -> tuple[list[float], float | None]:
  """Communication time per batch size for one FSDP/TP split.

  Args:
    fsdp: FSDP degree X; the TP degree is N_CHIPS / X.

  Returns:
    The comms time at each of BATCH_SIZES, and the batch size above which the
    layer is compute-bound (None if it never is).
  """
  tp = N_CHIPS // fsdp
  fsdp_time = (4 * D_MODEL * D_FF / (tp * W_ICI * mesh_axes(fsdp))
               if fsdp > 1 else 0.0)
  tp_per_b = 4 * D_MODEL / (fsdp * W_ICI * mesh_axes(tp)) if tp > 1 else 0.0
  comms = [max(fsdp_time, tp_per_b * b) for b in BATCH_SIZES]
  cross = fsdp_time / MATH_PER_B if tp_per_b < MATH_PER_B else None
  return comms, cross


def time_label(exponent: int) -> str:
  """Formats 10**exponent seconds with an SI prefix, e.g. -4 -> '100µs'."""
  prefixes = {-9: 'n', -6: 'µ', -3: 'm', 0: ''}
  base = 3 * math.floor(exponent / 3)
  return f'{10 ** (exponent - base)}{prefixes[base]}s'


def time_text(seconds: float) -> str:
  """Formats a duration to three significant figures, e.g. 3.9e-4 -> '390µs'."""
  for scale, unit in [(1, 's'), (1e-3, 'ms'), (1e-6, 'µs')]:
    if seconds >= scale:
      return f'{seconds / scale:.3g}{unit}'
  return f'{seconds / 1e-9:.3g}ns'


def short_count(n: float) -> str:
  """Formats a count to two significant figures, e.g. 390084 -> '390k'."""
  for scale, suffix in [(1e6, 'M'), (1e3, 'k'), (1, '')]:
    if n >= scale:
      return f'{float(f"{n / scale:.2g}"):g}{suffix}'
  return f'{n:.2g}'


def step_layout(x_min: float, cross: float | None) -> dict:
  """Layout overrides (shading + annotation) for one slider position.

  Args:
    x_min: The smallest plotted batch size, where the shading starts.
    cross: The crossover batch size, or None if comms-bound everywhere.

  Returns:
    A dict with 'shapes' and 'annotations' for Plotly's update method.
  """
  # With no crossover the shade spans the plot in paper coordinates; a huge
  # data-space x1 would drag the x-axis autorange out with it.
  shade = {
      'type': 'rect', 'xref': 'paper' if cross is None else 'x', 'yref': 'paper',
      'x0': 0 if cross is None else x_min, 'x1': 1 if cross is None else cross,
      'y0': 0, 'y1': 1, 'fillcolor': SHADE, 'line': {'width': 0},
      'layer': 'below',
  }
  note = {
      'xref': 'paper', 'x': 1, 'yref': 'paper', 'y': 1, 'showarrow': False,
      'xanchor': 'right', 'yanchor': 'top', 'xshift': -6, 'yshift': -4,
      'text': '<b>Comms-bound</b> at every batch size',
      'font': {'size': 12, 'color': COMMS_COLOR},
  }
  if cross is None:
    return {'shapes': [shade], 'annotations': [note]}
  rule = {
      'type': 'line', 'xref': 'x', 'yref': 'paper', 'x0': cross, 'x1': cross,
      'y0': 0, 'y1': 1, 'line': {'color': COMMS_COLOR, 'width': 1,
                                 'dash': 'dot'},
  }
  note.update(xref='x', x=math.log10(cross),
              text=f'<b>Comms-bound</b><br>below B ≈ {short_count(cross)}',
              align='right')
  return {'shapes': [shade, rule], 'annotations': [note]}


def build_figure() -> go.Figure:
  """Assembles the figure: one compute trace plus one comms trace per split."""
  xs = BATCH_SIZES
  compute = [MATH_PER_B * b for b in xs]
  splits = [2 ** k for k in range(12, -1, -1)]
  initial_step = splits.index(INITIAL_FSDP)
  results = [sharding_times(x) for x in splits]
  hover = 'B = %{x:,.0f}<br>%{customdata}<extra>%{fullData.name}</extra>'

  def trace(ys: list[float], **kwargs) -> go.Scatter:
    return go.Scatter(x=xs, y=ys, mode='lines', hovertemplate=hover,
                      customdata=[time_text(y) for y in ys], **kwargs)

  # Comms traces go first so the compute line draws on top where they overlap.
  fig = go.Figure()
  for i, (comms, _) in enumerate(results):
    fig.add_trace(trace(comms, name='Comms time', visible=i == initial_step,
                        legendrank=2,
                        line={'color': COMMS_COLOR, 'width': 2.5}))
  fig.add_trace(trace(compute, name='Compute time', legendrank=1,
                      line={'color': COMPUTE_COLOR, 'width': 2}))

  steps = []
  for i, (x, (_, cross)) in enumerate(zip(splits, results)):
    visible = [j == i for j in range(len(splits))] + [True]
    steps.append({
        'label': f'FSDP {x}, TP {N_CHIPS // x}', 'method': 'update',
        'args': [{'visible': visible}, step_layout(xs[0], cross)],
    })

  axis = {
      'type': 'log', 'showline': True, 'linewidth': 1, 'ticks': 'outside',
      'ticklen': 4, 'zeroline': False, 'exponentformat': 'power',
      'title': {'standoff': 8, 'font': {'size': 13}},
  }
  # Fixed ranges (in log10 units) so the axes hold still as the slider moves.
  all_y = compute + [y for comms, _ in results for y in comms if y > 0]
  x_range = [math.log10(xs[0]), math.log10(xs[-1])]
  y_range = [math.floor(math.log10(min(all_y))),
             math.ceil(math.log10(max(all_y)))]
  y_exps = list(range(y_range[0], y_range[1] + 1))
  initial = step_layout(xs[0], results[initial_step][1])
  fig.update_layout(
      font={'family': FONT, 'size': 12},
      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
      title={'text': 'MLP roofline on a TPU v5p 16x16x16 (D=8192, F=32768)',
             'x': 0, 'xref': 'paper', 'xanchor': 'left', 'y': 0.98,
             'yanchor': 'top', 'font': {'size': 15}},
      margin={'l': 64, 'r': 16, 't': 64, 'b': 8},
      xaxis={**axis, 'title': {**axis['title'], 'text': 'Batch size (tokens)'},
             'dtick': 1, 'range': x_range},
      yaxis={**axis, 'title': {**axis['title'], 'text': 'Time per layer'},
             'range': y_range, 'tickmode': 'array',
             'tickvals': [10.0 ** e for e in y_exps],
             'ticktext': [time_label(e) for e in y_exps]},
      legend={'orientation': 'h', 'x': 0, 'y': 1.02, 'xanchor': 'left',
              'yanchor': 'bottom', 'font': {'size': 12},
              'bgcolor': 'rgba(0,0,0,0)'},
      hovermode='x unified',
      hoverlabel={'font': {'family': FONT}},
      sliders=[{
          'active': initial_step, 'steps': steps, 'pad': {'t': 40, 'b': 0},
          'x': 0, 'len': 1, 'ticklen': 0, 'minorticklen': 0,
          'tickcolor': 'rgba(0,0,0,0)',
          'bgcolor': '#e8e8ec', 'activebgcolor': COMMS_COLOR,
          'bordercolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
          # Twenty tick labels never fit, so hide them and show the current one.
          'font': {'color': 'rgba(0,0,0,0)', 'size': 1},
          'currentvalue': {'prefix': '', 'xanchor': 'left',
                           'font': {'size': 13}, 'offset': 10},
      }],
      **initial,
  )
  return fig


# Runs inside the iframe: sizes the plot to the frame (tightening it on
# phones) and follows the parent page's light/dark theme.
POST_SCRIPT = """
(function() {
  var gd = document.getElementById('{plot_id}');
  var LIGHT = __LIGHT__, DARK = __DARK__;
  var root = null;
  try { root = window.parent.document.documentElement; } catch (e) {}

  function isDark() {
    if (root) return root.getAttribute('data-theme') === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  function recolor(layout, t) {
    layout.shapes.forEach(function(s) {
      if (s.type === 'rect') s.fillcolor = t.shade; else s.line.color = t.comms;
    });
    layout.annotations.forEach(function(a) { a.font.color = t.comms; });
  }

  function applyTheme() {
    var t = isDark() ? DARK : LIGHT;
    var slider = gd.layout.sliders[0];
    slider.steps.forEach(function(s) { recolor(s.args[1], t); });
    var current = slider.steps[slider.active].args[1];
    Plotly.update(gd, {'line.color': gd.data.map(function(_, i) {
      return i === gd.data.length - 1 ? t.compute : t.comms;
    })}, {
      'font.color': t.ink, 'legend.font.color': t.ink,
      'xaxis.color': t.muted, 'xaxis.linecolor': t.muted,
      'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.ink,
      'yaxis.color': t.muted, 'yaxis.linecolor': t.muted,
      'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.ink,
      'sliders[0].activebgcolor': t.comms,
      'sliders[0].bgcolor': isDark() ? '#3a3f44' : '#e8e8ec',
      'sliders[0].currentvalue.font.color': t.ink,
      shapes: JSON.parse(JSON.stringify(current.shapes)),
      annotations: JSON.parse(JSON.stringify(current.annotations)),
    });
  }

  // Full-density y ticks; phones keep every other decade.
  var yTicks = {vals: gd.layout.yaxis.tickvals.slice(),
                text: gd.layout.yaxis.ticktext.slice()};
  function keep(narrow) {
    return function(_, i) { return !narrow || i % 2 === 0; };
  }

  function fit() {
    var narrow = window.innerWidth < 520;
    Plotly.relayout(gd, {
      width: window.innerWidth, height: window.innerHeight,
      'margin.l': narrow ? 52 : 64, 'xaxis.dtick': narrow ? 2 : 1,
      'yaxis.tickvals': yTicks.vals.filter(keep(narrow)),
      'yaxis.ticktext': yTicks.text.filter(keep(narrow)),
      'title.font.size': narrow ? 13 : 15,
      'font.size': narrow ? 11 : 12,
    });
  }

  fit();
  applyTheme();
  window.addEventListener('resize', fit);
  if (root) {
    new MutationObserver(applyTheme).observe(
        root, {attributes: true, attributeFilter: ['data-theme']});
  }
})();
""".replace('__LIGHT__', json.dumps(LIGHT_THEME)).replace(
    '__DARK__', json.dumps(DARK_THEME))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--out',
                      default='assets/plotly/training-roofline.html')
  args = parser.parse_args()

  fig = build_figure()
  fig.write_html(
      args.out, include_plotlyjs='cdn', full_html=True,
      config={'displayModeBar': False}, post_script=POST_SCRIPT)
  # No page margin or scrollbars; the post script sizes the plot to the frame.
  html = pathlib.Path(args.out).read_text()
  html = html.replace(
      '<head>',
      '<head><meta name="viewport" content="width=device-width">'
      '<style>html,body{margin:0;overflow:hidden;background:transparent}'
      '</style>', 1)
  pathlib.Path(args.out).write_text(html)


if __name__ == '__main__':
  main()
