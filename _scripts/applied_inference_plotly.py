"""Builds the interactive LLaMA 3-70B generation figures in applied-inference.md.

Writes two pages, both with a slider over context length:

  assets/plotly/pareto.html: the latency/throughput Pareto frontier.
  assets/plotly/latency_breakdown_log.html: per-step time split into parameter
    loading, KV cache loading and FLOPs.

Times come from the simple roofline model in the chapter's "The code for this
is quite simple" block: LLaMA 3-70B with int8 weights and KV caches, 16-way
model parallelism on a TPU v5e 4x4. Each page loads plotly from a CDN and
follows the parent page's light/dark theme.

Usage:
  uv run --with plotly python _scripts/applied_inference_plotly.py
"""

import argparse
import collections.abc
import json
import math
import pathlib

import plotly.graph_objects as go

NUM_CHIPS = 16
PARAM_BYTES = 70e9  # int8, so one byte per parameter.
HBM_BW = 8.2e11  # TPU v5e bytes/s.
PEAK_FLOPS = 1.97e14  # TPU v5e bf16 FLOPs/s.
# Above this batch size the MLP FLOPs outlast the parameter loads (~120).
COMPUTE_BOUND_BATCH = PEAK_FLOPS / (2 * HBM_BW)

# Largest batch size plotted at each context length, i.e. roughly the most KV
# cache that fits in the 16 x 16GB of HBM next to the weights. These match the
# original figures (which don't follow a single exact rule), so keep them.
MAX_BATCH = {2048: 511, 4096: 276, 8192: 137, 16384: 68, 32768: 33,
             65536: 16}
CONTEXTS = list(MAX_BATCH)
INITIAL_STEP = 0
# Pareto dots are colored by log2(batch size) on one fixed scale, so a color
# means the same batch size at every context length.
BATCH_COLORBAR_TICKS = [1, 4, 16, 64, 256]
# Batch sizes labeled along the Pareto frontier, when they fit in memory.
LABELED_BATCHES = [1, 4, 16, 64, 256]
# A label within this factor of the largest batch would collide with the
# endpoint's label, so it is dropped.
MIN_LABEL_RATIO = 2

FONT = ('-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
        '"Helvetica Neue", Arial, sans-serif')

# Series colors per theme. The accent is the site's theme color; the three
# cost components use hues that stay distinct against both backgrounds.
LIGHT_THEME = {
    'ink': '#1f2328', 'muted': '#5b6470', 'grid': 'rgba(0,0,0,0.07)',
    'accent': '#b509ac', 'shade': 'rgba(181, 9, 172, 0.07)',
    'slider': '#e8e8ec', 'total': '#1f2328', 'params': '#1c7ed6',
    'kv': '#e8590c', 'flops': '#2f9e44', 'frontier': '#b4bac1',
    'labels': '#1f2328',
}
DARK_THEME = {
    'ink': '#e6e6e6', 'muted': '#9aa0a6', 'grid': 'rgba(255,255,255,0.08)',
    'accent': '#2698ba', 'shade': 'rgba(38, 152, 186, 0.12)',
    'slider': '#3a3f44', 'total': '#e6e6e6', 'params': '#4dabf7',
    'kv': '#ff922b', 'flops': '#51cf66', 'frontier': '#5c6370',
    'labels': '#e6e6e6',
}


def step_times(context: int) -> dict[str, collections.abc.Sequence[float]]:
  """Per-step decode times in milliseconds, for batch sizes 1..MAX_BATCH.

  Args:
    context: The sequence length whose KV cache each sequence reads.

  Returns:
    A dict of equal-length lists: 'batch', 'params', 'kv', 'flops', 'total'
    (times in ms) and 'throughput' (tokens / ms / chip).
  """
  batch = list(range(1, MAX_BATCH[context] + 1))
  params = PARAM_BYTES / (NUM_CHIPS * HBM_BW)
  # K and V, 8 KV heads of dim 128, 80 layers, one byte each.
  kv = [2 * (context * b) * 128 * 8 * 80 / (NUM_CHIPS * HBM_BW) for b in batch]
  flops = [2 * PARAM_BYTES * b / (NUM_CHIPS * PEAK_FLOPS) for b in batch]
  # The MLP overlaps weight loads with FLOPs; attention is always KV-bound.
  total = [1000 * (max(f, params) + k) for f, k in zip(flops, kv)]
  return {
      'batch': batch,
      'params': [1000 * params] * len(batch),
      'kv': [1000 * k for k in kv],
      'flops': [1000 * f for f in flops],
      'total': total,
      'throughput': [b / (t * NUM_CHIPS) for b, t in zip(batch, total)],
  }


def ms_label(value: float) -> str:
  """Formats a time in ms with an SI prefix, e.g. 0.1 -> '100µs', 20 -> '20ms'."""
  for scale, unit in [(1e3, 's'), (1, 'ms'), (1e-3, 'µs')]:
    if value >= scale:
      return f'{value / scale:.3g}{unit}'
  return f'{value / 1e-6:.3g}ns'


def log_range(lo: float, hi: float) -> list[float]:
  """Returns a fixed log-axis range [log10(lo), log10(hi)]."""
  return [math.log10(lo), math.log10(hi)]


def context_label(context: int) -> str:
  """Slider label for one context length, e.g. '8,192-token context'."""
  return f'{context:,}-token context'


def base_layout(title: str, steps: list[dict]) -> dict:
  """Layout shared by both figures: fonts, title, legend, axes and slider.

  Args:
    title: The figure title, drawn top left.
    steps: Slider steps, one per context length.

  Returns:
    Keyword arguments for Figure.update_layout.
  """
  theme = LIGHT_THEME
  axis = {
      'type': 'log', 'showline': True, 'linewidth': 1, 'ticks': 'outside',
      'ticklen': 4, 'zeroline': False, 'tickmode': 'array',
      'color': theme['muted'], 'linecolor': theme['muted'],
      'gridcolor': theme['grid'],
      'title': {'standoff': 8, 'font': {'size': 13, 'color': theme['ink']}},
  }
  return {
      'font': {'family': FONT, 'size': 12, 'color': theme['ink']},
      'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)',
      'title': {'text': title, 'x': 0, 'xref': 'paper', 'xanchor': 'left',
                'y': 0.98, 'yanchor': 'top', 'font': {'size': 15}},
      'margin': {'l': 64, 'r': 16, 't': 64, 'b': 8},
      'xaxis': axis, 'yaxis': axis,
      'legend': {'orientation': 'h', 'x': 0, 'y': 1.02, 'xanchor': 'left',
                 'yanchor': 'bottom', 'font': {'size': 12},
                 'bgcolor': 'rgba(0,0,0,0)', 'traceorder': 'normal'},
      'hoverlabel': {'font': {'family': FONT}},
      'sliders': [{
          'active': INITIAL_STEP, 'steps': steps, 'pad': {'t': 40, 'b': 0},
          'x': 0, 'len': 1, 'ticklen': 0, 'minorticklen': 0,
          'tickcolor': 'rgba(0,0,0,0)',
          'bgcolor': theme['slider'], 'activebgcolor': theme['accent'],
          'bordercolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
          # The current value names the context, so the ticks stay unlabeled.
          'font': {'color': 'rgba(0,0,0,0)', 'size': 1},
          'currentvalue': {'prefix': '', 'xanchor': 'left',
                           'font': {'size': 13}, 'offset': 10},
      }],
  }


def slider_steps(traces_per_step: int) -> list[dict]:
  """One slider step per context, showing that context's group of traces."""
  steps = []
  for i, context in enumerate(CONTEXTS):
    visible = [j // traces_per_step == i
               for j in range(len(CONTEXTS) * traces_per_step)]
    steps.append({'label': context_label(context), 'method': 'update',
                  'args': [{'visible': visible}]})
  return steps


PARETO_X = (5, 42)  # Per-token latency range, ms.
PARETO_Y = (0.006, 1.5)  # Throughput range, tokens/ms/chip.


def label_position(xs: collections.abc.Sequence[float],
                   ys: collections.abc.Sequence[float], i: int) -> str:
  """Where to put the label for point i so it clears the Pareto curve.

  Args:
    xs: Latencies along the curve.
    ys: Throughputs along the curve.
    i: The labeled point's index.

  Returns:
    A Plotly textposition: right of the curve where it climbs steeply, above
    and to the left where it runs flatter (the curve is concave, so it falls
    away below-left of the point), and below-right at a shallow start.
  """
  lo, hi = max(i - 1, 0), min(i + 1, len(xs) - 1)
  if lo == hi:
    return 'middle right'
  # Slope in screen-ish units, i.e. relative to each log axis's span. A label
  # to the right only clears the rising curve where it is steep.
  dx = math.log(xs[hi] / xs[lo]) / math.log(PARETO_X[1] / PARETO_X[0])
  dy = math.log(ys[hi] / ys[lo]) / math.log(PARETO_Y[1] / PARETO_Y[0])
  if dy > 2 * dx:
    return 'middle right'
  return 'bottom right' if i == 0 else 'top left'


def build_pareto() -> go.Figure:
  """Throughput against per-token latency, one curve per context length."""
  theme = LIGHT_THEME
  fig = go.Figure()
  for i, context in enumerate(CONTEXTS):
    t = step_times(context)
    fig.add_trace(go.Scatter(
        x=t['total'], y=t['throughput'], customdata=t['batch'],
        mode='lines+markers', name='Pareto frontier', meta='frontier',
        visible=i == INITIAL_STEP, showlegend=False,
        line={'color': theme['frontier'], 'width': 1.5},
        marker={
            'size': 6, 'color': [math.log2(b) for b in t['batch']],
            'colorscale': 'Viridis', 'cmin': 0,
            'cmax': math.log2(max(MAX_BATCH.values())),
            'showscale': True,
            'colorbar': {
                'title': {'text': 'Batch size', 'side': 'right',
                          'font': {'size': 12, 'color': theme['ink']}},
                'tickvals': [math.log2(b) for b in BATCH_COLORBAR_TICKS],
                'ticktext': [str(b) for b in BATCH_COLORBAR_TICKS],
                'tickfont': {'color': theme['muted']},
                'thickness': 12, 'len': 0.9, 'outlinewidth': 0, 'x': 1.02,
            },
        },
        hovertemplate=('B = %{customdata}<br>%{x:.3g}ms per token<br>'
                       '%{y:.3g} tokens/ms/chip<extra></extra>')))
    max_batch = len(t['batch'])
    labeled = [b for b in LABELED_BATCHES
               if b * MIN_LABEL_RATIO <= max_batch] + [max_batch]
    fig.add_trace(go.Scatter(
        x=[t['total'][b - 1] for b in labeled],
        y=[t['throughput'][b - 1] for b in labeled],
        text=[f'B={b}' for b in labeled],
        textposition=[label_position(t['total'], t['throughput'], b - 1)
                      for b in labeled],
        # Invisible markers keep the text offset from the point it labels.
        mode='markers+text', meta='labels', visible=i == INITIAL_STEP,
        showlegend=False, hoverinfo='skip', cliponaxis=False,
        marker={'size': 7, 'opacity': 0},
        textfont={'size': 11, 'color': theme['labels']}))

  title = 'Latency vs. throughput for LLaMA 3-70B on TPU v5e 4x4'
  layout = base_layout(title, slider_steps(2))
  x_ticks = [5, 7, 10, 15, 20, 30, 40]
  y_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
  layout['xaxis'] = {
      **layout['xaxis'], 'range': log_range(*PARETO_X), 'tickvals': x_ticks,
      'ticktext': [ms_label(v) for v in x_ticks],
      'title': {**layout['xaxis']['title'], 'text': 'Per-token latency'}}
  layout['yaxis'] = {
      **layout['yaxis'], 'range': log_range(*PARETO_Y), 'tickvals': y_ticks,
      'ticktext': [f'{v:g}' for v in y_ticks],
      'title': {**layout['yaxis']['title'],
                'text': 'Throughput (tokens/ms/chip)'}}
  layout['hovermode'] = 'closest'
  # No legend on this one, so the plot can start closer to the title; the
  # right margin holds the batch-size colorbar.
  layout['margin'] = {**layout['margin'], 't': 44, 'r': 72}
  layout['meta'] = {
      'wide': {'title.text': title},
      'narrow': {'title.text': 'Latency vs. throughput, LLaMA 3-70B'},
  }
  fig.update_layout(**layout)
  return fig


# (key, legend name, line width). The total goes last so it draws on top, but
# its legend entry comes first.
SERIES = [
    ('params', 'Param loading', 2),
    ('kv', 'KV loading', 2),
    ('flops', 'FLOPs', 2),
    ('total', 'Total', 3),
]


def build_latency_breakdown() -> go.Figure:
  """Per-step time and its components against batch size, per context."""
  theme = LIGHT_THEME
  fig = go.Figure()
  for i, context in enumerate(CONTEXTS):
    t = step_times(context)
    for key, name, width in SERIES:
      fig.add_trace(go.Scatter(
          x=t['batch'], y=t[key], name=name, meta=key, mode='lines',
          visible=i == INITIAL_STEP, legendgroup=key,
          legendrank=0 if key == 'total' else 1,
          line={'color': theme[key], 'width': width},
          customdata=[ms_label(v) for v in t[key]],
          hovertemplate='%{customdata}'))

  title = 'Decode step time for LLaMA 3-70B on TPU v5e 4x4'
  layout = base_layout(title, slider_steps(len(SERIES)))
  x_ticks = [2 ** k for k in range(10)]
  y_exps = [-2, -1, 0, 1]
  layout['xaxis'] = {
      **layout['xaxis'], 'range': log_range(1, 511), 'tickvals': x_ticks,
      'ticktext': [str(v) for v in x_ticks], 'hoverformat': ',d',
      'title': {**layout['xaxis']['title'], 'text': 'Batch size'}}
  # Pinned to the original figure's range, so the axis holds still.
  layout['yaxis'] = {
      **layout['yaxis'], 'range': [-2, math.log10(57)],
      'tickvals': [10.0 ** e for e in y_exps],
      'ticktext': [ms_label(10.0 ** e) for e in y_exps],
      'title': {**layout['yaxis']['title'], 'text': 'Time per decode step'}}
  layout['hovermode'] = 'x unified'
  # The pinned x-axis range clips the shade at the largest batch size.
  layout['shapes'] = [{
      'type': 'rect', 'xref': 'x', 'yref': 'paper',
      'x0': COMPUTE_BOUND_BATCH, 'x1': 511, 'y0': 0, 'y1': 1,
      'fillcolor': theme['shade'], 'line': {'width': 0}, 'layer': 'below',
  }]
  note = f'<b>Compute-bound</b><br>MLPs, B > {COMPUTE_BOUND_BATCH:.0f}'
  layout['annotations'] = [{
      'xref': 'x', 'x': math.log10(COMPUTE_BOUND_BATCH), 'yref': 'paper',
      'y': 0, 'xanchor': 'left', 'yanchor': 'bottom', 'xshift': 4,
      'yshift': 4, 'showarrow': False, 'align': 'left',
      'text': note,
      'font': {'size': 12, 'color': theme['accent']},
  }]
  # Phones get a shorter title, a two-row legend and a narrower note.
  layout['meta'] = {
      'wide': {'title.text': title, 'margin.t': 64,
               'annotations[0].text': note},
      'narrow': {'title.text': 'Decode step time, LLaMA 3-70B',
                 'margin.t': 84,
                 'annotations[0].text': '<b>Compute-<br>bound</b>'},
  }
  fig.update_layout(**layout)
  return fig


# Runs inside the iframe: sizes the plot to the frame (tightening it on
# phones) and follows the parent page's light/dark theme. Trace colors come
# from each trace's `meta`, which names its entry in the theme.
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

  function applyTheme() {
    var t = isDark() ? DARK : LIGHT;
    var colors = gd.data.map(function(d) { return t[d.meta]; });
    var layout = {
      'font.color': t.ink, 'legend.font.color': t.ink,
      'xaxis.color': t.muted, 'xaxis.linecolor': t.muted,
      'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.ink,
      'yaxis.color': t.muted, 'yaxis.linecolor': t.muted,
      'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.ink,
      'sliders[0].activebgcolor': t.accent, 'sliders[0].bgcolor': t.slider,
      'sliders[0].currentvalue.font.color': t.ink,
    };
    (gd.layout.shapes || []).forEach(function(_, i) {
      layout['shapes[' + i + '].fillcolor'] = t.shade;
    });
    (gd.layout.annotations || []).forEach(function(_, i) {
      layout['annotations[' + i + '].font.color'] = t.accent;
    });
    Plotly.update(gd, {
      'line.color': colors,
      'textfont.color': gd.data.map(function() { return t.labels; }),
    }, layout);
    // Only the Pareto dots carry a colorbar. Restyling colorbar attributes on
    // any other trace would give it a colorbar of its own.
    var dots = [];
    gd.data.forEach(function(d, i) { if (d.meta === 'frontier') dots.push(i); });
    if (dots.length) {
      Plotly.restyle(gd, {
        'marker.colorbar.title.font.color': t.ink,
        'marker.colorbar.tickfont.color': t.muted,
      }, dots);
    }
  }

  // Full-density x ticks; phones keep every other one.
  var xTicks = {vals: gd.layout.xaxis.tickvals.slice(),
                text: gd.layout.xaxis.ticktext.slice()};
  function keep(narrow) {
    return function(_, i) { return !narrow || i % 2 === 0; };
  }

  function fit() {
    var narrow = window.innerWidth < 520;
    Plotly.relayout(gd, Object.assign({
      width: window.innerWidth, height: window.innerHeight,
      'margin.l': narrow ? 52 : 64,
      'xaxis.tickvals': xTicks.vals.filter(keep(narrow)),
      'xaxis.ticktext': xTicks.text.filter(keep(narrow)),
      'title.font.size': narrow ? 13 : 15,
      'font.size': narrow ? 11 : 12,
      'legend.font.size': narrow ? 11 : 12,
    }, (gd.layout.meta || {})[narrow ? 'narrow' : 'wide']));
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


def write_page(fig: go.Figure, out: pathlib.Path) -> None:
  """Writes a figure as a standalone, frame-filling page.

  Args:
    fig: The figure to write.
    out: The HTML file to (over)write.
  """
  fig.write_html(
      out, include_plotlyjs='cdn', full_html=True,
      config={'displayModeBar': False}, post_script=POST_SCRIPT)
  # No page margin or scrollbars; the post script sizes the plot to the frame.
  html = out.read_text()
  html = html.replace(
      '<head>',
      '<head><meta name="viewport" content="width=device-width">'
      '<style>html,body{margin:0;overflow:hidden;background:transparent}'
      '</style>', 1)
  out.write_text(html)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--out-dir', default='assets/plotly', type=pathlib.Path)
  args = parser.parse_args()

  write_page(build_pareto(), args.out_dir / 'pareto.html')
  write_page(build_latency_breakdown(),
             args.out_dir / 'latency_breakdown_log.html')


if __name__ == '__main__':
  main()
