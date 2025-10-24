import numpy as np
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

# Configuration constants for plotting
BACKEND_CORRELATION_MINUTES = 24 * 60

COLORS = {
    "error": "#D65C5C",
    "anomaly": "#E8A86C",
    "warning": "#A9B0B8",
    "baseline": "#2A3242",
    "grid": "#323A4A",
    "component_neutral": "#8EC5E9",
}


def cycle_colors(keys: list) -> dict:
    """
    Creates a color map from a predefined palette for a list of keys.

    Args:
        keys: A list of keys to assign colors to.

    Returns:
        A dictionary mapping each unique key to a color hex string.
    """
    palette = [
        "#C96868", "#D1785E", "#DE8A6A", "#E69B74", "#EFAE82",
        "#F1B78E", "#F3C29B", "#F6CDA8", "#F8D8B6", "#FBE3C4",
    ]
    unique_keys = list(pd.Series(keys).dropna().unique())
    return {k: palette[i % len(palette)] for i, k in enumerate(unique_keys)}


def generate_color_map(ids_series: pd.Series) -> dict:
    """
    Generates a color map for a series of IDs using a matplotlib colormap.

    Args:
        ids_series: A pandas Series containing IDs.

    Returns:
        A dictionary mapping each unique ID to a color hex string.
    """
    unique_ids = sorted(pd.Series(ids_series).dropna().unique())
    cmap = plt.cm.get_cmap('tab10', len(unique_ids) or 1)
    color_map = {
        val: mcolors.to_hex(cmap(i), keep_alpha=False)
        for i, val in enumerate(unique_ids)
    }
    return color_map


def color_by_problem(problem_ids: pd.Series) -> dict:
    """
    Creates a color map for problem IDs using a cyclical color palette.
    This is an alias for cycle_colors.

    Args:
        problem_ids: A pandas Series of problem IDs.

    Returns:
        A dictionary mapping each unique problem ID to a color hex string.
    """
    return cycle_colors(problem_ids)


def plot_timeline_plotly(
    events_df: pd.DataFrame,
    res_df: pd.DataFrame,
    all_components: list = None,
    stem_height: float = 0.3,
    base_width: float = 2.2,
    stem_width: float = 2.5,
    link_width: float = 2.0
) -> go.Figure:
    """
    Generates an interactive event timeline using Plotly.

    This plot displays errors and warnings for different components over time,
    and draws curves to link correlated events.

    Args:
        events_df: DataFrame containing log events (errors and warnings).
        res_df: DataFrame containing correlation results from the analysis.
        all_components: An optional list of all components to display on the y-axis.
                        If not provided, it's inferred from the events data.
        stem_height: The vertical height of the stems for event markers.
        base_width: The width of the horizontal baseline for each component.
        stem_width: The width of the vertical stems for event markers.
        link_width: The width of the spline curves connecting correlated events.

    Returns:
        A Plotly graph objects Figure instance.
    """
    errs = events_df[events_df['level'] == 'ERROR'].copy()
    warns = events_df[events_df['level'] == 'WARNING'].copy()

    comps_present = list(events_df['component'].unique())
    comps = list(all_components) if all_components else comps_present

    for c in comps_present:
        if c not in comps:
            comps.append(c)
    comps = list(dict.fromkeys(comps))

    ymap = {c: i for i, c in enumerate(comps, start=1)}

    cmap_problems = color_by_problem(res_df['problem_id'])

    def pcolor(pid):
        return COLORS["warning"] if pd.isna(pid) else cmap_problems.get(pid, COLORS["error"])

    fig = go.Figure()

    if not events_df.empty:
        tmin, tmax = events_df['timestamp'].min(), events_df['timestamp'].max()
    else:
        now = pd.Timestamp.now()
        tmin, tmax = now, now + pd.Timedelta(hours=1)

    for c in comps:
        yy = ymap[c]
        fig.add_trace(go.Scatter(
            x=[tmin, tmax], y=[yy, yy],
            mode='lines',
            line=dict(color=COLORS["baseline"], width=base_width),
            hoverinfo='skip', showlegend=False,
            name=f'{c}_baseline'
        ))

    unique_errors_df = res_df.drop_duplicates(subset=['problem_file', 'problem_line'])
    pid_by_errkey = {
        (r['problem_file'], int(r['problem_line'])): r['problem_id']
        for _, r in unique_errors_df.iterrows()
    }

    for _, r in errs.iterrows():
        yy = ymap.get(r['component'])
        if yy is None:
            continue
        x = r['timestamp']
        pid = pid_by_errkey.get((r['file'], int(r['line'])), np.nan)
        col = pcolor(pid)

        fig.add_trace(go.Scatter(
            x=[x, x], y=[yy, yy + stem_height],
            mode='lines', line=dict(color=col, width=stem_width),
            hoverinfo='skip', showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[x], y=[yy + stem_height],
            mode='markers',
            marker=dict(size=11, symbol='square', color=col, line=dict(width=1, color='#1B2330')),
            name='ERROR',
            showlegend=False,
            hovertemplate=(
                "<b>ERROR</b><br>"
                f"component: {r['component']}<br>"
                f"file: {r['file']}:{r['line']}<br>"
                f"msg: {r['message']}<extra></extra>"
            )
        ))

    for _, r in warns.iterrows():
        yy = ymap.get(r['component'])
        if yy is None:
            continue
        x = r['timestamp']
        col = COLORS["warning"]
        fig.add_trace(go.Scatter(
            x=[x, x], y=[yy - stem_height, yy],
            mode='lines', line=dict(color=col, width=stem_width),
            hoverinfo='skip', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[yy - stem_height],
            mode='markers',
            marker=dict(size=10, symbol='circle', color=col, line=dict(width=1, color='#1B2330')),
            name='WARNING',
            showlegend=False,
            hovertemplate=(
                "<b>WARNING</b><br>"
                f"component: {r['component']}<br>"
                f"file: {r['file']}:{r['line']}<br>"
                f"msg: {r['message']}<extra></extra>"
            )
        ))

    errs_index = (errs.set_index(['file', 'line'])
                  if not errs.empty
                  else pd.DataFrame(columns=['timestamp', 'component']).set_index(['file', 'line']))

    link_win = timedelta(minutes=BACKEND_CORRELATION_MINUTES)
    for error_key, group in res_df.groupby(['problem_file', 'problem_line']):
        if error_key not in errs_index.index:
            continue
        e = errs_index.loc[error_key]
        if isinstance(e, pd.DataFrame):
            e = e.iloc[0]

        x1 = e['timestamp']
        y1 = ymap.get(e['component'])
        if y1 is None:
            continue

        time_lower_bound = x1 - link_win
        time_upper_bound = x1 + link_win
        cand = warns[(warns['timestamp'] >= time_lower_bound) &
                     (warns['timestamp'] <= time_upper_bound)].copy()

        if cand.empty:
            continue

        for _, correlation in group.iterrows():
            if cand.empty:
                break
            best_idx = (cand['timestamp'] - x1).abs().idxmin()
            w = cand.loc[best_idx]
            x2, y2 = w['timestamp'], ymap.get(w['component'])
            if y2 is None:
                cand.drop(best_idx, inplace=True)
                continue

            col = cmap_problems.get(correlation['problem_id'], COLORS["anomaly"])

            x_curve = [x1, x1, x2, x2]
            y_curve = [y1 + stem_height, y1 + stem_height + 0.35, y2 - stem_height - 0.35, y2 - stem_height]
            fig.add_trace(go.Scatter(
                x=x_curve, y=y_curve, mode='lines',
                line=dict(color=col, width=link_width),
                line_shape='spline', opacity=0.9,
                hoverinfo='skip', showlegend=False
            ))
            cand.drop(best_idx, inplace=True)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13),
        xaxis=dict(
            showgrid=True, gridcolor=COLORS["grid"],
            zeroline=False
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[ymap[c] for c in comps],
            ticktext=comps,
            range=[0, len(comps) + 1],
            showgrid=False
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode='x unified'
    )
    return fig


def plot_sankey_diagram(
    res_df: pd.DataFrame,
    component_id_to_text: dict = None
) -> go.Figure:
    """
    Generates a Sankey diagram to visualize flows between anomalies, problems, and components.

    Args:
        res_df: DataFrame containing the correlation results.
        component_id_to_text: Optional dictionary to map component IDs to display names.

    Returns:
        A Plotly graph objects Figure instance representing the Sankey diagram.
    """
    if res_df.empty:
        return go.Figure()

    def _hex_to_rgb01(h: str):
        r, g, b = mcolors.to_rgb(h)
        return r, g, b

    def _rgb01_to_rgba_str(rgb, alpha=1.0):
        r, g, b = rgb
        return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"

    def _lerp_rgb(rgb1, rgb2, t: float):
        return tuple(rgb1[i] + (rgb2[i] - rgb1[i]) * t for i in range(3))

    def _generate_shades(base_hex: str, n: int, lighten_to="#ffffff", darken_to="#000000"):
        if n <= 0:
            return []
        base = _hex_to_rgb01(base_hex)
        light = _hex_to_rgb01(lighten_to)
        dark = _hex_to_rgb01(darken_to)

        up = max(1, n // 2 + n % 2)
        dn = max(0, n - up)

        up_list = [_lerp_rgb(base, light, t) for t in np.linspace(0.15, 0.45, up)]
        dn_list = [_lerp_rgb(base, dark, t) for t in np.linspace(0.10, 0.30, dn)]
        shades = up_list + dn_list

        if n == 1:
            shades = [_lerp_rgb(base, light, 0.25)]
        return [mcolors.to_hex(c) for c in shades[:n]]

    def _cycle_or_expand(keys, base_palette_hex: list[str]):
        keys = list(pd.Series(keys).dropna().unique())
        if not keys:
            return {}
        colors = base_palette_hex.copy()

        while len(colors) < len(keys):
            seed = colors[-1] if colors else COLORS["component_neutral"]
            extra = _generate_shades(seed, min(len(keys) - len(colors), 6))
            for c in extra:
                if c not in colors:
                    colors.append(c)
                if len(colors) >= len(keys):
                    break
            if len(colors) < len(keys):
                tab20 = [mcolors.to_hex(x) for x in plt.cm.tab20.colors]
                for c in tab20:
                    if c not in colors:
                        colors.append(c)
                    if len(colors) >= len(keys):
                        break

        return {k: colors[i] for i, k in enumerate(keys)}

    unique_anomalies_texts = res_df['anomaly_text'].astype(str).unique()
    unique_problems_texts = res_df['problem_text'].astype(str).unique()
    unique_components = res_df['component'].astype(str).unique()

    node_labels = []

    anomaly_nodes = {a_text: i for i, a_text in enumerate(unique_anomalies_texts)}
    node_labels.extend(unique_anomalies_texts)

    problem_nodes = {p_text: i + len(unique_anomalies_texts) for i, p_text in enumerate(unique_problems_texts)}
    node_labels.extend(unique_problems_texts)

    component_nodes = {f"component_{c}": i + len(unique_anomalies_texts) + len(unique_problems_texts) for i, c in enumerate(unique_components)}
    node_labels.extend([
        (component_id_to_text.get(c, f"Component {c}") if component_id_to_text else f"Component {c}")
        for c in unique_components
    ])

    anomaly_palette = _generate_shades(COLORS["anomaly"], len(unique_anomalies_texts))
    anomaly_colors_map = _cycle_or_expand(unique_anomalies_texts, anomaly_palette)

    base_problem_map = cycle_colors(unique_problems_texts)
    base_problem_palette = [base_problem_map[p] for p in unique_problems_texts] if len(base_problem_map) == len(unique_problems_texts) else list(base_problem_map.values())
    if len(base_problem_palette) < len(unique_problems_texts):
        extra = _generate_shades(COLORS["error"], len(unique_problems_texts) - len(base_problem_palette))
        base_problem_palette += extra
    problem_colors_map = _cycle_or_expand(unique_problems_texts, base_problem_palette)

    component_palette = _generate_shades(COLORS["component_neutral"], len(unique_components), lighten_to="#f7fbff")
    component_colors_map = _cycle_or_expand(unique_components, component_palette)

    node_colors_list = []
    for a_text in unique_anomalies_texts:
        node_colors_list.append(anomaly_colors_map[a_text])
    for p_text in unique_problems_texts:
        node_colors_list.append(problem_colors_map[p_text])
    for c in unique_components:
        node_colors_list.append(component_colors_map[c])

    source, target, value, link_colors = [], [], [], []

    for (a_text, p_text), group in res_df.groupby(['anomaly_text', 'problem_text']):
        source.append(anomaly_nodes[a_text])
        target.append(problem_nodes[p_text])
        value.append(len(group))

        a_hex = anomaly_colors_map.get(a_text, COLORS["anomaly"])
        link_colors.append(_rgb01_to_rgba_str(_hex_to_rgb01(a_hex), 0.65))

    for (p_text, comp), group in res_df.groupby(['problem_text', 'component']):
        p_text, comp = str(p_text), str(comp)
        source.append(problem_nodes[p_text])
        target.append(component_nodes[f"component_{comp}"])
        value.append(len(group))

        p_hex = problem_colors_map.get(p_text, COLORS["error"])
        link_colors.append(_rgb01_to_rgba_str(_hex_to_rgb01(p_hex), 0.55))

    if not source:
        return go.Figure()

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=16, thickness=18,
            line=dict(color="rgba(255,255,255,0.08)", width=1),
            label=node_labels,
            color=node_colors_list
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
        )
    )])

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13),
        title_text="Flows: Anomalies → Problems → Components",
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig


def plot_timeline_bar_chart(events_df: pd.DataFrame, time_grain: str = "h") -> go.Figure:
    """
    Generates a stacked bar chart showing the frequency of events over time.

    Args:
        events_df: DataFrame containing log events (must include 'timestamp' and 'level').
        time_grain: The time granularity for resampling (e.g., 'h' for hourly, 'D' for daily).

    Returns:
        A Plotly graph objects Figure instance.
    """
    if events_df.empty:
        return go.Figure()

    df = events_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    anomaly = df[df['level'] == 'WARNING']
    errors = df[df['level'] == 'ERROR']

    idx = df.set_index('timestamp').resample(time_grain).size().index

    anomaly_counts = (anomaly.set_index('timestamp')
                      .resample(time_grain).size()
                      .reindex(idx, fill_value=0)
                      .reset_index(name='count'))
    error_counts = (errors.set_index('timestamp')
                    .resample(time_grain).size()
                    .reindex(idx, fill_value=0)
                    .reset_index(name='count'))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=anomaly_counts['timestamp'],
        y=anomaly_counts['count'],
        name='Anomalies',
        marker_color=COLORS["anomaly"],
        opacity=0.80
    ))
    fig.add_trace(go.Bar(
        x=error_counts['timestamp'],
        y=error_counts['count'],
        name='Errors',
        marker_color=COLORS["error"],
        opacity=0.80
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='stack',
        title_text="Event Frequency Over Time",
        xaxis_title="Time",
        yaxis_title="Count",
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"]),
        yaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
