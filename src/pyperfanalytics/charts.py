import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyperfanalytics.drawdowns import drawdowns


def chart_cum_returns(
    R: pd.Series | pd.DataFrame,
    wealth_index: bool = False,
    geometric: bool = True,
    begin: str = "axis",
    title: str = "Cumulative Returns",
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Create a cumulative returns chart using Plotly.

    Matches R's `chart.CumReturns` logic.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets.
    wealth_index : bool, default False
        If True, the series starts at $1. If False, starts at $0.
    geometric : bool, default True
        If True, uses geometric compounding (1+R). If False, uses arithmetic (sum).
    begin : str, default "axis"
        Not fully implemented. In R, determines if the chart starts at the first value or axis.
    title : str, default "Cumulative Returns"
        Chart title.
    colorset : list[str], optional
        List of colors for the traces.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object.
    r"""
    if isinstance(R, pd.Series):
        R = R.to_frame()

    fig = go.Figure()

    # Calculate cumulative returns
    # R implementation:
    # if wealth_index: starts at 1
    # else: starts at 0
    one = 0 if wealth_index else 1

    for i, col in enumerate(R.columns):
        series = R[col].dropna()
        if series.empty:
            continue

        if geometric:
            cum_ret = (1 + series).cumprod() - one
        else:
            cum_ret = (1 - one) + series.cumsum()

        # Add a starting point of (1-one) if not already there
        # R's chart.CumReturns often adds a starting point at the axis
        series.index[0]
        # To make it look like R, we might want to prepend a 0/1 at the period before
        # but for now let's keep it simple.

        color = colorset[i % len(colorset)] if colorset else None

        fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret.values, mode="lines", name=col, line=dict(color=color)))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return" if not wealth_index else "Value of $1",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )

    return fig


def chart_bar_returns(
    R: pd.Series | pd.DataFrame, title: str = "Returns", colorset: list[str] | None = None, **kwargs
) -> go.Figure:
    r"""
    Create a bar chart of period returns.

    Matches R's `chart.Bar` logic. Uses 'overlay' mode to maintain bar thickness.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets.
    title : str, default "Returns"
        Chart title.
    colorset : list[str], optional
        List of colors for the traces.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object.
    r"""
    if isinstance(R, pd.Series):
        R = R.to_frame()

    fig = go.Figure()

    for i, col in enumerate(R.columns):
        series = R[col].dropna()
        color = colorset[i % len(colorset)] if colorset else None

        fig.add_trace(
            go.Bar(
                x=series.index,
                y=series.values,
                name=col,
                marker_color=color,
                marker_line_width=0.5,  # Add a very thin line to define edges
                marker_line_color=color,
                opacity=0.7,  # Transparency allows seeing overlapping bars
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Return",
        barmode="overlay",  # Change from 'group' to 'overlay' to keep bars thick
        template="plotly_white",
    )

    return fig


def chart_drawdown(
    R: pd.Series | pd.DataFrame,
    geometric: bool = True,
    title: str = "Drawdown (Underwater)",
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Create a professional drawdown (underwater) chart showing losses from peak.

    Matches R's `chart.Drawdown` logic with enhanced interactive styling.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets.
    geometric : bool, default True
        If True, uses geometric compounding for drawdown calculation.
    title : str, default "Drawdown (Underwater)"
        Chart title.
    colorset : list[str], optional
        List of colors for the traces. If None, uses a professional semantic palette.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with semi-transparent area fill.
    r"""
    dd = drawdowns(R, geometric=geometric)
    if isinstance(dd, pd.Series):
        dd = dd.to_frame()

    fig = go.Figure()

    # Standard Project Palette (Plotly defaults)
    # Using semi-transparent RGBA for "lake" overlaps to maintain professional feel
    default_colors_hex = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

    def hex_to_rgba(hex_val, alpha):
        hex_val = hex_val.lstrip("#")
        lv = len(hex_val)
        rgb = tuple(int(hex_val[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"

    default_colors = [hex_to_rgba(c, 0.7) for c in default_colors_hex]
    line_colors = default_colors_hex

    colors = colorset if colorset else default_colors

    for i, col in enumerate(dd.columns):
        series = dd[col].dropna()
        color = colors[i % len(colors)]
        line_color = line_colors[i % len(line_colors)] if not colorset else color

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=col,
                fill="tozeroy",
                line=dict(color=line_color, width=1.5),
                fillcolor=color,
                hovertemplate=f"<b>{col}</b><br>Date: %{{x}}<br>Drawdown: %{{y:.2%}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20, color="#2C2B2A")),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=40, t=80, b=60),
        yaxis=dict(
            tickformat=".1%",
            zeroline=True,
            zerolinecolor="#2C2B2A",  # High-Water Mark line (Black/Dark Gray)
            zerolinewidth=2,
            gridcolor="#ECEBEA",  # Subtle light gray grid
            showgrid=True,
            ticksuffix=" ",
            range=[min(dd.min().min() * 1.1, -0.05), 0.01],  # Ensure space for labels and peaks
        ),
        xaxis=dict(gridcolor="#ECEBEA", showgrid=True, linecolor="#2C2B2A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255, 255, 255, 0)"),
    )

    return fig


def charts_performance_summary(
    R: pd.Series | pd.DataFrame,
    geometric: bool = True,
    wealth_index: bool = False,
    main: str | None = None,
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Create a combined dashboard containing Cumulative Returns, Period Returns, and Drawdown.

    Matches R's `charts.PerformanceSummary` logic.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets.
    geometric : bool, default True
        If True, uses geometric compounding.
    wealth_index : bool, default False
        If True, cumulative returns start at $1. If False, start at $0.
    main : str, optional
        Title for the dashboard.
    colorset : list[str], optional
        List of colors for the traces.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with 3 subplots.
    r"""
    if isinstance(R, pd.Series):
        R = R.to_frame()

    if main is None:
        main = f"{R.columns[0]} Performance"

    # Create subplots: 3 rows, 1 column
    # Shared x-axis
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Cumulative Return", "Returns", "Drawdown"),
        row_heights=[0.5, 0.25, 0.25],
    )

    # Cumulative Returns
    one = 0 if wealth_index else 1
    for i, col in enumerate(R.columns):
        series = R[col].dropna()
        if series.empty:
            continue

        if geometric:
            cum_ret = (1 + series).cumprod() - one
        else:
            cum_ret = (1 - one) + series.cumsum()

        color = colorset[i % len(colorset)] if colorset else None

        fig.add_trace(
            go.Scatter(x=cum_ret.index, y=cum_ret.values, name=col, legendgroup=col, line=dict(color=color)),
            row=1,
            col=1,
        )

    # Bar Returns
    for i, col in enumerate(R.columns):
        series = R[col].dropna()
        color = colorset[i % len(colorset)] if colorset else None
        fig.add_trace(
            go.Bar(x=series.index, y=series.values, name=col, legendgroup=col, showlegend=False, marker_color=color),
            row=2,
            col=1,
        )

    # Drawdown
    dd = drawdowns(R, geometric=geometric)
    if isinstance(dd, pd.Series):
        dd = dd.to_frame()
    for i, col in enumerate(dd.columns):
        series = dd[col].dropna()
        color = colorset[i % len(colorset)] if colorset else None
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=col,
                legendgroup=col,
                showlegend=False,
                fill="tozeroy",
                line=dict(color=color),
            ),
            row=3,
            col=1,
        )

    fig.update_layout(height=800, title_text=main, template="plotly_white", hovermode="x unified")

    # Update y-axis titles
    fig.update_yaxes(title_text="Cum. Return", row=1, col=1)
    fig.update_yaxes(title_text="Return", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown", row=3, col=1)

    return fig


def chart_bar_var(
    R: pd.Series | pd.DataFrame,
    width: int = 0,
    gap: int = 12,
    methods: str | list[str] | None = None,
    p: float = 0.95,
    main: str | None = None,
    colorset: list[str] | None = None,
    show_symmetric: bool = False,
    show_horizontal: bool = False,
    **kwargs,
) -> go.Figure:
    r"""
    Plot periodic returns as a bar chart with interactive risk metric overlays.

    Matches R's `chart.BarVaR` logic.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets.
    width : int, default 0
        Window width for rolling calculation. If 0, an expanding window is used.
    gap : int, default 12
        Minimum number of periods before the first risk value is calculated.
    methods : str or list[str], optional
        List of risk metrics (e.g., ["ModifiedVaR", "GaussianVaR"]).
    p : float, default 0.95
        Confidence level for risk calculation.
    main : str, optional
        Chart title.
    colorset : list[str], optional
        List of colors for the traces.
    show_symmetric : bool, default False
        If True, plots the upper-tail threshold (symmetric around zero).
    show_horizontal : bool, default False
        If True, adds a horizontal line at the latest risk value.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with bar returns and risk metric line overlays.
    r"""
    if methods is None:
        methods = ["ModifiedVaR", "GaussianVaR"]

    from pyperfanalytics.risk import es_gaussian, es_historical, es_modified, var_gaussian, var_historical, var_modified

    if isinstance(R, pd.Series):
        R = R.to_frame()

    if isinstance(methods, str):
        methods = [methods]

    fig = go.Figure()

    # 1. Bar Chart (Returns)
    col = R.columns[0]
    series = R[col].dropna()

    fig.add_trace(
        go.Bar(
            x=series.index,
            y=series.values,
            name="Return",
            marker_color="#D3D3D3",  # Professional light gray for bars
            marker_line_width=0,
            opacity=0.7,
            hovertemplate="Date: %{x}<br>Return: %{y:.4f}<extra></extra>",
        )
    )

    # 2. Risk Mapping
    risk_mapping = {
        "ModifiedVaR": var_modified,
        "GaussianVaR": var_gaussian,
        "HistoricalVaR": var_historical,
        "ModifiedES": es_modified,
        "GaussianES": es_gaussian,
        "HistoricalES": es_historical,
        "StdDev": lambda x, **kwargs: x.std(ddof=1),
    }

    # Coordinated professional palette
    risk_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

    # 3. Calculate and Add Risk Lines
    for i, method in enumerate(methods):
        if method == "none":
            continue

        func = risk_mapping.get(method)
        if not func:
            continue

        risk_vals = []
        for t in range(1, len(series) + 1):
            if width == 0:
                if t < gap:
                    risk_vals.append(np.nan)
                else:
                    window = series.iloc[:t]
                    risk_vals.append(-abs(func(window, p=p)))
            else:
                if t < width:
                    risk_vals.append(np.nan)
                else:
                    window = series.iloc[t - width : t]
                    risk_vals.append(-abs(func(window, p=p)))

        risk_series = pd.Series(risk_vals, index=series.index)
        color = colorset[i % len(colorset)] if colorset else risk_colors[i % len(risk_colors)]

        # Latest value for horizontal line
        latest_val = risk_series.dropna().iloc[-1] if not risk_series.dropna().empty else None

        fig.add_trace(
            go.Scatter(
                x=risk_series.index,
                y=risk_series.values,
                mode="lines",
                name=method,
                line=dict(color=color, width=2.5),
                hovertemplate=f"{method}: %{{y:.4f}}<extra></extra>",
            )
        )

        if show_horizontal and latest_val is not None:
            fig.add_hline(
                y=latest_val,
                line_dash="dot",
                line_color=color,
                line_width=1,
                annotation_text=f"Latest {method}",
                annotation_position="bottom left",
            )

        if show_symmetric:
            fig.add_trace(
                go.Scatter(
                    x=risk_series.index,
                    y=-risk_series.values,
                    mode="lines",
                    name=f"{method} (Upper)",
                    line=dict(color=color, dash="dash", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=main or f"{col} Returns with Risk Thresholds",
        xaxis_title="Date",
        yaxis_title="Return / Risk",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def chart_rolling_performance(
    R: pd.Series | pd.DataFrame,
    width: int = 12,
    FUN: str = "return_annualized",
    main: str | None = None,
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Wrapper to chart any performance metric over a rolling window.

    Matches R's `chart.RollingPerformance` logic.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets.
    width : int, default 12
        Rolling window size.
    FUN : str, default "return_annualized"
        Name of the function to apply over the rolling window.
    main : str, optional
        Chart title.
    colorset : list[str], optional
        List of colors for the traces.
    **kwargs
        Additional arguments passed to the `FUN` function or Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with time-series line chart.
    r"""
    import pyperfanalytics as pa

    if isinstance(R, pd.Series):
        R = R.to_frame()

    # Map string function name to actual function
    func = getattr(pa, FUN, None)
    if func is None:
        raise ValueError(f"Function {FUN} not found in pyperfanalytics.")

    results = pd.DataFrame(index=R.index)

    for col in R.columns:
        series = R[col]
        # Custom rolling apply because some functions need the whole series or extra args
        rolling_vals = []
        for i in range(len(series)):
            if i < width - 1:
                rolling_vals.append(np.nan)
            else:
                window = series.iloc[i - width + 1 : i + 1]
                # Pass kwargs (like scale, geometric, etc.)
                val = func(window, **kwargs)
                if hasattr(val, "iloc"):
                    val = val.iloc[0]
                rolling_vals.append(val)
        results[col] = rolling_vals

    fig = go.Figure()
    for i, col in enumerate(results.columns):
        color = colorset[i % len(colorset)] if colorset else None
        fig.add_trace(go.Scatter(x=results.index, y=results[col], mode="lines", name=col, line=dict(color=color)))

    fig.update_layout(
        title=main or f"Rolling {width}-period {FUN}", xaxis_title="Date", yaxis_title=FUN, template="plotly_white"
    )

    return fig


def chart_var_sensitivity(
    R: pd.Series | pd.DataFrame,
    methods: list[str] | None = None,
    main: str | None = None,
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Create a chart of VaR and ES estimates across a range of confidence levels.

    Matches R's `chart.VaRSensitivity` logic. Shows how risk estimates change
    between 89% and 99% confidence levels.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets. Only the first column is used if a DataFrame is passed.
    methods : list[str], optional
        List of risk metrics to plot (e.g., ["GaussianVaR", "ModifiedVaR"]).
    main : str, optional
        Chart title.
    colorset : list[str], optional
        List of colors for the traces.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object showing risk metrics vs. confidence level.
    r"""
    if methods is None:
        methods = ["GaussianVaR", "ModifiedVaR", "HistoricalVaR", "GaussianES", "ModifiedES", "HistoricalES"]

    from pyperfanalytics.risk import es_gaussian, es_historical, es_modified, var_gaussian, var_historical, var_modified

    if isinstance(R, pd.DataFrame):
        series = R.iloc[:, 0].dropna()
    else:
        series = R.dropna()

    # Standard ascending sequence for X-axis
    p_seq = np.linspace(0.89, 0.99, 21)

    risk_mapping = {
        "GaussianVaR": var_gaussian,
        "ModifiedVaR": var_modified,
        "HistoricalVaR": var_historical,
        "GaussianES": es_gaussian,
        "ModifiedES": es_modified,
        "HistoricalES": es_historical,
    }

    fig = go.Figure()

    # Professional coordinated palette (Plotly default sequence is well-balanced)
    default_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

    for i, method in enumerate(methods):
        func = risk_mapping.get(method)
        if not func:
            continue

        vals = []
        for p in p_seq:
            val = func(series, p=p)
            if isinstance(val, (pd.Series, pd.DataFrame)):
                val = val.iloc[0]
            vals.append(-abs(val))

        color = colorset[i % len(colorset)] if colorset else default_colors[i % len(default_colors)]
        fig.add_trace(
            go.Scatter(
                x=p_seq,
                y=vals,
                mode="lines+markers",
                name=method,
                line=dict(color=color, width=2.5),
                marker=dict(size=5, opacity=0.8),
                opacity=0.9,
            )
        )

    fig.update_layout(
        title=main or "Risk Confidence Sensitivity",
        xaxis_title="Confidence Level",
        yaxis_title="Value at Risk / Expected Shortfall (Return)",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def chart_histogram(
    R: pd.Series | pd.DataFrame,
    breaks: int = 30,
    methods: list[str] | None = None,
    p: float = 0.95,
    main: str | None = None,
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Create a histogram of returns with optional curve fits and risk markers.

    Matches R's `chart.Histogram` logic.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets. Only the first column is used if a DataFrame is passed.
    breaks : int, default 30
        Number of bins for the histogram.
    methods : list[str], optional
        Overlay types to add: "add.density", "add.normal", "add.risk".
    p : float, default 0.95
        Confidence level for risk markers.
    main : str, optional
        Chart title.
    colorset : list[str], optional
        List of colors for the histogram and overlays.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with histogram and optional line/marker overlays.
    r"""
    if methods is None:
        methods = ["add.density", "add.normal", "add.risk"]

    from scipy.stats import gaussian_kde, norm

    from pyperfanalytics.risk import var_historical, var_modified

    if isinstance(R, pd.DataFrame):
        series = R.iloc[:, 0].dropna()
    else:
        series = R.dropna()

    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=series,
            nbinsx=breaks,
            name="Frequency",
            histnorm="probability density" if any(m in methods for m in ["add.density", "add.normal"]) else "",
            marker_color="lightgray" if colorset is None else colorset[0],
            opacity=0.7,
        )
    )

    x_range = np.linspace(series.min(), series.max(), 200)

    if "add.normal" in methods:
        mu, std = series.mean(), series.std(ddof=1)
        p_normal = norm.pdf(x_range, mu, std)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=p_normal,
                mode="lines",
                name="Normal Dist",
                line=dict(color=colorset[1] if colorset and len(colorset) > 1 else "blue"),
            )
        )

    if "add.density" in methods:
        kde = gaussian_kde(series)
        p_kde = kde(x_range)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=p_kde,
                mode="lines",
                name="KDE",
                line=dict(color=colorset[2] if colorset and len(colorset) > 2 else "green"),
            )
        )

    if "add.risk" in methods:
        # Calculate VaR
        mvar = var_modified(series, p=p)
        hvar = var_historical(series, p=p)

        # We plot them as vertical lines
        # var_modified usually returns positive loss, but on histogram we want the return value
        # In PA, invert=TRUE returns the quantile value (negative)
        # Our risk functions return positive loss. So -abs(val) is the location on x-axis.
        for _i, (val, label, color) in enumerate(
            [(-abs(mvar), f"{p * 100}% ModVaR", "red"), (-abs(hvar), f"{p * 100}% HistVaR", "darkred")]
        ):
            fig.add_vline(
                x=val,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="top left",
                annotation_textangle=-90,
                annotation_yshift=10,
            )  # Small shift up to clear the axis line if needed

    fig.update_layout(
        title=main or f"Histogram of {series.name if hasattr(series, 'name') else 'Returns'}",
        xaxis_title="Returns",
        yaxis_title="Density" if any(m in methods for m in ["add.density", "add.normal"]) else "Frequency",
        template="plotly_white",
    )

    return fig


def chart_boxplot(
    R: pd.Series | pd.DataFrame,
    sort_by: str | None = None,
    sort_ascending: bool = True,
    main: str = "Return Distribution Comparison",
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Create a horizontal box and whiskers plot to compare return distributions.

    Matches R's `chart.Boxplot` logic.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets.
    sort_by : str, optional
        Sort assets by "mean", "median", or "variance".
    sort_ascending : bool, default True
        If True, sorts in ascending order.
    main : str, default "Return Distribution Comparison"
        Chart title.
    colorset : list[str], optional
        List of colors for the boxes.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with horizontal box plots and outlier markers.
    r"""
    if isinstance(R, pd.Series):
        R = R.to_frame()

    # Handle sorting
    if sort_by == "mean":
        cols = R.mean().sort_values(ascending=sort_ascending).index
    elif sort_by == "median":
        cols = R.median().sort_values(ascending=sort_ascending).index
    elif sort_by == "variance":
        cols = R.var().sort_values(ascending=sort_ascending).index
    else:
        # Default: reverse order to match top-down visual flow in horizontal plot
        cols = R.columns[::-1]

    fig = go.Figure()

    # Professional uniform color or muted palette
    fill_color = colorset[0] if colorset else "rgba(99, 110, 250, 0.5)"
    line_color = colorset[1] if colorset and len(colorset) > 1 else "rgb(99, 110, 250)"

    for col in cols:
        fig.add_trace(
            go.Box(
                x=R[col],  # x for horizontal
                name=col,
                marker=dict(
                    color=line_color,
                    size=4,
                    outliercolor="rgba(239, 85, 59, 0.6)",
                    line=dict(width=1, color="rgba(239, 85, 59, 0.6)"),
                ),
                fillcolor=fill_color,
                line=dict(width=1.5),
                boxpoints="outliers",  # Only show outliers to keep it clean
                orientation="h",
            )
        )

    fig.update_layout(
        title=main,
        xaxis_title="Returns",
        yaxis_title="Assets",
        template="plotly_white",
        showlegend=False,
        height=max(400, 40 * len(cols)),  # Dynamic height based on number of assets
        margin=dict(l=150),  # Extra space for long asset names
        xaxis=dict(zeroline=True, zerolinecolor="black", gridcolor="lightgray"),
    )

    return fig


def chart_qqplot(R: pd.Series | pd.DataFrame, main: str | None = None, **kwargs) -> go.Figure:
    r"""
    Create a Quantile-Quantile plot with a normal reference and confidence bands.

    Matches R's `chart.QQPlot` logic.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets. Only the first column is used if a DataFrame is passed.
    main : str, optional
        Chart title.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with scatter plot, reference line, and shaded confidence area.
    r"""
    from scipy import stats

    if isinstance(R, pd.DataFrame):
        series = R.iloc[:, 0].dropna()
    else:
        series = R.dropna()

    # 1. Calculate theoretical quantiles
    (osm, osr), (slope, intercept, r) = stats.probplot(series, dist="norm")

    fig = go.Figure()

    # 2. Add Confidence Bands (approximate 95%)
    # Standard error of quantiles
    n = len(series)
    x_range = np.linspace(osm.min(), osm.max(), 100)
    y_line = slope * x_range + intercept

    # Simple approximation for confidence bands
    # Based on the formula: SE(q) = (1/f(q)) * sqrt(p(1-p)/n)
    # For a normal distribution, f(q) is the PDF.
    p = stats.norm.cdf(x_range)
    se = (1.0 / stats.norm.pdf(x_range)) * np.sqrt(p * (1 - p) / n) * slope

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_range, x_range[::-1]]),
            y=np.concatenate([y_line + 1.96 * se, (y_line - 1.96 * se)[::-1]]),
            fill="toself",
            fillcolor="rgba(239, 85, 59, 0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% CI",
            showlegend=True,
        )
    )

    # 3. Reference Line
    fig.add_trace(
        go.Scatter(
            x=x_range, y=y_line, mode="lines", name="Normal Reference", line=dict(color="#EF553B", width=2, dash="dash")
        )
    )

    # 4. Sample Points
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            name="Sample Quantiles",
            marker=dict(size=7, color="#636EFA", opacity=0.6, line=dict(width=0.5, color="white")),
        )
    )

    fig.update_layout(
        title=main or f"Normal Q-Q Plot: {series.name if hasattr(series, 'name') else 'Returns'}",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        template="plotly_white",
        width=800,
        height=800,  # Square aspect ratio is standard for QQ
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def chart_correlation(
    R: pd.Series | pd.DataFrame, main: str = "Correlation Matrix", method: str = "pearson", **kwargs
) -> go.Figure:
    r"""
    Visualization of a Correlation Matrix with distributions and scatter plots.

    Matches R's `chart.Correlation` logic.
    - Diagonal: Histogram + Density.
    - Lower Triangle: Scatter plots with linear regression lines.
    - Upper Triangle: Correlation coefficients with significance stars.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets.
    main : str, default "Correlation Matrix"
        Chart title.
    method : str, default "pearson"
        Correlation method (passed to `pearsonr` or similar).
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with N x N subplots.
    r"""
    import numpy as np
    from scipy.stats import gaussian_kde, pearsonr

    if isinstance(R, pd.Series):
        R = R.to_frame()

    R = R.dropna()
    cols = R.columns
    n = len(cols)

    # Create N x N subplots
    fig = make_subplots(
        rows=n,
        cols=n,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.02,
        horizontal_spacing=0.02,
        column_titles=cols.tolist() if n < 10 else None,
        row_titles=cols.tolist() if n < 10 else None,
    )

    for i in range(n):  # row
        for j in range(n):  # col
            row_idx = i + 1
            col_idx = j + 1

            x_data = R.iloc[:, j]
            y_data = R.iloc[:, i]

            if i == j:
                # --- Diagonal: Histogram + KDE ---
                fig.add_trace(
                    go.Histogram(x=x_data, name=cols[j], showlegend=False, marker_color="lightgray", nbinsx=20),
                    row=row_idx,
                    col=col_idx,
                )
                # Add KDE (optional, but R has it)
                gaussian_kde(x_data)
                np.linspace(x_data.min(), x_data.max(), 100)
                # Scale KDE to histogram height approximately or use secondary y-axis
                # For simplicity, we just add the hist.

            elif i > j:
                # --- Lower Triangle: Scatter + Trendline ---
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode="markers",
                        marker=dict(size=4, color="blue", opacity=0.5),
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                # Add linear trendline
                m, b = np.polyfit(x_data, y_data, 1)
                fig.add_trace(
                    go.Scatter(
                        x=x_data, y=m * x_data + b, mode="lines", line=dict(color="red", width=1), showlegend=False
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            else:
                # --- Upper Triangle: Correlation Text ---
                r, p = pearsonr(x_data, y_data)

                # Significance stars
                stars = ""
                if p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
                elif p < 0.1:
                    stars = "."

                # Size based on correlation magnitude (like R)
                font_size = 10 + abs(r) * 20

                fig.add_trace(
                    go.Scatter(
                        x=[0.5],
                        y=[0.5],
                        mode="text",
                        text=[f"{r:.4f}{stars}"],
                        textfont=dict(size=font_size),
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                # Hide axes for text panel
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=row_idx, col=col_idx)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=row_idx, col=col_idx)

    fig.update_layout(
        height=200 * n, width=200 * n, title_text=main, template="plotly_white", margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig


def chart_rolling_correlation(
    Ra: pd.Series | pd.DataFrame,
    Rb: pd.Series | pd.DataFrame,
    width: int = 12,
    main: str | None = None,
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Chart of rolling correlation between two sets of assets.

    Matches R's `chart.RollingCorrelation` logic.

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Returns of first set of assets.
    Rb : pd.Series or pd.DataFrame
        Returns of second set of assets (often a benchmark).
    width : int, default 12
        Rolling window size.
    main : str, optional
        Chart title.
    colorset : list[str], optional
        List of colors for the traces.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object showing multi-line chart of pairwise correlations.
    r"""
    if isinstance(Ra, pd.Series):
        Ra = Ra.to_frame()
    if isinstance(Rb, pd.Series):
        Rb = Rb.to_frame()

    fig = go.Figure()

    # We calculate pairwise rolling correlation
    for i, col_a in enumerate(Ra.columns):
        for j, col_b in enumerate(Rb.columns):
            # Align
            merged = pd.concat([Ra[col_a], Rb[col_b]], axis=1).dropna()
            if len(merged) < width:
                continue

            # Rolling correlation
            roll_corr = merged.iloc[:, 0].rolling(window=width).corr(merged.iloc[:, 1])

            name = f"{col_a} vs {col_b}"
            color = colorset[(i + j) % len(colorset)] if colorset else None

            fig.add_trace(
                go.Scatter(x=roll_corr.index, y=roll_corr.values, mode="lines", name=name, line=dict(color=color))
            )

    fig.update_layout(
        title=main or f"Rolling {width}-period Correlation",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis_range=[-1.1, 1.1],
        template="plotly_white",
    )

    return fig


def chart_risk_return_scatter(
    R: pd.Series | pd.DataFrame,
    Rf: float = 0.0,
    scale: int | None = None,
    geometric: bool = True,
    add_sharpe: list[float] | None = None,
    main: str = "Annualized Return and Risk",
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Create a risk-return scatter plot with Sharpe ratio indifference lines.

    Matches R's `chart.RiskReturnScatter` logic.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets.
    Rf : float, default 0.0
        Risk-free rate for the Sharpe ratio lines intercept.
    scale : int, optional
        Annualization scale. If None, it is inferred from the data.
    geometric : bool, default True
        If True, uses geometric compounding for annualization.
    add_sharpe : list[float], optional
        List of Sharpe ratio values to draw as reference lines. Defaults to [1, 2, 3].
    main : str, default "Annualized Return and Risk"
        Chart title.
    colorset : list[str], optional
        List of colors for the assets.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with labeled scatter plot and reference lines.
    r"""
    if add_sharpe is None:
        add_sharpe = [1, 2, 3]

    from pyperfanalytics.returns import return_annualized, std_dev_annualized
    from pyperfanalytics.utils import _get_scale

    if isinstance(R, pd.Series):
        R = R.to_frame()

    if scale is None:
        scale = _get_scale(R)

    ann_ret = return_annualized(R, scale=scale, geometric=geometric)
    ann_std = std_dev_annualized(R, scale=scale)

    # Ensure ann_ret and ann_std are Series for consistent property access in type checking
    if not isinstance(ann_ret, (pd.Series, pd.DataFrame)):
        ann_ret = pd.Series([ann_ret], index=R.columns)
    if not isinstance(ann_std, (pd.Series, pd.DataFrame)):
        ann_std = pd.Series([ann_std], index=R.columns)

    fig = go.Figure()

    # 1. Calculate annualized Rf for intercept
    # R's table.AnnualizedReturns uses (1+Rf)^scale - 1 if geometric
    if geometric:
        rf_ann = (1 + Rf) ** scale - 1
    else:
        rf_ann = Rf * scale

    # 2. Set limits with 0.02 padding like R
    max_risk = ann_std.max() + 0.02
    min_ret = min(0, ann_ret.min())
    max_ret = ann_ret.max() + 0.02

    # 3. Add Sharpe ratio lines: y = rf_ann + SR * x
    x_range = np.linspace(0, max_risk, 100)
    for sr in add_sharpe:
        y_range = rf_ann + sr * x_range
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_range,
                mode="lines",
                name=f"SR={sr}",
                line=dict(color="lightgray", width=1, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 4. Add Assets
    # Use a color sequence for markers
    asset_colors = (
        colorset
        if colorset
        else [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=ann_std.values,
            y=ann_ret.values,
            mode="markers+text",
            text=ann_ret.index,
            textposition="top right",
            marker=dict(
                size=12,
                color=[asset_colors[i % len(asset_colors)] for i in range(len(ann_ret))],
                line=dict(width=1, color="white"),
                opacity=0.8,
            ),
            name="Assets",
            hovertext=[
                f"{name}<br>Return: {r:.4f}<br>Risk: {s:.4f}"
                for name, r, s in zip(ann_ret.index, ann_ret.values, ann_std.values, strict=False)
            ],
            hoverinfo="text",
        )
    )

    # 5. Update Layout to match R style
    fig.update_layout(
        title=main,
        xaxis_title="Annualized Risk (StdDev)",
        yaxis_title="Annualized Return",
        xaxis=dict(
            range=[0, max_risk],
            zeroline=True,
            zerolinecolor="black",
            gridcolor="lightgray",
            mirror=True,
            showline=True,
            linecolor="black",
        ),
        yaxis=dict(
            range=[min_ret, max_ret],
            zeroline=True,
            zerolinecolor="black",
            gridcolor="lightgray",
            mirror=True,
            showline=True,
            linecolor="black",
        ),
        width=800,
        height=800,  # Square plot
        template="plotly_white",
    )

    return fig


def chart_relative_performance(
    Ra: pd.Series | pd.DataFrame,
    Rb: pd.Series | pd.DataFrame,
    main: str = "Relative Performance",
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Plot the ratio of cumulative performance between two assets over time.

    Matches R's `chart.RelativePerformance` logic. Values > 1 indicate outperformance
    of the first asset (Ra) relative to the second (Rb).

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Returns of first asset(s).
    Rb : pd.Series or pd.DataFrame
        Returns of second asset(s) (benchmark).
    main : str, default "Relative Performance"
        Chart title.
    colorset : list[str], optional
        List of colors for the traces.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with the relative performance ratio chart.
    r"""
    if isinstance(Ra, pd.Series):
        Ra = Ra.to_frame()
    if isinstance(Rb, pd.Series):
        Rb = Rb.to_frame()

    fig = go.Figure()

    # Pairwise comparison as in R
    color_idx = 0
    for col_a in Ra.columns:
        for col_b in Rb.columns:
            # Align
            merged = pd.concat([Ra[col_a], Rb[col_b]], axis=1).dropna()
            if merged.empty:
                continue

            # Cumulative returns (Geometric)
            cum_a = (1 + merged.iloc[:, 0]).cumprod()
            cum_b = (1 + merged.iloc[:, 1]).cumprod()

            ratio = cum_a / cum_b

            name = f"{col_a} / {col_b}"
            color = colorset[color_idx % len(colorset)] if colorset else None
            color_idx += 1

            fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, mode="lines", name=name, line=dict(color=color)))

    # Add horizontal line at 1.0 (benchmark)
    fig.add_hline(y=1.0, line_dash="solid", line_color="darkgray")

    fig.update_layout(
        title=main, xaxis_title="Date", yaxis_title="Relative Performance (Ratio)", template="plotly_white"
    )

    return fig


def chart_capture_ratios(
    Ra: pd.Series | pd.DataFrame,
    Rb: pd.Series | pd.DataFrame,
    main: str = "Capture Ratio",
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Scatter plot of Upside Capture versus Downside Capture against a benchmark.

    Matches R's `chart.CaptureRatios` logic.

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Returns of asset(s).
    Rb : pd.Series or pd.DataFrame
        Returns of a benchmark asset (only the first column is used).
    main : str, default "Capture Ratio"
        Chart title.
    colorset : list[str], optional
        List of colors for the assets.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with (1,1) benchmark crosshairs.
    r"""
    from pyperfanalytics.returns import down_capture, up_capture

    if isinstance(Ra, pd.Series):
        Ra = Ra.to_frame()
    if isinstance(Rb, pd.Series):
        Rb = Rb.to_frame()

    # We only support one benchmark for this chart generally
    benchmark_name = Rb.columns[0]
    rb_series = Rb.iloc[:, 0]

    up_caps = []
    down_caps = []
    names = []

    for col in Ra.columns:
        up_caps.append(up_capture(Ra[col], rb_series))
        down_caps.append(down_capture(Ra[col], rb_series))
        names.append(col)

    # Set the charts to show the origin and the benchmark point (1,1)
    # PerformanceAnalytics logic:
    # xlim = c(min(0.75, downside - 0.2), max(1.25, downside + 0.2))
    # ylim = c(min(0.75, upside - 0.2), max(1.25, upside + 0.2))
    xmin = min(0.75, min(down_caps) - 0.2)
    xmax = max(1.25, max(down_caps) + 0.2)
    ymin = min(0.75, min(up_caps) - 0.2)
    ymax = max(1.25, max(up_caps) + 0.2)

    fig = go.Figure()

    # Add diagonal line y=x
    limit = max(xmax, ymax)
    fig.add_trace(
        go.Scatter(
            x=[0, limit],
            y=[0, limit],
            mode="lines",
            line=dict(color="darkgray", dash="dash", width=1),
            name="y=x",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add crosshairs for benchmark at (1,1)
    fig.add_hline(y=1.0, line_color="darkgray", line_width=1)
    fig.add_vline(x=1.0, line_color="darkgray", line_width=1)

    # Add benchmark point
    fig.add_trace(
        go.Scatter(
            x=[1.0],
            y=[1.0],
            mode="markers+text",
            text=[benchmark_name],
            textposition="bottom right",
            marker=dict(size=12, color="black"),
            name=benchmark_name,
            showlegend=False,
        )
    )

    # Add Asset points
    asset_colors = colorset if colorset else ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    fig.add_trace(
        go.Scatter(
            x=down_caps,
            y=up_caps,
            mode="markers+text",
            text=names,
            textposition="top right",
            marker=dict(
                size=10,
                color=[asset_colors[i % len(asset_colors)] for i in range(len(names))],
                line=dict(width=1, color="white"),
            ),
            name="Assets",
            hovertext=[
                f"{n}<br>Up Capture: {u:.2%}<br>Down Capture: {d:.2%}"
                for n, u, d in zip(names, up_caps, down_caps, strict=False)
            ],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=main,
        xaxis_title="Downside Capture",
        yaxis_title="Upside Capture",
        xaxis=dict(range=[xmin, xmax], zeroline=True, zerolinecolor="black", gridcolor="lightgray"),
        yaxis=dict(range=[ymin, ymax], zeroline=True, zerolinecolor="black", gridcolor="lightgray"),
        width=800,
        height=800,
        template="plotly_white",
    )

    return fig


def chart_rolling_regression(
    Ra: pd.Series | pd.DataFrame,
    Rb: pd.Series | pd.DataFrame,
    width: int = 12,
    Rf: float | pd.Series | pd.DataFrame = 0.0,
    attribute: str = "Beta",
    main: str | None = None,
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Chart of rolling regression performance metrics (Alpha, Beta, or R-Squared).

    Matches R's `chart.RollingRegression` logic.

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Returns of asset(s).
    Rb : pd.Series or pd.DataFrame
        Returns of benchmark(s).
    width : int, default 12
        Rolling window size.
    Rf : float or pd.Series or pd.DataFrame, default 0.0
        Risk-free rate.
    attribute : str, default "Beta"
        Regression metric to plot: "Alpha", "Beta", or "R-Squared".
    main : str, optional
        Chart title.
    colorset : list[str], optional
        List of colors for the traces.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with rolling regression metric.
    r"""
    from scipy import stats

    from pyperfanalytics.returns import return_excess

    if isinstance(Ra, pd.Series):
        Ra = Ra.to_frame()
    if isinstance(Rb, pd.Series):
        Rb = Rb.to_frame()

    # Calculate excess returns
    xRa = return_excess(Ra, Rf)
    xRb = return_excess(Rb, Rf)

    fig = go.Figure()

    color_idx = 0
    for col_a in xRa.columns:
        for col_b in xRb.columns:
            # Align
            merged = pd.concat([xRa[col_a], xRb[col_b]], axis=1).dropna()
            if len(merged) < width:
                continue

            vals = []
            for i in range(len(merged)):
                if i < width - 1:
                    vals.append(np.nan)
                else:
                    window = merged.iloc[i - width + 1 : i + 1]
                    # Regression y ~ x
                    slope, intercept, r_value, p_value, std_err = stats.linregress(window.iloc[:, 1], window.iloc[:, 0])

                    if attribute == "Alpha":
                        vals.append(intercept)
                    elif attribute == "Beta":
                        vals.append(slope)
                    elif attribute == "R-Squared":
                        vals.append(r_value**2)
                    else:
                        raise ValueError(f"Unknown attribute: {attribute}")

            res_series = pd.Series(vals, index=merged.index)
            name = f"{col_a} to {col_b}"
            color = colorset[color_idx % len(colorset)] if colorset else None
            color_idx += 1

            fig.add_trace(
                go.Scatter(x=res_series.index, y=res_series.values, mode="lines", name=name, line=dict(color=color))
            )

    fig.update_layout(
        title=main or f"Rolling {width}-period {attribute}",
        xaxis_title="Date",
        yaxis_title=attribute,
        template="plotly_white",
    )

    return fig


def charts_rolling_regression(
    Ra: pd.Series | pd.DataFrame,
    Rb: pd.Series | pd.DataFrame,
    width: int = 12,
    Rf: float | pd.Series | pd.DataFrame = 0.0,
    main: str = "Rolling Regression Summary",
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Dashboard with Rolling Alpha, Rolling Beta, and Rolling R-Squared charts.

    Matches R's `charts.RollingRegression` logic.

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Returns of asset(s).
    Rb : pd.Series or pd.DataFrame
        Returns of benchmark(s).
    width : int, default 12
        Rolling window size.
    Rf : float or pd.Series or pd.DataFrame, default 0.0
        Risk-free rate.
    main : str, default "Rolling Regression Summary"
        Dashboard title.
    colorset : list[str], optional
        List of colors for the traces.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with 3 subplots (Alpha, Beta, R-Squared).
    r"""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Rolling Alpha", "Rolling Beta", "Rolling R-Squared"),
    )

    # 1. Alpha
    f_alpha = chart_rolling_regression(Ra, Rb, width=width, Rf=Rf, attribute="Alpha", colorset=colorset)
    for trace in f_alpha.data:
        fig.add_trace(trace, row=1, col=1)

    # 2. Beta
    f_beta = chart_rolling_regression(Ra, Rb, width=width, Rf=Rf, attribute="Beta", colorset=colorset)
    for trace in f_beta.data:
        trace.showlegend = False  # Only show legend once
        fig.add_trace(trace, row=2, col=1)

    # 3. R-Squared
    f_r2 = chart_rolling_regression(Ra, Rb, width=width, Rf=Rf, attribute="R-Squared", colorset=colorset)
    for trace in f_r2.data:
        trace.showlegend = False
        fig.add_trace(trace, row=3, col=1)

    fig.update_layout(height=900, title_text=main, template="plotly_white", hovermode="x unified")

    # Update y-axis titles for each subplot
    fig.update_yaxes(title_text="Alpha", zeroline=True, zerolinecolor="black", row=1, col=1)
    fig.update_yaxes(title_text="Beta", zeroline=True, zerolinecolor="black", row=2, col=1)
    fig.update_yaxes(title_text="R-Squared", row=3, col=1)

    return fig


def chart_acf(R: pd.Series | pd.DataFrame, maxlag: int | None = None, main: str | None = None, **kwargs) -> go.Figure:
    r"""
    Create an Autocorrelation Function (ACF) chart.

    Matches R's `chart.ACF` logic (often called via `chart.ACFplus`).

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets. Only the first column is used if a DataFrame is passed.
    maxlag : int, optional
        Maximum number of lags to display. If None, it is calculated based on sample size.
    main : str, optional
        Chart title.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with ACF bars and confidence interval lines.
    r"""
    from statsmodels.tsa.stattools import acf

    if isinstance(R, pd.DataFrame):
        series = R.iloc[:, 0].dropna()
    else:
        series = R.dropna()

    num = len(series)
    if maxlag is None:
        maxlag = int(np.ceil(10 + np.sqrt(num)))

    # Calculate ACF, excluding lag 0 as in R's chart.ACFplus
    acf_vals = acf(series, nlags=maxlag, fft=True)[1:]
    lags = np.arange(1, len(acf_vals) + 1)

    # Confidence intervals +/- 2/sqrt(n)
    ci = 2.0 / np.sqrt(num)

    fig = go.Figure()

    fig.add_trace(go.Bar(x=lags, y=acf_vals, name="ACF", marker_color="lightgray"))

    # Add CI lines
    fig.add_hline(y=ci, line_dash="dash", line_color="blue", opacity=0.5)
    fig.add_hline(y=-ci, line_dash="dash", line_color="blue", opacity=0.5)
    fig.add_hline(y=0, line_color="black")

    fig.update_layout(
        title=main or f"Autocorrelation of {series.name if hasattr(series, 'name') else 'Returns'}",
        xaxis_title="Lag",
        yaxis_title="ACF",
        template="plotly_white",
    )

    return fig


def chart_acf_plus(
    R: pd.Series | pd.DataFrame, maxlag: int | None = None, main: str | None = None, **kwargs
) -> go.Figure:
    r"""
    Create a chart with both ACF and PACF subplots.

    Matches R's `chart.ACFplus` logic.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets. Only the first column is used if a DataFrame is passed.
    maxlag : int, optional
        Maximum number of lags to display. If None, it is calculated based on sample size.
    main : str, optional
        Chart title.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with 2 subplots (ACF and PACF).
    r"""
    from statsmodels.tsa.stattools import acf, pacf

    if isinstance(R, pd.DataFrame):
        series = R.iloc[:, 0].dropna()
    else:
        series = R.dropna()

    num = len(series)
    if maxlag is None:
        maxlag = int(np.ceil(10 + np.sqrt(num)))

    acf_vals = acf(series, nlags=maxlag, fft=True)[1:]
    pacf_vals = pacf(series, nlags=maxlag)[1:]
    lags = np.arange(1, len(acf_vals) + 1)

    ci = 2.0 / np.sqrt(num)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"),
    )

    # ACF
    fig.add_trace(go.Bar(x=lags, y=acf_vals, name="ACF", marker_color="lightgray", showlegend=False), row=1, col=1)

    # PACF
    fig.add_trace(go.Bar(x=lags, y=pacf_vals, name="PACF", marker_color="lightgray", showlegend=False), row=2, col=1)

    # Add CI lines to both
    for r in [1, 2]:
        fig.add_hline(y=ci, line_dash="dash", line_color="blue", opacity=0.5, row=r, col=1)
        fig.add_hline(y=-ci, line_dash="dash", line_color="blue", opacity=0.5, row=r, col=1)
        fig.add_hline(y=0, line_color="black", row=r, col=1)

    fig.update_layout(
        height=700,
        title_text=main or f"Time Series Diagnostics: {series.name if hasattr(series, 'name') else ''}",
        template="plotly_white",
    )

    return fig


def chart_events(
    R: pd.Series | pd.DataFrame,
    dates: list[str | pd.Timestamp],
    prior: int = 12,
    post: int = 12,
    main: str | None = None,
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Plots a time series with event dates aligned.

    Relative X-axis shows periods before and after the event.
    Matches R's `chart.Events` logic.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns or price series. Only the first column is used.
    dates : list of str or pd.Timestamp
        List of event dates to align.
    prior : int, default 12
        Number of periods to show before each event.
    post : int, default 12
        Number of periods to show after each event.
    main : str, optional
        Chart title.
    colorset : list[str], optional
        List of colors for the event lines.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object with aligned event traces.
    r"""
    if isinstance(R, pd.Series):
        R = R.to_frame()

    series = R.iloc[:, 0]

    fig = go.Figure()

    # Relative index
    rel_index = np.arange(-prior, post + 1)

    for i, date in enumerate(dates):
        # Find index of date
        try:
            # Flexible date matching
            dt = pd.to_datetime(date)
            # Find closest index or exact match
            origin_idx = series.index.get_indexer([dt], method="nearest")[0]
        except Exception:
            print(f"Warning: Could not find date {date} in index.")
            continue

        # Extract window
        start = origin_idx - prior
        end = origin_idx + post + 1

        # Slice and pad if necessary
        window_data = []
        for j in range(start, end):
            if j < 0 or j >= len(series):
                window_data.append(np.nan)
            else:
                window_data.append(series.iloc[j])

        color = colorset[i % len(colorset)] if colorset else None

        fig.add_trace(
            go.Scatter(
                x=rel_index,
                y=window_data,
                mode="lines+markers",
                name=str(date),
                line=dict(color=color),
                marker=dict(size=4),
            )
        )

    # Add vertical line at event point (0)
    fig.add_vline(x=0, line_dash="dash", line_color="black")

    fig.update_layout(
        title=main or f"Event Study: {series.name}",
        xaxis_title="Periods to Event",
        yaxis_title="Value",
        template="plotly_white",
        xaxis=dict(tickmode="linear", tick0=-prior, dtick=max(1, (prior + post) // 10)),
    )

    return fig


def chart_snail_trail(
    R: pd.Series | pd.DataFrame,
    Rf: float = 0.0,
    width: int = 12,
    stepsize: int = 12,
    scale: int | None = None,
    geometric: bool = True,
    main: str = "Annualized Return and Risk Trail",
    add_sharpe: list[float] | None = None,
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Create a snail trail chart showing rolling risk-return evolution.

    Matches R's `chart.SnailTrail`.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of assets.
    Rf : float, default 0.0
        Risk-free rate for Sharpe ratio line calculation.
    width : int, default 12
        Rolling window width for annualized return and risk.
    stepsize : int, default 12
        Periods between trail markers.
    scale : int, optional
        Periods per year (e.g., 252 for daily).
    geometric : bool, default True
        If True, use geometric compounding.
    main : str, default "Annualized Return and Risk Trail"
        Chart title.
    add_sharpe : list[float], optional
        Sharpe ratio reference lines to add (default [1, 2, 3]).
    colorset : list[str], optional
        List of colors for asset trails.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object showing risk-return trails.
    r"""
    if add_sharpe is None:
        add_sharpe = [1, 2, 3]

    from pyperfanalytics.returns import return_annualized, std_dev_annualized
    from pyperfanalytics.utils import _get_scale

    if isinstance(R, pd.Series):
        R = R.to_frame()

    if scale is None:
        scale = _get_scale(R)

    fig = go.Figure()

    # Standard colors for Sharpe lines
    if geometric:
        rf_ann = (1 + Rf) ** scale - 1
    else:
        rf_ann = Rf * scale

    # Calculate rolling metrics
    all_trace_data = []
    for col in R.columns:
        series = R[col].dropna()
        dates = []
        rets = []
        stds = []

        # PerformanceAnalytics logic: (nrow(y)%%stepsize+1):nrow(y)
        start_idx = len(series) % stepsize
        for i in range(start_idx + width, len(series) + 1, stepsize):
            window = series.iloc[i - width : i]

            # Extract scalar values safely
            ret_val = return_annualized(window, scale=scale, geometric=geometric)
            if isinstance(ret_val, (pd.Series, pd.DataFrame)):
                ret_val = ret_val.iloc[0]

            std_val = std_dev_annualized(window, scale=scale)
            if isinstance(std_val, (pd.Series, pd.DataFrame)):
                std_val = std_val.iloc[0]

            rets.append(ret_val)
            stds.append(std_val)
            dates.append(series.index[i - 1])

        all_trace_data.append({"name": col, "rets": rets, "stds": stds, "dates": [d.strftime("%Y-%m") for d in dates]})

    # Global limits for Sharpe lines
    max_std = max([max(d["stds"]) for d in all_trace_data]) if all_trace_data else 0.1
    max_std = max_std * 1.2
    x_range = np.linspace(0, max_std, 100)
    for sr in add_sharpe:
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=rf_ann + sr * x_range,
                mode="lines",
                line=dict(color="lightgray", width=1, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add Trails
    asset_colors = colorset if colorset else ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    for i, data_dict in enumerate(all_trace_data):
        color = asset_colors[i % len(asset_colors)]

        # To create a "trail" effect (fading), we can't easily do it with one trace in Plotly
        # unless we use multiple segments or a color array.
        # Simple version: one line with markers
        fig.add_trace(
            go.Scatter(
                x=data_dict["stds"],
                y=data_dict["rets"],
                mode="lines+markers",
                name=data_dict["name"],
                text=data_dict["dates"],
                line=dict(color=color, width=2),
                marker=dict(
                    size=8,
                    color=np.arange(len(data_dict["rets"])),  # Gradient over time
                    colorscale=[[0, "white"], [1, color]],
                    showscale=False,
                ),
                hovertemplate=(
                    "<b>%{name}</b><br>Date: %{text}<br>Ann. Return: %{y:.4f}<br>Ann. StdDev: %{x:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=main,
        xaxis_title="Annualized Standard Deviation",
        yaxis_title="Annualized Return",
        template="plotly_white",
        width=800,
        height=800,
    )

    return fig


def chart_stacked_bar(
    w: pd.Series | pd.DataFrame, main: str = "Stacked Bar Chart", colorset: list[str] | None = None, **kwargs
) -> go.Figure:
    r"""
    Create a stacked bar plot.

    Commonly used for weights or contributions. Matches R's `chart.StackedBar`.

    Parameters
    ----------
    w : pd.Series or pd.DataFrame
        Values to stack (e.g., weights or return contributions).
    main : str, default "Stacked Bar Chart"
        Chart title.
    colorset : list[str], optional
        List of colors for the stacks.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object.
    r"""
    if isinstance(w, pd.Series):
        w = w.to_frame()

    fig = go.Figure()

    for i, col in enumerate(w.columns):
        color = colorset[i % len(colorset)] if colorset else None
        fig.add_trace(go.Bar(x=w.index, y=w[col], name=col, marker_color=color))

    fig.update_layout(
        title=main,
        xaxis_title="Date",
        yaxis_title="Value",
        barmode="relative",  # Essential for positive/negative stacking
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def chart_component_returns(
    R: pd.Series | pd.DataFrame,
    weights: list[float] | np.ndarray | pd.Series | None = None,
    main: str = "Component Returns Contribution",
    **kwargs,
) -> go.Figure:
    r"""
    Plots the contribution of each asset to the portfolio return.

    Contribution is calculated as :math:`R_i \times W_i`. Matches R's `chart.ComponentReturns`.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of component assets.
    weights : array-like, optional
        Weights of each asset. Defaults to equal weights.
    main : str, default "Component Returns Contribution"
        Chart title.
    **kwargs
        Additional arguments passed to `chart_stacked_bar`.

    Returns
    -------
    go.Figure
        Plotly figure object.
    r"""
    if isinstance(R, pd.Series):
        R = R.to_frame()

    if weights is None:
        weights = np.ones(R.shape[1]) / R.shape[1]

    # Contribution = Return * Weight
    contrib = R.multiply(weights, axis=1)

    return chart_stacked_bar(contrib, main=main, **kwargs)


def chart_ecdf(
    R: pd.Series | pd.DataFrame,
    main: str = "Empirical Cumulative Distribution Function",
    xlab: str = "Returns",
    ylab: str = "Cumulative Probability",
    colorset: list[str] | None = None,
    **kwargs,
) -> go.Figure:
    r"""
    Create an aesthetically enhanced ECDF chart overlaid with a normal CDF.

    Matches R's `chart.ECDF`.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Returns of an asset. Only the first column is used.
    main : str, default "Empirical Cumulative Distribution Function"
        Chart title.
    xlab : str, default "Returns"
        X-axis label.
    ylab : str, default "Cumulative Probability"
        Y-axis label.
    colorset : list[str], optional
        List of colors for ECDF and Normal lines.
    **kwargs
        Additional arguments passed to Plotly layout.

    Returns
    -------
    go.Figure
        Plotly figure object.
    r"""
    from scipy.stats import norm

    if isinstance(R, pd.DataFrame):
        series = R.iloc[:, 0].dropna()
    else:
        series = R.dropna()

    sorted_data = np.sort(series)
    y_ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    fig = go.Figure()

    # Coordinated colors
    c_ecdf = colorset[0] if colorset else "#636EFA"  # Modern Blue
    c_norm = colorset[1] if colorset and len(colorset) > 1 else "#EF553B"  # Muted Red

    # Normal CDF Background Line (Reference)
    mu, std = series.mean(), series.std(ddof=1)
    x_range = np.linspace(sorted_data[0] - abs(sorted_data[0]) * 0.1, sorted_data[-1] + abs(sorted_data[-1]) * 0.1, 300)
    y_normal = norm.cdf(x_range, mu, std)

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_normal,
            mode="lines",
            name="Normal Distribution",
            line=dict(color=c_norm, width=2, dash="dot"),
            opacity=0.8,
        )
    )

    # ECDF Step Trace with Area Fill
    fig.add_trace(
        go.Scatter(
            x=sorted_data,
            y=y_ecdf,
            mode="lines",
            name="Empirical CDF",
            line=dict(color=c_ecdf, shape="hv", width=3),
            fill="tozeroy",  # Subtle fill under the curve
            fillcolor="rgba(99, 110, 250, 0.1)",
        )
    )

    fig.update_layout(
        title=main,
        xaxis_title=xlab,
        yaxis_title=ylab,
        template="plotly_white",
        yaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="lightgray"),
        xaxis=dict(gridcolor="lightgray", zeroline=True, zerolinecolor="black"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x",
    )

    return fig


def chart_scatter(
    x: pd.Series | pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    main: str = "Scatter Plot",
    xlab: str | None = None,
    ylab: str | None = None,
    marginal: str = "rug",  # 'histogram', 'rug', 'box', 'violin'
    add_regression: bool = True,
    **kwargs,
) -> go.Figure:
    r"""
    Create a scatter plot with optional marginal distributions and regression line.

    Matches R's `chart.Scatter`.

    Parameters
    ----------
    x : pd.Series or pd.DataFrame
        Independent variable.
    y : pd.Series or pd.DataFrame
        Dependent variable.
    main : str, default "Scatter Plot"
        Chart title.
    xlab : str, optional
        X-axis label.
    ylab : str, optional
        Y-axis label.
    marginal : str, default "rug"
        Type of marginal plot (e.g., 'histogram', 'rug', 'box', 'violin').
    add_regression : bool, default True
        If True, adds an OLS regression line.
    **kwargs
        Additional arguments passed to plotly.express scatter.

    Returns
    -------
    go.Figure
        Plotly figure object.
    r"""
    import plotly.express as px

    if isinstance(x, pd.DataFrame):
        x_name = x.columns[0]
        x_series = x.iloc[:, 0]
    else:
        x_name = x.name if hasattr(x, "name") else "x"
        x_series = x

    if isinstance(y, pd.DataFrame):
        y_name = y.columns[0]
        y_series = y.iloc[:, 0]
    else:
        y_name = y.name if hasattr(y, "name") else "y"
        y_series = y

    # Align data
    df_plot = pd.concat([x_series, y_series], axis=1).dropna()
    df_plot.columns = [x_name, y_name]

    fig = px.scatter(
        df_plot,
        x=x_name,
        y=y_name,
        marginal_x=marginal,
        marginal_y=marginal,
        trendline="ols" if add_regression else None,
        template="plotly_white",
        title=main,
        labels={x_name: xlab or x_name, y_name: ylab or y_name},
    )

    # Styling regression line if present
    if add_regression:
        fig.update_traces(line=dict(color="red", width=1), selector=dict(mode="lines"))

    return fig
