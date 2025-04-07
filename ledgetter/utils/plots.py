
import numpy as np
import plotly.graph_objects as go

def get_losses_plot(losses, steps):
    iterations = list(map(len, losses))
    boundaries = np.cumsum([0] + iterations)
    x_vals = np.arange(sum(iterations))
    y_vals = np.concatenate(losses)

    fig = go.Figure()

    # Add logarithmic scale trace (left y-axis)
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        name="logarithmic",
        yaxis="y1",
        line=dict(color="red"),
    ))

    # Add linear scale trace (right y-axis)
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        name="linear",
        yaxis="y2",
        line=dict(color="blue"),
    ))

    # Add alternating background spans and step annotations
    for i in range(len(iterations)):
        start = boundaries[i] if i != 0 else boundaries[0] - 0.1 * boundaries[-1]
        end = boundaries[i + 1]
        color = "wheat" if i % 2 == 0 else "lightblue"
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            opacity=0.3,
            line_width=0,
            layer="below"
        )
        # Text annotation at top
        fig.add_annotation(
            x=(start + end) / 2,
            y=1,
            xref="x",
            yref="paper",
            text=steps[i],
            showarrow=False,
            font=dict(size=12),
            yanchor="top"
        )

    fig.update_layout(
        title="Loss over Iterations with Optimization Steps",
        xaxis=dict(title="Iteration"),
        yaxis=dict(
            title=dict(text="Logarithmic Loss", font=dict(color="red")),
            type="log",
            tickfont=dict(color="red"),
        ),
        yaxis2=dict(
            title=dict(text="Linear Loss", font=dict(color="blue")),
            overlaying="y",
            side="right",
            tickfont=dict(color="blue"),
            showgrid=True,
        ),
        legend=dict(x=0.75, y=0.95),
        template="simple_white",
    )

    return fig
