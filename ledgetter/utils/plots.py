
import numpy as np
import ledgetter.utils.vector_tools as vector_tools
import plotly.graph_objects as go
import plotly.express as px
import jax

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


def plot_directional(light_power, light_directions, mask, albedo, names=None):
    rel_power = light_power / jax.numpy.max(light_power)
    albedomap = vector_tools.build_masked(mask, jax.numpy.uint8(255*jax.numpy.clip(albedo/jax.numpy.quantile(albedo,0.95), 0, 1)), fill_value=jax.numpy.nan)
    x0, y0, r = mask.shape[1]/2, mask.shape[0]/2, min(mask.shape[1], mask.shape[0])/2
    image = go.Heatmap(
        z=jax.numpy.mean(albedomap, axis=-1),
        colorscale='gray',
        zmin=0,
        zmax=255,
        showscale=False,
    )
    lights = go.Scatter(
        x= x0 + light_directions[:,0]*r,
        y= y0 + light_directions[:,1]*r,
        mode='markers+text' if names is not None else 'markers',
        marker=dict(
            size=10,
            color=rel_power,
            colorscale='solar',
            colorbar=dict(title='Relative Power'),
            cmin=0,
            cmax=1,
            symbol='star',
            line=dict(width=1.5, color='black')
        ),
        name='Lights',
        text=names,
        showlegend=True
    )

    image_legend = go.Scatter(
        x=[None], y=[None],  # point invisible
        mode='markers',
        marker=dict(
            size=10,
            color='gray',
            symbol='square'
        ),
        name='Albedo Map',
        showlegend=True
    )

    origin_marker = go.Scatter(
        x=[x0],
        y=[y0],
        mode='markers',
        marker=dict(
            symbol='cross',
            size= 10,
            color='black',
            line=dict(width=1.5)
        ),
        name='Origin',
        showlegend=True
    )

    layout = go.Layout(
        xaxis=dict(title='X', showgrid=False, scaleanchor='y'),
        yaxis=dict(title='Y', showgrid=False),
        title='Directional Light',
        legend=dict(x=0, y=1),
    )
    fig = go.Figure(data=[image, image_legend, origin_marker, lights], layout=layout)
    
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=x0-r, y0=y0-r,
        x1=x0+r, y1=y0+r,
        opacity=0.2,
        fillcolor="blue",
        line_color="black",
        name='Unit Sphere',
        showlegend=True
    )


    return fig

def plot_punctual(light_power, light_locations, mask, points, albedo, names=None):
    rel_power = light_power / np.max(light_power)
    pointmap, albedomap = vector_tools.build_masked(mask, points, fill_value=np.nan), vector_tools.build_masked(mask, jax.numpy.clip(albedo/jax.numpy.quantile(albedo,0.95), 0, 1), fill_value=np.nan)

    surface = go.Surface(
        x=pointmap[:, :, 0],
        y=pointmap[:, :, 1],
        z=pointmap[:, :, 2],
        surfacecolor=np.mean(albedomap, axis=2),
        colorscale='gray',
        showscale=False,
        opacity=0.8,
        name='Surface',
        showlegend=True
    )

    scatter = go.Scatter3d(
        x=light_locations[:, 0],
        y=light_locations[:, 1],
        z=light_locations[:, 2],
        mode='markers+text' if names is not None else 'markers',
        marker=dict(
            size=6,
            color=rel_power,
            colorscale='solar',
            colorbar=dict(title='Relative Power'),
            cmin=0,
            cmax=1,
        ),
        text=names if names is not None else None,
        textposition="top center",
        name="Light",
        showlegend=True
    )

    origin_marker = go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(
            symbol='cross',
            size= 6,
            color='black',
            line=dict(width=1.5)
        ),
        name='Camera',
        showlegend=True
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        title='Directional Light',
        legend=dict(x=0, y=1),
    )

    fig = go.Figure(data= [scatter, origin_marker, surface], layout=layout)
    return fig

def plot_LED(light_power, light_locations, light_principal_direction, mu, mask, points, albedo, names=None):
    rel_power = light_power / np.max(light_power)
    radius = jax.numpy.mean(jax.numpy.linalg.norm(light_locations-jax.numpy.mean(points,axis=0), axis=-1))
    pointmap, albedomap = vector_tools.build_masked(mask, points, fill_value=np.nan), vector_tools.build_masked(mask, jax.numpy.clip(albedo/jax.numpy.quantile(albedo,0.95), 0, 1), fill_value=np.nan)

    surface = go.Surface(
        x=pointmap[:, :, 0],
        y=pointmap[:, :, 1],
        z=pointmap[:, :, 2],
        surfacecolor=np.mean(albedomap, axis=2),
        colorscale='gray',
        showscale=False,
        opacity=0.8,
        name='Surface',
        showlegend=True
    )

    scatter = go.Scatter3d(
        x=light_locations[:, 0],
        y=light_locations[:, 1],
        z=light_locations[:, 2],
        mode='markers+text' if names is not None else 'markers',
        marker=dict(
            size=3,
            color='black',
        ),
        text=names if names is not None else None,
        textposition="top center",
        name="Light",
        showlegend=True
    )

    origin_marker = go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(
            symbol='cross',
            size= 6,
            color='black',
            line=dict(width=1.5)
        ),
        name='Camera',
        showlegend=True
    )

    surfaces = []
    colors = ['Reds', 'Greens', 'Blues']
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    for l in range(light_principal_direction.shape[0]):
        for c in range(3):
            show_colorbar = l==0
            flux = jax.numpy.power(jax.numpy.maximum(0,light_principal_direction[l,0]*x+light_principal_direction[l,1]*y+light_principal_direction[l,2]*z), mu[c])
            scale = flux * radius*0.1 * rel_power[l]
            surface_led = go.Surface(
                x=light_locations[l, 0]+x*scale, y=light_locations[l, 1]+y*scale, z=light_locations[l, 2]+z*scale,
                surfacecolor=flux,
                colorscale=colors[c],
                opacity = 0.2,
                name='Emission profile' if (l==0 and c==0) else None,
                showlegend= (l==0 and c==0),
                showscale = show_colorbar,
                colorbar=dict(
                    title=['Red', 'Green', 'Blue'][c],
                    x=1.0 + 0.05 * c,  # adjust spacing
                    len=0.5
                ) if show_colorbar else None
            )
            surfaces.append(surface_led)
    


    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        title='Directional Light',
        legend=dict(x=0, y=1),
    )

    fig = go.Figure(data= [scatter, origin_marker, surface]+surfaces, layout=layout)

    return fig


