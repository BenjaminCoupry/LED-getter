
import numpy as np
import ledgetter.utils.vector_tools as vector_tools
import ledgetter.space.spherical_harmonics as spherical_harmonics
import ledgetter.utils.functions as functions
import plotly.graph_objects as go
import plotly.express as px
import jax

def plot_losses(losses, steps):
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


def get_directional_lights(dir_light_power, light_directions, names=None):
    rel_power = dir_light_power / np.max(dir_light_power)
    x = light_directions[:, 0]
    y = light_directions[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.degrees(np.arctan2(y, x))
    lights = go.Scatterpolar(
        r=r,
        theta=theta,
        mode='markers+text' if names is not None else 'markers',
        marker=dict(
            size=10,
            color=rel_power,
            colorscale='solar',
            colorbar=dict(
                title='Relative Power'),
            cmin=0,
            cmax=1,
            symbol='star',
            line=dict(width=1.5, color='black')
        ),
        textfont=dict(size=14),
        name='Lights',
        text=names,
        showlegend=True
    )
    objects = [lights]
    return objects

def get_surface(mask, points, rho):
    step_u, step_v = max(1,int(mask.shape[0]/100)), max(1,int(mask.shape[1]/100))
    pointmap = vector_tools.build_masked(mask, points, fill_value=np.nan)[::step_u, ::step_v]
    rhomap = vector_tools.build_masked(mask, jax.numpy.clip(rho/jax.numpy.quantile(rho,0.95), 0, 1), fill_value=np.nan)[::step_u, ::step_v]
    surface = go.Surface(
        x=pointmap[:, :, 0],
        y=pointmap[:, :, 1],
        z=pointmap[:, :, 2],
        surfacecolor=np.mean(rhomap, axis=2),
        colorscale='gray',
        showscale=False,
        opacity=0.8,
        name='Surface',
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
    objects = [surface, origin_marker]
    return objects

def get_punctual_lights(light_power, light_locations, names=None):
    rel_power = light_power / np.max(light_power)
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
    objects = [scatter]
    return objects

def get_anisotropies(light_power, light_locations, radius, anisotropy_func):
    rel_power = light_power / np.max(light_power)
    objects = []
    colors = ['Reds', 'Greens', 'Blues']
    theta = np.linspace(0, 2 * np.pi, 50)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    cartesian = np.stack([x,y,z], axis=-1)
    for l in range(light_locations.shape[0]):
        for c in range(3):
            flux = anisotropy_func(cartesian, l, c)
            scale = flux * radius*0.1 * rel_power[l]
            surface_led = go.Surface(
                x=light_locations[l, 0]+x*scale, y=light_locations[l, 1]+y*scale, z=light_locations[l, 2]+z*scale,
                surfacecolor=flux,
                colorscale=colors[c],
                opacity = 0.2,
                name='Emission profile' if (l==0 and c==0) else None,
                showlegend = (l==0 and c==0),
                showscale=False
            )
            objects.append(surface_led)
    return objects

def get_LED_anisotropies(light_power, light_locations, radius, light_principal_direction, mu):
    anisotropy_func = lambda cartesian, l, c : np.power(np.maximum(0,np.sum(light_principal_direction[l]*cartesian, axis=-1)), mu[c])
    objects = get_anisotropies(light_power, light_locations, radius, anisotropy_func)
    return objects

def get_harmonic_anisotropies(light_power, light_locations, radius, light_principal_direction, free_rotation, coefficients, l_max):
    anisotropy_func = lambda cartesian, l, c : jax.nn.relu(spherical_harmonics.oriented_sh_function(cartesian, light_principal_direction[l], free_rotation[l], coefficients[:,c], l_max))
    objects = get_anisotropies(light_power, light_locations, radius, anisotropy_func)
    return objects


def plot_directional_light(dir_light_power, light_directions, names=None):
    lights = get_directional_lights(dir_light_power, light_directions, names=names)

    layout = go.Layout(
        polar=dict(
                radialaxis=dict(showgrid=True, showline=False),
                angularaxis=dict(showgrid=True, direction='counterclockwise'),
            ),
        title='Directional Light',
        legend=dict(x=0, y=1),
    )

    fig = go.Figure(data= lights, layout=layout)
    return fig

def plot_punctual_light(light_power, light_locations, mask, points, rho, names=None):
    surface = get_surface(mask, points, rho)
    lights = get_punctual_lights(light_power, light_locations, names=names)

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

    fig = go.Figure(data= surface + lights, layout=layout)
    return fig

def plot_LED_light(light_power, light_locations, light_principal_direction, mu, mask, points, rho, names=None):
    radius = jax.numpy.mean(jax.numpy.linalg.norm(light_locations-jax.numpy.mean(points,axis=0), axis=-1))
    surface = get_surface(mask, points, rho)
    lights = get_punctual_lights(light_power, light_locations, names=names)
    anisotropies = get_LED_anisotropies(light_power, light_locations, radius, light_principal_direction, mu)

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

    fig = go.Figure(data= surface + lights + anisotropies, layout=layout)
    return fig

def plot_harmonic_light(light_power, light_locations, light_principal_direction, free_rotation, coefficients, l_max, mask, points, rho, names=None):
    radius = jax.numpy.mean(jax.numpy.linalg.norm(light_locations-jax.numpy.mean(points,axis=0), axis=-1))
    surface = get_surface(mask, points, rho)
    lights = get_punctual_lights(light_power, light_locations, names=names)
    anisotropies = get_harmonic_anisotropies(light_power, light_locations, radius, light_principal_direction, free_rotation, coefficients, l_max)

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

    fig = go.Figure(data= surface + lights + anisotropies, layout=layout)
    return fig


def get_plot_light(light):
    match light:
        case 'directional':
            return plot_directional_light
        case 'punctual':
            return plot_punctual_light
        case 'LED':
            return plot_LED_light
        case 'harmonic':
            return plot_harmonic_light
        case _:
            return None