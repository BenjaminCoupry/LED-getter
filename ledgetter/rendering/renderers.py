import jax

def lambertian_renderer(light_direction, light_intensity, normals, points, rho):
    render =  jax.nn.relu(jax.numpy.einsum('...li, ...lc, ...i, ...c -> ...cl', light_direction, light_intensity, normals, rho))
    return render

def specular_renderer(light_direction, light_intensity, normals, points, rho_spec, tau_spec):
    camera_direction = - points/jax.numpy.linalg.norm(points, axis=-1, keepdims=True)
    light_dot = 2 * jax.nn.relu(jax.numpy.einsum('...i, ...li -> ...l',normals, light_direction))
    total_reflexion = jax.numpy.einsum('...l, ...i -> ...li', light_dot, normals) - light_direction
    specular_angular_factor = jax.nn.relu(jax.numpy.einsum('...li, ...i -> ...l', total_reflexion, camera_direction))
    reflected_energy = jax.numpy.power(specular_angular_factor + 1e-6, tau_spec)
    render = jax.numpy.einsum('...l, ...lc, ...-> ...cl', reflected_energy, light_intensity, rho_spec)
    return render
