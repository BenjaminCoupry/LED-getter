import jax
import open3d
import numpy

def to_homogeneous(v):
    """converts vectors to homogeneous coordinates

    Args:
        v (Array ..., dim): input vectors

    Returns:
        Array ..., dim+1: homogeneous coordinates of the input vectors
    """
    append_term = jax.numpy.ones(jax.numpy.shape(v)[:-1]+(1,))
    homogeneous = jax.numpy.append(v,append_term,axis=-1)
    return homogeneous

def get_camera_rays(coordinates, K, R):
    homogeneous_coordinates = to_homogeneous(coordinates)
    inv_K = jax.numpy.linalg.inv(K)
    camera_rays = jax.numpy.einsum('ui, ik, ...k -> ...u', R, inv_K, homogeneous_coordinates)
    return camera_rays

def get_rototranslation_matrix(R, t):
    Rt = jax.numpy.block([[R, jax.numpy.expand_dims(t,axis=-1)],[jax.numpy.zeros((1,3)),jax.numpy.ones((1,1))]])
    return Rt

def build_geometry(t, d, normals_world, intersection_time, R):
    Rt = get_rototranslation_matrix(R, t)
    mask = intersection_time < jax.numpy.inf
    points_world = t + d * jax.numpy.expand_dims(intersection_time, axis=-1)
    normals_map = jax.numpy.einsum('ik,...k->...i', R.T, normals_world)
    points_map = jax.numpy.einsum('ik,...k->...i', jax.numpy.linalg.inv(Rt), to_homogeneous(points_world))[...,:3]
    return mask, normals_map, points_map

def o3d_render(mesh, rays):
    scene = open3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    ans = scene.cast_rays(open3d.core.Tensor(rays))
    normals_world = ans['primitive_normals'].numpy()
    intersection_time = ans['t_hit'].numpy()
    return normals_world, intersection_time

def geometry_raycast(mesh, K, R, t, coordinates):
    d = get_camera_rays(coordinates, K, R)
    rays = jax.numpy.concatenate(jax.numpy.broadcast_arrays(t, d), axis=-1)
    normals_world, intersection_time = o3d_render(mesh, numpy.asarray(rays))
    geometry = build_geometry(t, d, jax.numpy.asarray(normals_world), jax.numpy.asarray(intersection_time), R)
    return geometry

