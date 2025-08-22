import jax
import open3d
import numpy
import ledgetter.utils.vector_tools as vector_tools
import ledgetter.optim.quadratics as quadratics

def get_intersection_point(t, d, intersection_time):
    point = t + d * jax.numpy.expand_dims(intersection_time, axis=-1)
    return point

def get_mesh_raycaster(mesh):
    """
    Creates a raycasting function for a given 3D mesh using Open3D and JAX.
    This function initializes a raycasting scene with the provided mesh and returns
    a raycaster function. The raycaster function computes intersections of rays with
    the mesh, returning information about the intersection points, normals, and a mask
    indicating valid intersections.
    Parameters:
        mesh (open3d.geometry.TriangleMesh): The 3D mesh to be used for raycasting.
    Returns:
        function: A raycaster function that takes the following inputs:
            - t (jax.numpy.ndarray): The origins of the rays, with shape (..., 3).
            - d (jax.numpy.ndarray): The directions of the rays, with shape (..., 3).
          The raycaster function returns:
            - mask (jax.numpy.ndarray): A boolean array indicating which rays intersect
              the mesh, with shape (...,).
            - normals (jax.numpy.ndarray): The normals at the intersection points, with
              shape (..., 3).
            - points (jax.numpy.ndarray): The intersection points in 3D space, with
              shape (..., 3).
    """

    scene = open3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    def o3d_render(rays):
        ans = scene.cast_rays(open3d.core.Tensor(rays))
        normals_world = ans['primitive_normals'].numpy()
        intersection_time = ans['t_hit'].numpy()
        return normals_world, intersection_time
    def raycaster(t, d):
        rays = jax.numpy.concatenate(jax.numpy.broadcast_arrays(t, d), axis=-1)
        out_type = (jax.ShapeDtypeStruct(rays.shape[:-1]+(3,), jax.numpy.float32), jax.ShapeDtypeStruct(rays.shape[:-1], jax.numpy.float32))
        normals, intersection_time = jax.pure_callback(lambda rays : o3d_render(numpy.asarray(rays)), out_type, rays)
        mask = intersection_time < jax.numpy.inf
        points = get_intersection_point(t, d, intersection_time)
        return mask, normals, points, intersection_time
    return raycaster

def get_sphere_raycaster(center, radius):
    def get_intersection_time(t, d):
        a = jax.numpy.square(vector_tools.norm_vector(d)[0])
        b = 2 * jax.numpy.einsum('...i, ...i -> ...', d, t - center)
        c = jax.numpy.square(vector_tools.norm_vector(t-center)[0]) -jax.numpy.square(radius)
        x1, x2 = quadratics.quadratic_roots(a, b, c)
        formated_x1 = jax.numpy.where(jax.numpy.logical_or(x1<0, jax.numpy.isnan(x1)), jax.numpy.inf, x1)
        formated_x2 = jax.numpy.where(jax.numpy.logical_or(x2<0, jax.numpy.isnan(x2)), jax.numpy.inf, x2)
        intersection_time = jax.numpy.minimum(formated_x1, formated_x2)
        return intersection_time
    def raycaster(t, d):
      intersection_time = get_intersection_time(t, d)
      mask = intersection_time < jax.numpy.inf
      points = get_intersection_point(t, d, intersection_time)
      normals = vector_tools.norm_vector(points-center)[1]
      return mask, normals, points, intersection_time
    return raycaster


def merge_raycasters(raycasters):
    def raycaster(t, d):
        mask, normals, points, intersection_time = raycasters[0](t,d)
        for raycaster in raycasters[1:]:
            new_mask, new_normals, new_points, new_intersection_time = raycaster(t,d)
            replace_mask = new_intersection_time < intersection_time
            mask, normals, points, intersection_time = jax.numpy.where(replace_mask, new_mask, mask), jax.numpy.where(replace_mask[..., None], new_normals, normals), jax.numpy.where(replace_mask[..., None], new_points, points), jax.numpy.where(replace_mask, new_intersection_time, intersection_time)
        return mask, normals, points, intersection_time
    return raycaster