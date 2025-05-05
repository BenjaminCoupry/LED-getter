import jax
import open3d
import numpy

def get_mesh_raycaster(mesh):
    def o3d_render(rays):
        scene = open3d.t.geometry.RaycastingScene(nthreads=15)
        scene.add_triangles(mesh)
        ans = scene.cast_rays(open3d.core.Tensor(rays))
        normals_world = ans['primitive_normals'].numpy()
        intersection_time = ans['t_hit'].numpy()
        return normals_world, intersection_time
    def raycaster(t, d):
        rays = jax.numpy.concatenate(jax.numpy.broadcast_arrays(t, d), axis=-1)
        out_type = (jax.ShapeDtypeStruct(rays.shape[:-1]+(3,), jax.numpy.float32), jax.ShapeDtypeStruct(rays.shape[:-1], jax.numpy.float32))
        normals, intersection_time = jax.pure_callback(lambda rays : o3d_render(numpy.asarray(rays)), out_type, rays)
        mask = intersection_time < jax.numpy.inf
        points = t + d * jax.numpy.expand_dims(intersection_time, axis=-1)
        return mask, normals, points
    return raycaster