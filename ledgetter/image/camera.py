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
    """Computes camera rays from pixel coordinates.

    Args:
        coordinates (Array ..., 2): Pixel coordinates.
        K (Array 3, 3): Camera intrinsic matrix.
        R (Array 3, 3): Camera rotation matrix.

    Returns:
        Array ..., 3: Camera rays in world coordinates.
    """
    homogeneous_coordinates = to_homogeneous(coordinates)
    inv_K = jax.numpy.linalg.inv(K)
    camera_rays = jax.numpy.einsum('ui, ik, ...k -> ...u', R, inv_K, homogeneous_coordinates)
    return camera_rays

def get_rototranslation_matrix(R, t):
    """Builds a 4x4 roto-translation matrix from rotation and translation.

    Args:
        R (Array 3, 3): Rotation matrix.
        t (Array 3,): Translation vector.

    Returns:
        Array 4, 4: Roto-translation matrix.
    """
    Rt = jax.numpy.block([[R, jax.numpy.expand_dims(t,axis=-1)],[jax.numpy.zeros((1,3)),jax.numpy.ones((1,1))]])
    return Rt

def build_geometry(t, d, normals_world, intersection_time, R):
    """Constructs 3D geometry from ray intersections.

    Args:
        t (Array 3,): Camera position.
        d (Array ..., 3): Ray directions.
        normals_world (Array ..., 3): Surface normals in world coordinates.
        intersection_time (Array ...): Distance along rays to intersection points.
        R (Array 3, 3): Camera rotation matrix.

    Returns:
        Tuple:
            - Array ...: Mask indicating valid intersections.
            - Array ..., 3: Surface normals in camera coordinates.
            - Array ..., 3: 3D points in camera coordinates.
    """
    Rt = get_rototranslation_matrix(R, t)
    mask = intersection_time < jax.numpy.inf
    points_world = t + d * jax.numpy.expand_dims(intersection_time, axis=-1)
    normals_map = jax.numpy.einsum('ik,...k->...i', R.T, normals_world)
    points_map = jax.numpy.einsum('ik,...k->...i', jax.numpy.linalg.inv(Rt), to_homogeneous(points_world))[...,:3]
    return mask, normals_map, points_map

def o3d_render(mesh, rays):
    """Performs raycasting on a 3D mesh using Open3D.

    Args:
        mesh (open3d.t.geometry.TriangleMesh): The input 3D mesh.
        rays (Array ..., 6): Ray origins and directions.

    Returns:
        Tuple:
            - Array ..., 3: Surface normals in world coordinates.
            - Array ...: Distance along rays to intersection points.
    """
    scene = open3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    ans = scene.cast_rays(open3d.core.Tensor(rays))
    normals_world = ans['primitive_normals'].numpy()
    intersection_time = ans['t_hit'].numpy()
    return normals_world, intersection_time

def get_geometry(mesh, K, R, t):
    """Creates a function that computes geometry from pixel coordinates.

    Args:
        mesh (open3d.t.geometry.TriangleMesh): The input 3D mesh.
        K (Array 3, 3): Camera intrinsic matrix.
        R (Array 3, 3): Camera rotation matrix.
        t (Array 3,): Camera position.

    Returns:
        Callable[[Array ..., 2], Tuple]:
            - Array ...: Mask indicating valid intersections.
            - Array ..., 3: Surface normals in camera coordinates.
            - Array ..., 3: 3D points in camera coordinates.
    """
    callback = lambda rays : o3d_render(mesh, numpy.asarray(rays))
    def geometry(coordinates):
        d = get_camera_rays(coordinates, K, R)
        rays = jax.numpy.concatenate(jax.numpy.broadcast_arrays(t, d), axis=-1)
        out_type = (jax.ShapeDtypeStruct(coordinates.shape[:-1]+(3,), jax.numpy.float32), jax.ShapeDtypeStruct(coordinates.shape[:-1], jax.numpy.float32))
        normals_world, intersection_time = jax.pure_callback(callback, out_type, rays)
        mask, normals_map, points_map = build_geometry(t, d, jax.numpy.asarray(normals_world), jax.numpy.asarray(intersection_time), R)
        return mask, normals_map, points_map
    return geometry


