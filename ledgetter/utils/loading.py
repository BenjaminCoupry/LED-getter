import numpy
import jax
import tqdm
import rawpy
import os
import open3d
import yaml
import json
import pathlib
import ledgetter.image.undistort as undistort
import ledgetter.image.grids as grids
import ledgetter.image.camera as camera
import ledgetter.utils.meshroom as meshroom
import ledgetter.space.raycasting as raycasting
import imageio.v3 as iio
import ledgetter.utils.files as files
import ledgetter.utils.light_serialization as light_serialization
import ledgetter.utils.functions as functions
import ledgetter.models.models as models
import ledgetter.utils.vector_tools as vector_tools
import ledgetter.image.grids as grids
import ledgetter.utils.chuncks as chuncks


def get_pixelmap(size):
    """
    Generates a pixel coordinate map for an image of given dimensions.

    Parameters:
    size (dict | str): A dictionary containing image 'width' and 'height' or a path to image.

    Returns:
    jax.numpy.ndarray: An array of shape (height, width, 2) representing pixel coordinates.
    """
    if type(size) is dict :
        width, height = int(size['width']), int(size['height'])
    elif type(size) is str :
        format = pathlib.Path(size).suffix.lower()
        if format in {'.jpg', '.jpeg', '.png', '.nef'}:
            props = iio.improps(size)
            width, height = props.shape[1], props.shape[0]
        else:
            raise ValueError(f"Unknown size format: {format}")
    else:
        raise ValueError(f"Unknown size type: {type(size)}")
    width_range, height_range = jax.numpy.arange(0,width),jax.numpy.arange(0,height)
    coordinates = jax.numpy.stack(jax.numpy.meshgrid(width_range,height_range),axis=-1)
    return coordinates

def load_image(path):
    """
    Loads an image from a given file path, supporting both developed and raw images.

    Parameters:
    path (str): The file path to the image.

    Returns:
    numpy.ndarray: A floating-point image array normalized to [0,1].
    """
    if type(path) is str:
        format = pathlib.Path(path).suffix.lower()
        if format in {'.jpg', '.jpeg', '.png'}: #given a developed image
            image = jax.numpy.asarray(iio.imread(path)/255.0)
        elif format in {'.nef'}: #given a raw image
            with rawpy.imread(path) as raw:
                image = jax.numpy.asarray(raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True, gamma=(1,1), half_size=False, user_flip = 0)/(2**16-1))
        else:
            raise ValueError(f"Unknown image format: {format}")
    elif type(path) is list:
        image = load_image(path[0])
        for p in path[1:]:
            image = jax.numpy.maximum(image, load_image(p))
    else:
        raise ValueError(f"Unknown image path type: {type(path)}")
    return image

def extract_pixels(image, pixels, pose=None, kernel_span=5, batch_size=100, mask=None, apply_undisto=True):
    """
    Extracts pixel values from an image, with optional undistortion.

    Parameters:
    image (numpy.ndarray): The input image.
    pixels (jax.numpy.ndarray): The pixel coordinates to extract.
    pose (dict, optional): Camera pose information for undistortion.
    kernel_span (int, optional): Kernel span for undistortion. Defaults to 5.
    batch_size (int, optional): Batch size for processing. Defaults to 100.

    Returns:
    tuple: A tuple containing the extracted pixel values and a mask indicating valid pixels, and the shapes of the problem.
    """
    if pose and pose['distorsion'] is not None and apply_undisto: #given undistorsion
        K, distorsion = jax.numpy.asarray(pose['K']), jax.numpy.asarray(pose['distorsion'])
        undisto_image, mask = jax.device_put(jax.lax.map(lambda coordinates : undistort.undistorted_image(K, distorsion, image, coordinates, kernel_span, mask=mask), pixels, batch_size=batch_size), pixels.device)
    else: #without undistorsion
        grid = grids.get_grid_from_array(jax.numpy.swapaxes(image, 0, 1), valid_mask = (jax.numpy.swapaxes(mask, 0, 1) if mask is not None else None))
        undisto_image, mask = grid(pixels)
    return undisto_image, mask

def load_images(paths, pixels, pose=None, kernel_span=3, batch_size=100, apply_undisto=True):
    """
    Loads multiple images and extracts pixel values from them.

    Parameters:
    paths (list of str): List of image file paths.
    pixels (jax.numpy.ndarray): The pixel coordinates to extract.
    pose (dict, optional): Camera pose information for undistortion.
    kernel_span (int, optional): Kernel span for undistortion. Defaults to 5.
    batch_size (int, optional): Batch size for processing. Defaults to 100.

    Returns:
    tuple: A tuple containing the extracted pixel values, a mask, and the shape metadata.
    """
    n_pix, n_im, n_c = pixels.shape[:-1], len(paths), 3
    stored = jax.numpy.empty(n_pix + (n_c, n_im), dtype=jax.numpy.float32)
    masks = jax.numpy.empty(n_pix + (n_im,), dtype=jax.numpy.bool)
    for i in tqdm.tqdm(range(n_im), desc = 'loading'):
        image_path = paths[i]
        image = load_image(image_path)
        extracted, mask_i = extract_pixels(image, pixels, pose=pose, kernel_span=kernel_span, batch_size=batch_size, apply_undisto=apply_undisto)
        stored = stored.at[..., i].set(extracted)
        masks = masks.at[..., i].set(mask_i)
    mask = jax.numpy.all(masks, axis=-1)
    return stored, mask, (n_pix, n_im, n_c)

def load_mesh(path, transform=None, flip_mesh=True):
    """
    Loads a 3D mesh from a given file path.

    Parameters:
    path (str): The file path to the mesh.

    Returns:
    open3d.t.geometry.TriangleMesh: The loaded and transformed 3D mesh.
    """
    flip = jax.numpy.diag(jax.numpy.asarray([1, -1, -1, 1])) if flip_mesh else jax.numpy.eye(4)
    transform = jax.numpy.eye(4) if transform is None else transform
    loaded_mesh = open3d.t.io.read_triangle_mesh(path)
    fliped_mesh = loaded_mesh.transform(numpy.asarray(flip))
    mesh = fliped_mesh.transform(numpy.asarray(transform))
    return mesh

def load_mesh_geometry(path, pixels, pose, flip_mesh=True):
    format = pathlib.Path(path).suffix.lower()
    if format in {'.obj', '.ply'}:
        K = jax.numpy.asarray(pose['K'])
        if pose['R'] is not None and pose['t'] is not None:
            R, t =  jax.numpy.asarray(pose['R']), jax.numpy.asarray(pose['t'])
            transform = camera.get_rototranslation_matrix(R, t, to_camera=True)
        else:
            transform=None
        mesh = load_mesh(path, transform=transform, flip_mesh=flip_mesh)
        raycaster = raycasting.get_mesh_raycaster(mesh)
        geometry = jax.jit(camera.get_geometry(raycaster, K), backend='cpu')
        mask, normals, points = jax.device_put(geometry(pixels), device=pixels.device)
    else:
        raise ValueError(f"Unknown mesh format: {format}")
    return mask, normals, points, raycaster

def load_npz_geometry(path, pixels, pose=None, batch_size=100, apply_undisto=True):
    format = pathlib.Path(path).suffix.lower()
    if format in {'.npz'}:
        with numpy.load(path) as loaded:
            normals_key = 'normals' if 'normals' in loaded else ('normalmap' if 'normalmap' in loaded else None)
            points_key = 'points' if 'points' in loaded else ('pointmap' if 'pointmap' in loaded else None)
            mask_loaded = jax.numpy.asarray(loaded['mask'], dtype=jax.numpy.bool)
            if points_key is not None:
                points_loaded = jax.numpy.asarray(loaded[points_key], dtype=jax.numpy.float32)
                pointmap_loaded = points_loaded if points_loaded.ndim == 3 else vector_tools.build_masked(mask_loaded, points_loaded)
            else:
                pointmap_loaded = jax.numpy.full(mask_loaded.shape + (3,), jax.numpy.nan, dtype=jax.numpy.float32)
            if normals_key is not None:
                normals_loaded = jax.numpy.asarray(loaded[normals_key], dtype=jax.numpy.float32)
                normalmap_loaded = normals_loaded if normals_loaded.ndim == 3 else vector_tools.build_masked(mask_loaded, normals_loaded)
            else:
                normalmap_loaded = jax.numpy.full(mask_loaded.shape + (3,), jax.numpy.nan, dtype=jax.numpy.float32)
        raycaster=None
        points, mask_points = extract_pixels(pointmap_loaded, pixels, pose=pose, kernel_span=5, batch_size=batch_size, mask=mask_loaded, apply_undisto=apply_undisto)
        normals_unorm, mask_normals = extract_pixels(normalmap_loaded, pixels, pose=pose, kernel_span=5, batch_size=batch_size, mask=mask_loaded, apply_undisto=apply_undisto)
        normals = vector_tools.norm_vector(normals_unorm)[1]
        mask = jax.numpy.logical_and(mask_points, mask_normals)
    else:
        raise ValueError(f"Unknown geometry format: {format}")
    return mask, normals, points, raycaster

def load_image_geometry(path, pixels, pose=None, batch_size=100, apply_undisto=True):
    format = pathlib.Path(path).suffix.lower()
    if format in {'.png'}:
        normalmap_image = load_image(path)
        normalmap_loaded = vector_tools.rgb_to_r3(normalmap_image*255.0)*jax.numpy.asarray([1,-1,-1])
        normals_norm = vector_tools.norm_vector(normalmap_loaded)[0]
        mask_normals = jax.numpy.logical_and(normals_norm>0.95, normals_norm<1.05)
        mask_path = pathlib.Path(path).with_name("mask.png")
        if os.path.isfile(mask_path):
            mask_loaded_raw = jax.numpy.asarray(iio.imread(mask_path)) > 0
            mask_loaded = mask_loaded_raw if jax.numpy.ndim(mask_loaded_raw) == 2 else jax.numpy.any(mask_loaded_raw, axis=-1)
            mask_data = jax.numpy.logical_and(mask_normals, mask_loaded)
        else:
            mask_data = mask_normals
        raycaster=None
        normals_unorm, mask = extract_pixels(normalmap_loaded, pixels, pose=pose, kernel_span=5, batch_size=batch_size, mask=mask_data, apply_undisto=apply_undisto)
        normals = vector_tools.norm_vector(normals_unorm)[1]
        points = jax.numpy.full(normals.shape, jax.numpy.nan, dtype=jax.numpy.float32)
    else:
        raise ValueError(f"Unknown image format: {format}")
    return mask, normals, points, raycaster

def load_sphere_geometry(path, pixels, pose):
    format = pathlib.Path(path).suffix.lower()
    if format in {'.yaml', '.json'}:
        with open(path, 'r') as f:
            if format in {'.yaml'}:
                sphere_dict= yaml.safe_load(f)
            elif format in {'.json'}:
                sphere_dict = json.load(f)
        spheres = list(map(lambda ob : sphere_dict['objects'][ob]['sphere'], sphere_dict['objects']))
        centers_world = jax.numpy.asarray(list(map(lambda s : [s['center']['x'], s['center']['y'], s['center']['z']], spheres)))
        radii = jax.numpy.asarray(jax.numpy.asarray(list(map(lambda s : s['radius'], spheres))))
        K = jax.numpy.asarray(pose['K'])
        if pose['R'] is not None and pose['t'] is not None:
            R, t =  jax.numpy.asarray(pose['R']), jax.numpy.asarray(pose['t'])
            transform = camera.get_rototranslation_matrix(R, t, to_camera=True)
        else:
            transform=None
        centers = camera.apply_transform(transform, centers_world)
        raycasters = list(map(lambda c, r : raycasting.get_sphere_raycaster(c, r), jax.numpy.unstack(centers, axis=0), jax.numpy.unstack(radii, axis=0)))
        raycaster = raycasting.merge_raycasters(raycasters)
        geometry = jax.jit(camera.get_geometry(raycaster, K), backend='cpu')
        mask, normals, points = jax.device_put(geometry(pixels), device=pixels.device)
    else:
        raise ValueError(f"Unknown sphere format: {format}")
    return mask, normals, points, raycaster

def load_geometry(path, pixels, pose=None, flip_mesh=True, batch_size=100, apply_undisto=True):
    """
    Loads 3D geometry data from a file or extracts it from a mesh.

    Parameters:
    path (str): The file path to the geometry, a path to a mesh or a Meshroom project directory.
    pixels (jax.numpy.ndarray): The pixel coordinates to extract geometry for.
    pose (dict, optional): Camera pose information for projection.

    Returns:
    tuple: A tuple containing a mask, normal vectors, and 3D points.
    """
    
    format = pathlib.Path(path).suffix.lower()
    if format in {'.npz', '.png'}: #given .npz geometry
        if format in {'.npz'}:
            mask, normals, points, raycaster = load_npz_geometry(path, pixels, pose=pose, batch_size=batch_size, apply_undisto=apply_undisto)
        elif format in {'.png'}:
            mask, normals, points, raycaster = load_image_geometry(path, pixels, pose=pose, batch_size=batch_size, apply_undisto=apply_undisto)
    elif format in {'.obj', '.ply'} or os.path.isdir(path):  #extracting geometry from a mesh
        mesh_path = meshroom.get_mesh_path(path) if os.path.isdir(path) else path #direct path or path to a meshroom project
        mask, normals, points, raycaster = load_mesh_geometry(mesh_path, pixels, pose, flip_mesh=flip_mesh)
    elif format in {'.yaml', '.json'}:
        mask, normals, points, raycaster = load_sphere_geometry(path, pixels, pose=pose)
    else:
        raise ValueError(f"Unknown geometry format: {format}")
    return mask, normals, points, raycaster

def load_pose(path, aligned_image_path=None):
    """
    Loads a camera pose from a json file or a Meshroom project.

    Parameters:
    path (str): The file path to the pose data or a Meshroom project directory.
    aligned_image_path (str, optional): The image path used to locate the corresponding pose,
        can be a path to an image or a list

    Returns:
    dict: A dictionary containing camera pose parameters.
    """
    format = pathlib.Path(path).suffix.lower()
    if format in {'.json', '.yaml'}: #given pose.json
        with open(path, 'r') as f:
            if format in {'.yaml'}:
                pose_dict= yaml.safe_load(f)
            elif format in {'.json'}:
                pose_dict = json.load(f)
        if 'K' in pose_dict:
            K = jax.numpy.asarray(pose_dict['K'])
        else:
            K = camera.build_K_matrix(pose_dict['camera']['focal'], pose_dict['camera']['principal_point']['x'], pose_dict['camera']['principal_point']['y'])
        R = jax.numpy.asarray(pose_dict['R']) if 'R' in pose_dict else None
        t = jax.numpy.asarray(pose_dict['t']) if 't' in pose_dict else None
        width = int(pose_dict['width']) if 'width' in pose_dict else None
        height = int(pose_dict['height']) if 'height' in pose_dict else None
        distorsion = jax.numpy.asarray(pose_dict['distorsion']) if 'distorsion' in pose_dict else None
        pose = {'K':K, 'R':R, 't':t, 'width':width, 'height':height, 'distorsion':distorsion}
    elif os.path.isdir(path) and aligned_image_path is not None: #given meshroom project
        sfm_path = meshroom.get_sfm_path(path)
        with open(sfm_path, 'r') as f:
            sfm = json.load(f)
        view_id = meshroom.get_view_id(sfm, aligned_image_path)
        pose = meshroom.get_pose(sfm, view_id)
    else:
        pose=None
    return pose


            
def load_light(path, model=None, light_names=None, flip_lp=False):
    format = pathlib.Path(path).suffix.lower()
    if format in {'.jax'}:
        with open(path, "rb") as f:
            serialized = f.read()
        light = light_serialization.deserialize_light(serialized)
    elif format in {'.npz', '.lp'} and model is not None:
        light_values = load_light_values(path, light_names=light_names, flip_lp=flip_lp)
        light_raw = models.get_light(model['light'])
        light = functions.filter_args(jax.jit(lambda points, pixels : light_raw(**(light_values | {'points':points, 'pixels':pixels}))))
    else:
        raise ValueError(f"Unknown light format: {format}, with {'known' if model is not None else 'unknown'} model and {'known' if light_names is not None else 'unknown'} light names")
    return light


def load_losses(path):
    format = pathlib.Path(path).suffix.lower()
    if format in {'.npz'}:
        with numpy.load(path, allow_pickle=True) as losses_archive:
            losses_values = dict(losses_archive)
            losses = [(name, losses_values[name]) for name in losses_values['steps_order']]
    else:
        raise ValueError(f"Unknown losses format: {format}")
    return losses

def load_model(path):
    format = pathlib.Path(path).suffix.lower()
    if format in {'.json'}:
        with open(path, 'r') as f:
            model_dict = json.load(f)
        model = {'light': model_dict['light'], 'renderers': model_dict['renderers'], 'parameters': set(model_dict['parameters']), 'data':   set(model_dict['data'])}
    elif format in {'.npz'}:
        with numpy.load(path) as light_archive:
            light_values_keys = set(light_archive.keys())
        model = models.model_from_parameters(set(), light_values_keys)
    elif format in {'.lp'}:
        model = {'light': 'directional', 'renderers': [], 'parameters': {'light_directions', 'dir_light_power'}, 'data':  set()}
    else:
        raise ValueError(f"Unknown model format: {format}")
    return model


def load_lp(path, light_names, flip_lp = False):
    format = pathlib.Path(path).suffix.lower()
    if format in {'.lp'}:
        names_dir = numpy.loadtxt(path,skiprows=1,dtype=str,usecols=0)
        unsorted_dir = numpy.loadtxt(path,skiprows=1,dtype=float,usecols=(1,2,3))
        order_dir = numpy.asarray([numpy.argwhere(names_dir==n)[0,0] for n in light_names])
        light_directions_raw = unsorted_dir[order_dir,:] * jax.numpy.asarray([1, 1, 1] if not flip_lp else [1, -1, -1])
        light_intensity_path = pathlib.Path(path).with_name("light_intensity.lp")
        if os.path.isfile(light_intensity_path):
            names_int = numpy.loadtxt(light_intensity_path,skiprows=1,dtype=str,usecols=0)
            unsorted_int = numpy.loadtxt(light_intensity_path,skiprows=1,dtype=float,usecols=1)
            order_int = numpy.asarray([numpy.argwhere(names_int==n)[0,0] for n in light_names])
            dir_light_power = unsorted_int[order_int]
        else:
            dir_light_power = None
        light_directions_norm, light_directions = vector_tools.norm_vector(light_directions_raw)
        dir_light_power = dir_light_power if dir_light_power is not None else light_directions_norm
    else:
        raise ValueError(f"Unknown lp format: {format}")
    return light_directions, dir_light_power


def load_light_values(path, light_names=None, flip_lp=False):
    format = pathlib.Path(path).suffix.lower()
    if format in {'.npz'}:
        with numpy.load(path) as light_archive:
            light_values = {k: jax.numpy.asarray(v) for k, v in light_archive.items() if k not in {'mask', 'validity_mask'}}
    elif format in {'.lp'} and light_names is not None:
        light_directions, dir_light_power = load_lp(path, light_names, flip_lp = flip_lp)
        light_values = {'light_directions': light_directions, 'dir_light_power': dir_light_power}
    else:
        raise ValueError(f"Unknown light values format: {format} with {'known' if light_names is not None else 'unknown'} light names")
    return light_values


def load_light_dict(path, do_load_light_values = True, do_load_light=None, do_load_model=True, do_load_losses=True, light_names = None, flip_lp=False):
    if path is None:
        model_path, light_values_path, losses_path, light_path, lp_path = None, None, None, None, None
    elif os.path.isdir(path):
        model_path, light_values_path, losses_path, light_path, lp_path =\
            os.path.join(path or '', 'model.json'), os.path.join(path or '', 'values', 'values.npz'),\
            os.path.join(path or '', 'losses','losses.npz'), os.path.join(path or '', 'light','light_function.jax'), os.path.join(path or '', 'light','light_direction.lp')
    elif pathlib.Path(path).suffix.lower() in {'.lp'} and light_names is not None:
        model_path, light_values_path, losses_path, light_path, lp_path = path, path, None, path, path
    else:
        raise ValueError(f"Unknown light format: {path} with {'known' if light_names is not None else 'unknown'} light names")
    light_values = load_light_values(files.first_existing_file([light_values_path, lp_path]), light_names=light_names, flip_lp=flip_lp) if path and do_load_light_values and (os.path.isfile(light_values_path) or os.path.isfile(lp_path)) else {}
    model = load_model(files.first_existing_file([model_path, light_values_path, lp_path])) if path and do_load_model and (os.path.isfile(model_path) or os.path.isfile(light_values_path) or os.path.isfile(lp_path)) else None
    light = load_light(files.first_existing_file([light_path, light_values_path, lp_path]), model=model, light_names=light_names, flip_lp=flip_lp) if path and do_load_light and (os.path.isfile(light_path) or os.path.isfile(light_values_path)or os.path.isfile(lp_path)) else None
    losses = load_losses(losses_path) if path and do_load_losses and losses_path is not None and os.path.isfile(losses_path) else []
    light_dict = {'model': model, 'light_values': light_values, 'light': light, 'losses': losses}
    return light_dict

def load_chuncked_values(paths):
    step = int(jax.numpy.sqrt(len(paths)))
    chuncker, _ = chuncks.get_chuncker((step, step))
    value_dicts, masks, atomic_values = [], [], None
    for i, path in enumerate(paths):
        with numpy.load(path) as values_archive:
            value_dict = {k : jax.numpy.asarray(values_archive[k]) for k in values_archive if models.is_pixelwise(k)}
            mask = jax.numpy.asarray(values_archive['mask'])
            value_dicts.append(value_dict)
            masks.append(mask)
            if i == len(paths)-1:
                atomic_values = {k : jax.numpy.asarray(values_archive[k]) for k in values_archive if (k not in {'mask'} and not models.is_pixelwise(k))}
                ref = value_dict
    full_shape = (sum(mask.shape[0] for mask in masks[::step]), sum(mask.shape[1] for mask in masks[:step]))
    ref_shapes = {k : jax.ShapeDtypeStruct(full_shape + jax.numpy.shape(ref[k])[1:], jax.numpy.dtype(ref[k])) for k in ref}
    full_values = {k : jax.numpy.empty(ref_shapes[k].shape, ref_shapes[k].dtype) for k in ref}
    full_mask = jax.numpy.zeros(full_shape, dtype=bool)
    for value_dict, mask, chunck in zip(value_dicts, masks, chuncker):
        full_values = {k : full_values[k].at[chunck].set(vector_tools.build_masked(mask, value_dict[k])) for k in ref}
        full_mask = full_mask.at[chunck].set(mask)
    values = atomic_values | {k : full_values[k][full_mask] for k in ref}
    return values, full_mask
    


