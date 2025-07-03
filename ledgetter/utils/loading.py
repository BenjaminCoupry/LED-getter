import numpy
import jax
import tqdm
import rawpy
import os
import open3d
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
    format = pathlib.Path(path).suffix.lower()
    if format in {'.jpg', '.jpeg', '.png'}: #given a developed image
        image = jax.numpy.asarray(iio.imread(path)/255.0)
    elif format in {'.nef'}: #given a raw image
        with rawpy.imread(path) as raw:
            image = jax.numpy.asarray(raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True, gamma=(1,1), half_size=False, user_flip = 0)/(2**16-1))
    else:
        raise ValueError(f"Unknown image format: {format}")
    return image

def extract_pixels(image, pixels, pose=None, kernel_span=5, batch_size=100):
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
    if pose : #given undistorsion
        K, distorsion = jax.numpy.asarray(pose['K']), jax.numpy.asarray(pose['distorsion'])
        grid = undistort.get_undistorted_image(K, distorsion, jax.numpy.asarray(image), kernel_span)
    else: #without undistorsion
        grid = grids.get_grid_from_array(jax.numpy.swapaxes(image, 0, 1))
    undisto_image, mask = jax.lax.map(grid, pixels, batch_size=batch_size)
    return undisto_image, mask

def load_images(paths, pixels, pose=None, kernel_span=3, batch_size=100):
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
    stored = jax.numpy.empty(n_pix + (n_c, n_im), dtype=numpy.float32)
    for i in tqdm.tqdm(range(n_im), desc = 'loading'):
        image_path = paths[i]
        image = load_image(image_path)
        with jax.default_device(jax.devices("gpu")[0]):
            extracted, mask = extract_pixels(image, pixels, pose=pose, kernel_span=kernel_span, batch_size=batch_size)
        stored = stored.at[..., i].set(extracted)
    return stored, mask, (n_pix, n_im, n_c)

def load_mesh(path, transform):
    """
    Loads a 3D mesh from a given file path.

    Parameters:
    path (str): The file path to the mesh.

    Returns:
    open3d.t.geometry.TriangleMesh: The loaded and transformed 3D mesh.
    """
    mesh = open3d.t.io.read_triangle_mesh(path).transform(numpy.diag([1,-1,-1,1])).transform(numpy.asarray(transform))
    return mesh

def load_geometry(path, pixels, pose=None):
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
            with numpy.load(path) as loaded:
                mask_loaded = loaded['mask']
                if 'normalmap' in loaded:
                    normalmap_loaded = loaded['normalmap'].astype(numpy.float32)
                elif 'normals' in loaded:
                    normals_loaded = loaded['normals'].astype(numpy.float32)
                    normalmap_loaded = normals_loaded if normals_loaded.ndim == 3 else vector_tools.build_masked(mask_loaded, normals_loaded)
                else:
                    raise ValueError(f"No normals found in {path}")
                if 'pointmap' in loaded:
                    pointmap_loaded = loaded['pointmap'].astype(numpy.float32)
                elif 'points' in loaded:
                    points_loaded = loaded['points'].astype(numpy.float32)
                    pointmap_loaded = points_loaded if points_loaded.ndim == 3 else vector_tools.build_masked(mask_loaded, points_loaded)
                else:
                    raise ValueError(f"No points found in {path}")
        elif format in {'.png'}:
            normalmap_image = iio.imread(path)
            normalmap_loaded = vector_tools.rgb_to_r3(normalmap_image)*jax.numpy.asarray([1,-1,-1])
            normals_norm = vector_tools.norm_vector(normalmap_loaded)[0]
            mask_loaded = jax.numpy.logical_and(normals_norm>0.95, normals_norm<1.05)
            pointmap_loaded = jax.numpy.full(normalmap_loaded.shape, jax.numpy.nan, dtype=jax.numpy.float32)
        normalmap_grid, mask_grid, points_grid = grids.get_grid_from_array(jax.numpy.swapaxes(normalmap_loaded, 0, 1)), grids.get_grid_from_array(jax.numpy.swapaxes(mask_loaded, 0, 1)), grids.get_grid_from_array(jax.numpy.swapaxes(pointmap_loaded, 0, 1))
        geometry = lambda pixels : ((lambda mask, normalmap, points : (jax.numpy.logical_and(mask[0], mask[1]), normalmap[0], points[0]))(mask_grid(pixels), normalmap_grid(pixels), points_grid(pixels)))
        raycaster = None #TODO : raycaster from depthmap
        backend = 'cpu'
    elif format in {'.obj', '.ply'} or os.path.isdir(path):  #extracting geometry from a mesh
        mesh_path = meshroom.get_mesh_path(path) if os.path.isdir(path) else path #direct path or path to a meshroom project
        K, R, t = jax.numpy.asarray(pose['K']), jax.numpy.asarray(pose['R']), jax.numpy.asarray(pose['t'])
        transform = camera.get_rototranslation_matrix(R, t, to_camera=True)
        mesh = load_mesh(mesh_path, transform)
        raycaster = raycasting.get_mesh_raycaster(mesh)
        geometry = camera.get_geometry(raycaster, K)
        backend = None
    else:
        raise ValueError(f"Unknown geometry format: {format}")
    mask, normals, points = jax.jit(geometry, backend = backend)(pixels)
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
    if pathlib.Path(path).suffix.lower() in {'.json'}: #given pose.json
        with open(path, 'r') as f:
            pose = json.load(f)
    elif os.path.isdir(path) and aligned_image_path is not None: #given meshroom project
        sfm_path = meshroom.get_sfm_path(path)
        with open(sfm_path, 'r') as f:
            sfm = json.load(f)
        view_id = meshroom.get_view_id(sfm, aligned_image_path)
        pose = meshroom.get_pose(sfm, view_id)
    else:
        raise ValueError(f"Unknown pose format: {path}")
    return pose


            
def load_light(path, model=None, light_names=None):
    format = pathlib.Path(path).suffix.lower()
    if format in {'.jax'}:
        with open(path, "rb") as f:
            serialized = f.read()
        light = light_serialization.deserialize_light(serialized)
    elif format in {'.npz', '.lp'} and model is not None:
        light_values = load_light_values(path, light_names=light_names)
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

def load_light_values(path, light_names=None):
    format = pathlib.Path(path).suffix.lower()
    if format in {'.npz'}:
        with numpy.load(path) as light_archive:
            light_values = {k: jax.numpy.asarray(v) for k, v in light_archive.items() if k not in {'mask', 'validity_mask'}}
    elif format in {'.lp'} and light_names is not None:
        names_dir = numpy.loadtxt(path,skiprows=1,dtype=str,usecols=0)
        unsorted_dir = numpy.loadtxt(path,skiprows=1,dtype=float,usecols=(1,2,3))
        order_dir = numpy.asarray([numpy.argwhere(names_dir==n)[0,0] for n in light_names])
        light_directions_raw = unsorted_dir[order_dir,:]
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
        light_values = {'light_directions': light_directions, 'dir_light_power': dir_light_power}
    else:
        raise ValueError(f"Unknown light values format: {format} with {'known' if light_names is not None else 'unknown'} light names")
    return light_values


def load_light_dict(path, do_load_light_values = True, do_load_light=None, do_load_model=True, do_load_losses=True, light_names = None):
    if path is None:
        model_path, light_values_path, losses_path, light_path, lp_path = None, None, None, None, None
    elif os.path.isdir(path):
        model_path, light_values_path, losses_path, light_path, lp_path =\
            os.path.join(path or '', 'model.json'), os.path.join(path or '', 'values', 'values.npz'),\
            os.path.join(path or '', 'losses','losses.npz'), os.path.join(path or '', 'light','light_function.jax'), os.path.join(path or '', 'light','light_direction.lp')
    elif pathlib.Path(path).suffix.lower() in {'.lp'} and light_names is not None:
        model_path, light_values_path, losses_path, light_path, lp_path = path, path, path, path, path
    else:
        raise ValueError(f"Unknown light format: {path} with {'known' if light_names is not None else 'unknown'} light names")
    light_values = load_light_values(files.first_existing_file([light_values_path, lp_path]), light_names=light_names) if path and do_load_light_values and (os.path.isfile(light_values_path) or os.path.isfile(lp_path)) else {}
    model = load_model(files.first_existing_file([model_path, light_values_path, lp_path])) if path and do_load_model and (os.path.isfile(model_path) or os.path.isfile(light_values_path) or os.path.isfile(lp_path)) else None
    light = load_light(files.first_existing_file([light_path, light_values_path, lp_path]), model=model, light_names=light_names) if path and do_load_light and (os.path.isfile(light_path) or os.path.isfile(light_values_path)or os.path.isfile(lp_path)) else None
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
    


