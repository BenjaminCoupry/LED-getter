import numpy
import jax
import tqdm
import rawpy
import os
import open3d
import json
import pathlib
import ledgetter.image.undistort as undistort
import ledgetter.image.lanczos as lanczos
import ledgetter.image.camera as camera
import ledgetter.utils.meshroom as meshroom
import ledgetter.space.raycasting as raycasting
import imageio.v3 as iio
import jax.export
import ledgetter.utils.functions as functions
import ledgetter.rendering.models as models


def chunck_index(chunck, length):
    """
    Computes the slice corresponding to a given chunk index in a partitioned sequence.

    Parameters:
    chunck (tuple): A tuple containing the section index and the total number of sections.
    length (int): The total length of the sequence.

    Returns:
    slice: A slice object representing the range of indices for the given chunk.
    """
    section, n_sections = chunck
    n_each_section, extras = divmod(length, n_sections)
    section_sizes = ([0] +
                        extras * [n_each_section+1] +
                        (n_sections-extras) * [n_each_section])
    div_points = numpy.cumsum(section_sizes)
    chunck_slice =  slice(div_points[section], div_points[section+1])
    return chunck_slice, numpy.empty(length)[chunck_slice].shape[0]

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
    else:
        props = iio.improps(size)
        width, height = props.shape[1], props.shape[0]
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
    if pathlib.Path(path).suffix.lower() in {'.jpg', '.jpeg', '.png'}: #given a developed image
        image = jax.numpy.asarray(iio.imread(path)/255.0)
    else : #given a raw image
        with rawpy.imread(path) as raw:
            image = jax.numpy.asarray(raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True, gamma=(1,1), half_size=False, user_flip = 0)/(2**16-1))
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
        grid = lanczos.grid_from_array(jax.numpy.swapaxes(image, 0, 1))
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
    if pathlib.Path(path).suffix.lower() in {'.npz'}: #given .npz geometry
        with numpy.load(path) as loaded:
            normalmap_loaded, mask_loaded, points_loaded = loaded['normalmap'], loaded['mask'], loaded['points']
        normalmap_grid, mask_grid, points_grid = lanczos.grid_from_array(jax.numpy.swapaxes(normalmap_loaded, 0, 1)), lanczos.grid_from_array(jax.numpy.swapaxes(mask_loaded, 0, 1)), lanczos.grid_from_array(jax.numpy.swapaxes(points_loaded, 0, 1))
        geometry = lambda pixels : ((lambda mask, normalmap, points : (jax.numpy.logical_and(mask[0], mask[1]), normalmap[0], points[0]))(mask_grid(pixels), normalmap_grid(pixels), points_grid(pixels)))
        raycaster = None #TODO : raycaster from depthmap
    else : #extracting geometry from a mesh
        K, R, t = jax.numpy.asarray(pose['K']), jax.numpy.asarray(pose['R']), jax.numpy.asarray(pose['t'])
        transform = camera.get_rototranslation_matrix(R, t, to_camera=True)
        mesh_path = meshroom.get_mesh_path(path) if os.path.isdir(path) else path #direct path or path to a meshroom project
        mesh = load_mesh(mesh_path, transform)
        raycaster = raycasting.get_mesh_raycaster(mesh)
        geometry = camera.get_geometry(raycaster, K)
    mask, normals, points = jax.jit(geometry)(pixels)
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
    elif os.path.isdir(path) : #given meshroom project
        sfm_path = meshroom.get_sfm_path(path)
        with open(sfm_path, 'r') as f:
            sfm = json.load(f)
        view_id = meshroom.get_view_id(sfm, aligned_image_path) #
        pose = meshroom.get_pose(sfm, view_id)
    return pose


def load_light(path):
    if pathlib.Path(path).suffix.lower() in {'.jax'}:
        with open(path, "rb") as f:
            serialized = f.read()
        light = functions.filter_args(lambda points, pixels : jax.export.deserialize(serialized).call(points, pixels))
    elif pathlib.Path(path).suffix.lower() in {'.npz'}:
        with numpy.load(path) as light_archive:
            light_values = dict(light_archive)
        light_model = models.model_from_parameters(light_values, {})
        light_raw = models.get_light(light_model['light'])
        light = functions.filter_args(jax.jit(lambda points, pixels : light_raw(**(light_values | {'points':points, 'pixels':pixels}))))
    return light