import numpy
import jax
import tqdm
import rawpy
import ledgetter.image.undistort as undistort
import ledgetter.image.camera as camera


def load_raw_images(raw_paths, pose, pixels, kernel_span=5, batch_size=100):
    K, distorsion = jax.numpy.asarray(pose['K']), jax.numpy.asarray(pose['distorsion'])
    n_pix, n_im, n_c = pixels.shape[:-1], len(raw_paths), 3
    stored = numpy.empty(n_pix + (n_c, n_im), dtype=numpy.float32)
    for i in tqdm.tqdm(range(n_im), desc = 'loading'):
        image_path = raw_paths[i]
        with rawpy.imread(image_path) as raw:
            image = numpy.asarray(raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True, gamma=(1,1), half_size=False, user_flip = 0)/(2**16-1))
        def load_np(image):
            undistorted_grid = undistort.get_undistorted_image(K, distorsion, jax.numpy.asarray(image), kernel_span)
            undisto_image, padded = jax.lax.map(undistorted_grid, pixels, batch_size=batch_size)
            return numpy.asarray(undisto_image), numpy.asarray(padded == 0)
        stored[..., i], mask = load_np(image)
    return stored, mask, (n_pix, n_im, n_c)

def load_geometry(mesh, pose, pixels):
    K, R, t = jax.numpy.asarray(pose['K']), jax.numpy.asarray(pose['R']), jax.numpy.asarray(pose['t'])
    geometry = camera.geometry_raycast(mesh, K, R, t, pixels)
    mask, normals_map, points_map = geometry
    return numpy.asarray(mask, dtype=bool), numpy.asarray(normals_map), numpy.asarray(points_map)

def load_light(light_path, specular=False):
    with numpy.load(light_path) as loaded:
        pixels_light, rho_light, light_locations, light_power, light_principal_direction, mu = loaded['pixels'], loaded['rho'], loaded['light_locations'], loaded['light_power'], loaded['light_principal_direction'], loaded['mu']
        if specular:
            rho_spec, tau_spec = loaded['tau_spec'], loaded['rho_spec']
    if not specular:
        return pixels_light, rho_light, light_locations, light_power, light_principal_direction, mu
    else:
        return pixels_light, rho_light, light_locations, light_power, light_principal_direction, mu, rho_spec, tau_spec

def chunck_index(chunck, length):
    section, n_sections = chunck
    n_each_section, extras = divmod(length, n_sections)
    section_sizes = ([0] +
                        extras * [n_each_section+1] +
                        (n_sections-extras) * [n_each_section])
    div_points = numpy.cumsum(section_sizes)
    chunck_slice =  slice(div_points[section], div_points[section+1])
    return chunck_slice

def get_pixelmap(pose):
    width, height = int(pose['width']), int(pose['height'])
    width_range, height_range = jax.numpy.arange(0,width),jax.numpy.arange(0,height)
    coordinates = jax.numpy.stack(jax.numpy.meshgrid(width_range,height_range),axis=-1)
    return coordinates