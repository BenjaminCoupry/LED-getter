import ledgetter.utils.files as files
import argparse

def update_light_dict(light_dict, name=None, losses_values=None, model=None, values=None, light=None):
    new_losses = light_dict['losses'] if (losses_values is None or name is None) else light_dict['losses'] + [(files.get_new_unique_name(list(map(lambda u : u[0], light_dict['losses'])), name), losses_values)]
    light_dict = {'model': light_dict['model'] if model is None else model,
                   'light_values': light_dict['light_values'] if values is None else values,
                     'light': light_dict['light'] if light is None else light,
                       'losses' : new_losses}
    return light_dict

def parse_merge_args():
    parser = argparse.ArgumentParser(description="Merge sliced values")
    parser.add_argument(
        '--backend', type=str, default='gpu', choices=['cpu', 'gpu'],
        help="Backend for heavy computations."
    )
    parser.add_argument(
        '--out_path', type=str, required=True,
        help="Output path for the merged values."
    )
    parser.add_argument(
        '--paths', nargs='+', type=str, required=True,
        help="Ordered paths to sliced values."
    )
    return parser.parse_args()


def parse_main_args():
    parser = argparse.ArgumentParser(description="Estimate light for photometric stereo")

    parser.add_argument(
        '--pattern', type=str, choices=['directional', 'rail', 'punctual', 'LED', 'specular', 'harmonic', 'grid', 'lambertian_harmonic', 'PS'], required=True,
        help="Lighting pattern type."
    )
    parser.add_argument(
        '--skip_export', type=str, nargs='+', choices=['images', 'lightmaps', 'values', 'losses', 'model', 'light', 'misc'],
        help="Exports to skip."
    )
    parser.add_argument(
        '--ps_images_paths', nargs='+', type=str, required=True, action='append',
        help="Photometric stereo images. Call multiple times to superpose images."
    )
    parser.add_argument(
        '--out_path', type=str, required=True,
        help="Output path for the estimated lighting."
    )
    parser.add_argument(
        '--step', type=int, default=1,
        help="Decimation of input dimension"
    )
    parser.add_argument(
        '--slice_i', type=int, default=0,
        help="Decimation shift"
    )
    parser.add_argument(
        '--meshroom_project', type=str, default=None,
        help="Path to the Meshroom project (optional)."
    )
    parser.add_argument(
        '--aligned_image_path', type=str, default=None,
        help="Path to the PS image aligned in the meshroom project (optional)."
    )
    parser.add_argument(
        '--geometry_path', type=str, default=None,
        help="Path to 3D mesh file (optional)."
    )
    parser.add_argument(
        '--pose_path', type=str, default=None,
        help="Path to camera pose as a JSON (optional)."
    )
    parser.add_argument(
        '--black_image_path', type=str, default=None,
        help="Path to black reference image (optional)."
    )
    parser.add_argument(
        '--loaded_light_folder', type=str, default=None,
        help="Path to another light estimation to use as a start (optional)."
    )
    parser.add_argument(
        '--delta', type=float, default=0.01,
        help="Huber loss parameter."
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help="Learning rate for optimization."
    )
    parser.add_argument(
        '--tqdm_refresh', type=int, default=0,
        help="Refresh rate for tqdm progress bar."
    )
    parser.add_argument(
        '--iterations', type=int, default=10000,
        help="Number of descent steps."
    )
    parser.add_argument(
        '--ps_chunck_number', type=int, default=100,
        help="Number of chuncks for PS treatement."
    )
    parser.add_argument(
        '--backend', type=str, default='gpu', choices=['cpu', 'gpu'],
        help="Backend for heavy computations."
    )

    parser.add_argument(
        '--pixel_step', type=int, default=None,
        help="Size of pixel step for grid light estimation."
    )

    parser.add_argument(
        '--spheres_to_load', nargs='+', type=str, default=None,
        help="Names of the spheres to load when loading sphere geometry, default is all."
    )

    parser.add_argument(
        '--flip_lp', action='store_true', help='Flip coordinates (y,z) when loading light from a .lp'
        )
    
    parser.add_argument(
        '--not_flip_mesh', action='store_true', help='Do not flip coordinates (y,z) when loading a 3D mesh'
        )
    
    parser.add_argument(
        '--not_apply_geometry_images_undisto', action='store_true', help='Do not apply undisto when loading geometry from images or npz'
        )

    parser.add_argument(
        '--not_apply_images_undisto', action='store_true', help='Do not apply undisto when loading images'
        )
    
    parser.add_argument(
        '--remove_image_gamma', action='store_true', help='Remove the gamma correction when loading a developped image'
        )


    return parser.parse_args()