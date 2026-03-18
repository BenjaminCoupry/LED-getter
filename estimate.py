import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

import pipeline.common as common
import pipeline.preprocessing as preprocessing
import pipeline.outputs as outputs
import pipeline.light_estimation as light_estimation
import pipeline.ps_estimation as ps_estimation
import ledgetter.utils.chuncks as chuncks
import ledgetter.utils.loading as loading
import jax
import itertools

def preprocess(args, sliced):
    added_values = {} if args.pixel_step is None else {'pixel_step': args.pixel_step}
    values, images, mask, raycaster, shapes, full_shape, output, optimizer, scale, light_dict, light_names, pose =\
        preprocessing.preprocess(
            list(zip(*args.ps_images_paths)),
            sliced=sliced,
            meshroom_project=args.meshroom_project,
            aligned_image_path=args.aligned_image_path,
            geometry_path=args.geometry_path,
            pose_path=args.pose_path,
            black_image_path=args.black_image_path,
            loaded_light_folder=args.loaded_light_folder,
            load_light_function=True,
            learning_rate=args.learning_rate,
            tqdm_refresh=args.tqdm_refresh,
            added_values=added_values,
            flip_lp=args.flip_lp,
            flip_mesh= (not args.not_flip_mesh),
            apply_geometry_images_undisto = (not args.not_apply_geometry_images_undisto),
            apply_images_undisto= (not args.not_apply_images_undisto),
            spheres_to_load=args.spheres_to_load,
            remove_image_gamma = args.remove_image_gamma
        )
    return values, images, mask, raycaster, shapes, full_shape, output, optimizer, scale, light_dict, light_names, pose

def full_slices_ps(args):
    chuncker, _ = chuncks.get_chuncker((args.step, args.step))
    values_paths = []
    for slice_i, sliced in enumerate(chuncker):
        print(f"Computing slice {slice_i:05d}")
        values, images, mask, raycaster, shapes, full_shape, output, optimizer, scale, light_dict, light_names, pose = preprocess(args, sliced)
        return_ps_only = all(x in args.skip_export for x in {'images', 'lightmaps', 'light', 'misc'})
        light_dict, validity_mask = ps_estimation.estimate_ps(args.iterations, values, images, mask, raycaster, shapes, output, optimizer, scale, light_dict, delta=args.delta, chunck_number = args.ps_chunck_number, return_ps_only=return_ps_only, backend=args.backend)
        out_path = os.path.join(args.out_path, 'full_slices_PS', f'slice_{slice_i:05d}')
        values_paths.append(os.path.join(out_path, 'values', 'values.npz'))
        outputs.export_results(out_path, validity_mask, light_dict, mask, images, light_names, skip = args.skip_export, pose=pose)
        del values, images, mask, raycaster, shapes, full_shape, output, optimizer, scale, light_dict, light_names, pose, validity_mask
    print("Merging slices")
    values, full_mask = loading.load_chuncked_values(values_paths)
    out_path = os.path.join(args.out_path, 'full_slices_PS' ,'merged')
    light_dict = {'light_values': {k: v for k, v in values.items() if k not in {'validity_mask'}}}
    outputs.export_values(os.path.join(args.out_path,'values'), light_dict, full_mask, values['validity_mask'])

def estimation(args):
    chuncker, _ = chuncks.get_chuncker((args.step, args.step))
    sliced = next(itertools.islice(chuncker, args.slice_i, args.slice_i + 1, None))
    values, images, mask, raycaster, shapes, full_shape, output, optimizer, scale, light_dict, light_names, pose = preprocess(args, sliced)
    for pattern in args.pattern:
        print(f"Computing pattern {pattern}")
        if pattern=='PS':
            return_ps_only = all(x in args.skip_export for x in {'images', 'lightmaps', 'light', 'misc'})
            light_dict, validity_mask = ps_estimation.estimate_ps(args.iterations, values, images, mask, raycaster, shapes, output, optimizer, scale, light_dict, delta=args.delta, chunck_number = args.ps_chunck_number, return_ps_only=return_ps_only, backend=args.backend)
        else:
            light_dict, validity_mask = light_estimation.estimate_light(args.iterations, pattern, values, images, mask, raycaster, shapes, output, optimizer, scale, light_dict, delta=args.delta)

        out_path = os.path.join(args.out_path, pattern, f'slice_{args.slice_i:05d}')
        outputs.export_results(out_path, validity_mask, light_dict, mask, images, light_names, skip = args.skip_export, pose=pose)

def main():
    args = common.parse_main_args()
    is_full_slices_ps = (len(args.pattern) ==1) and (args.pattern[0]=='PS') and (args.slice_i == -1)
    with jax.default_device(jax.devices(args.backend)[0]):
        if is_full_slices_ps:
            full_slices_ps(args)
        else:
            estimation(args)


if __name__ == "__main__":
    main()