import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"


import pipeline.common as common
import pipeline.preprocessing as preprocessing
import pipeline.outputs as outputs
import pipeline.light_estimation as light_estimation
import pipeline.ps_estimation as ps_estimation
import ledgetter.utils.chuncks as chuncks
import jax
import itertools

def main():
    args = common.parse_args()
    is_ps = args.pattern == 'PS'
    chuncker, _ = chuncks.get_chuncker((args.step, args.step))
    sliced = list(chuncker)[args.slice_i]
    with jax.default_device(jax.devices(args.backend)[0]):
        values, images, mask, raycaster, shapes, full_shape, output, optimizer, scale, light_dict, light_names =\
            preprocessing.preprocess(
                args.ps_images_paths,
                sliced=sliced,
                meshroom_project=args.meshroom_project,
                aligned_image_path=args.aligned_image_path,
                geometry_path=args.geometry_path,
                pose_path=args.pose_path,
                black_image_path=args.black_image_path,
                loaded_light_folder=args.loaded_light_folder,
                load_light_function=is_ps,
                learning_rate=args.learning_rate,
                tqdm_refresh=args.tqdm_refresh,
                added_values={}
            )
        
        if is_ps:
            return_ps_only = all(s in args.skip_export for s in ('images', 'lightmaps', 'light', 'misc'))
            light_dict, validity_mask = ps_estimation.estimate_ps(args.iterations, values, images, mask, raycaster, shapes, output, optimizer, scale, light_dict, delta=args.delta, chunck_number = args.ps_chunck_number, return_ps_only=return_ps_only)
        else:
            light_dict, validity_mask = light_estimation.estimate_light(args.iterations, args.pattern, values, images, mask, raycaster, shapes, output, optimizer, scale, light_dict, delta=args.delta)
        outputs.export_results(args.out_path, validity_mask, light_dict, mask, images, light_names, skip = args.skip_export)

if __name__ == "__main__":
    main()