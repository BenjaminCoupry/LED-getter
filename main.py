import pipeline.common as common
import pipeline.preprocessing as preprocessing
import pipeline.outputs as outputs
import pipeline.light_estimation as light_estimation
import pipeline.ps_estimation as ps_estimation
import ledgetter.utils.chuncks as chuncks
import jax
import os
import itertools

def main():
    args = common.parse_args()
    is_ps = args.pattern == 'PS'
    chuncker, _ = chuncks.get_chuncker((args.step, args.step))
    slicer = chuncker if is_ps else itertools.islice(chuncker, -1, None, None)
    for i, sliced in enumerate(slicer):
        print(f"SLICE {i:05d}")
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
                    load_light_function=False,
                    learning_rate=args.learning_rate,
                    tqdm_refresh=args.tqdm_refresh,
                    added_values={}
                )
            
            if is_ps:
                light_dict, validity_mask = ps_estimation.estimate_ps(args.iterations, values, images, mask, raycaster, shapes, output, optimizer, scale, light_dict, delta=args.delta, chunck_number = args.ps_chunck_number)
            else:
                light_dict, validity_mask = light_estimation.estimate_light(args.iterations, args.pattern, values, images, mask, raycaster, shapes, output, optimizer, scale, light_dict, delta=args.delta)

            real_out_path = os.path.join(args.out_path, f"{i:05d}") if is_ps else args.out_path
            outputs.export_results(real_out_path, validity_mask, light_dict, mask, images, light_names)

if __name__ == "__main__":
    main()