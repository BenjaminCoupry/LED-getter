import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

import glob
import pipeline.preprocessing as preprocessing
import pipeline.outputs as outputs
import pipeline.light_estimation as light_estimation
import pipeline.ps_estimation as ps_estimation
import optax
import numpy
import jax

import ledgetter.models.defaults as defaults
import ledgetter.models.values_generator as values_generator
import ledgetter.optim.gradient_descent as gradient_descent
import ledgetter.models.models as models
import functools
import ledgetter.utils.chuncks as chuncks


import ledgetter.utils.plots as plots

#TODO : faire les decorateurs d'inputs dans la resolution de la PS


step = 21
ps_step = 4
it = 1000
pattern = 'directional'

loaded_light_folder = f'/media/bcoupry/T7 Shield/ChauvetJCHMVPS/light/{pattern}'
ps_images_paths = sorted(glob.glob(f'/media/bcoupry/T7 Shield/ChauvetJCHMVPS/rti/DSC_*.NEF'))[80:]
out_path = f'/media/bcoupry/T7 Shield/ChauvetJCHMVPS/light/{pattern}_1'
project_path = f'/media/bcoupry/T7 Shield/ChauvetJCHMVPS/meshroom'


sliced = next(chuncks.get_chuncker((step, step))[0])
values, images, mask, raycaster, shapes, full_shape, output, optimizer, scale, light_dict, light_names =\
      preprocessing.preprocess(ps_images_paths, sliced=sliced, meshroom_project=project_path, loaded_light_folder=loaded_light_folder, load_light_function=True)
light_dict, validity_mask = light_estimation.estimate_light(it, pattern, values, images, mask, raycaster, shapes, output, optimizer, scale, light_dict)

outputs.export_results(out_path, validity_mask,light_dict, mask, images, light_names)

light_dict, validity_mask = ps_estimation.estimate_ps(it, values, images, mask, raycaster, shapes, output, optimizer, scale, light_dict)

outputs.export_results(out_path+'_PS', validity_mask, light_dict, mask, images, light_names)

print()