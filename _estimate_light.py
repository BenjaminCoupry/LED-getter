import glob
import pipeline.estimate_light.preprocessing as preprocessing
import pipeline.estimate_light.optim_steps as optim_steps
import pipeline.estimate_light.outputs as outputs
import optax
import numpy
import jax

mode = 'LED'

view_id, mesh_id =3, 3
ps_images_paths = sorted(glob.glob(f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/PS_{view_id:02d}/DSC_*.NEF'))
out_path = f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/PS_{view_id:02d}/light_v3_{mode}'
step = 10
project_path = f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/meshroom/{mesh_id:02d}'

points, normals, pixels, images, validity_mask, mask, shapes, output, optimizer = preprocessing.preprocess(ps_images_paths, sliced=(slice(None,None, step), slice(None,None, step)), threshold=(0.5,0.5), meshroom_project=project_path, storage_device = jax.devices("cpu")[0])

if mode=='grid':
    iterations = {'lambertian':1000}
    parameters, data, losses, steps = optim_steps.estimate_grid_light(points, normals, images, pixels, shapes, output, optimizer, mask, validity_mask, iterations, 800)
    outputs.export_results(out_path, mask, parameters, data, losses, steps, images, ps_images_paths)

elif mode=='LED':
    iterations = {'directional' : 400, 'rail':300, 'punctual':4000, 'LED' : 4000, 'specular':3000}
    parameters, data, losses, steps = optim_steps.estimate_physical_light(points, normals, images, pixels, shapes, output, optimizer, mask, validity_mask, iterations)
    outputs.export_results(out_path, mask, parameters, data, losses, steps, images, ps_images_paths)


#TODO : tester image PNG, avec ou sans distorsion, mesh ou sphere
