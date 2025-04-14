import glob
import pipeline.estimate_light.preprocessing as preprocessing
import pipeline.estimate_light.optim_steps as optim_steps
import pipeline.estimate_light.outputs as outputs
import optax
import numpy
import jax

mode = 'LED'


ps_images_paths = sorted(glob.glob(f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/PS_02/DSC_*.NEF'))
out_path = f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/lights/PS_02/LED_local'
step = 10
project_path = f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/meshroom/02'


sliced = (slice(None,None, step), slice(None,None, step))
points, normals, pixels, images, raycaster, mask, shapes, output, optimizer, scale = preprocessing.preprocess(ps_images_paths, sliced=sliced, meshroom_project=project_path)

if mode=='grid':
    valid_options = {'local_threshold':0.5, 'global_threshold':0.1, 'dilation':2, 'erosion':9, 'raycaster' : raycaster, 'radius' : 0.0005*scale}
    iterations = {'lambertian':1000}
    parameters, data, losses, steps = optim_steps.estimate_grid_light(points, normals, images, pixels, shapes, output, optimizer, mask, valid_options, iterations, 800)
    outputs.export_results(out_path, mask, parameters, data, losses, steps, images, ps_images_paths)

elif mode=='LED':
    valid_options={'local_threshold':0.5, 'global_threshold':0.1, 'dilation':2, 'erosion':9, 'raycaster' : raycaster, 'radius' : 0.0005*scale}
    iterations = {'directional' : 600, 'rail':300, 'punctual':4000, 'LED' : 5000, 'specular':3000}
    #del iterations['rail']
    parameters, data, losses, steps = optim_steps.estimate_physical_light(points, normals, images, pixels, shapes, output, optimizer, mask, valid_options, iterations)
    outputs.export_results(out_path, mask, parameters, data, losses, steps, images, ps_images_paths)

# del points, normals, pixels, images, validity_mask, mask, shapes, output, optimizer,  parameters, data, losses, steps

#TODO : tester image PNG, avec ou sans distorsion, mesh ou sphere

#TODO : tester avec une black image
#TODO Tester les harmoniques spheriques