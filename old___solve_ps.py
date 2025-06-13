import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"


import glob
import pipeline.preprocessing as preprocessing
import pipeline.ps_pipeline as ps_pipeline
import pipeline._outputs_old as _outputs_old

import numpy
import jax
import imageio.v3 as iio
import os

view_id, mesh_id = 2, 2
ps_images_paths = sorted(glob.glob(f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/PS_{view_id:02d}/DSC_*.NEF'))
out_path = f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/photo_stereo/SH_PS_{view_id:02d}'
step = 4
project_path = f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/meshroom/{mesh_id:02d}'
light_path = f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/lights/PS_{view_id:02d}/LED_SH'





with jax.default_device(jax.devices("cpu")[0]):
    light_dict, shape = preprocessing.prepare_ps(light_path, ps_images_paths)
    rho_global, normals_global, validity_global, mask_global = jax.numpy.zeros(shape+(3,)), jax.numpy.zeros(shape+(3,)), jax.numpy.zeros(shape), jax.numpy.zeros(shape, dtype=bool)
    for i in range(step*step):
        u, v = jax.numpy.unravel_index(i, (step, step))
        sliced=(slice(u,None, step), slice(v,None, step))
        points, normals, pixels, images, raycaster, mask, shapes, output, optimizer, scale = preprocessing.preprocess(ps_images_paths, sliced=sliced, meshroom_project=project_path)
        valid_options={'local_threshold':0.5, 'global_threshold':0.1, 'dilation':0, 'erosion':30, 'raycaster' : raycaster, 'radius' : 0.0001*scale}
        iterations = {'PS':3000}
        parameters, data, losses, steps = ps_pipeline.solve_ps(light_dict, points, normals, images, pixels, shapes, output, optimizer, mask, valid_options, iterations, chunck_number = 100)
        ps_values = parameters | data
        x, y = ps_values['pixels'][:,0], ps_values['pixels'][:,1]
        rho_global, normals_global, validity_global, mask_global= rho_global.at[y, x].set(ps_values['rho']), normals_global.at[y, x].set(ps_values['normals']), validity_global.at[y, x].set(jax.numpy.mean(ps_values['validity_mask'], axis=(-1,-2))), mask_global.at[y, x].set(True)

    os.makedirs(out_path, exist_ok=True)
    numpy.savez(os.path.join(out_path, 'photo_stereo.npz'), normalmap=normals_global, albedomap=rho_global, mask=mask_global)
    iio.imwrite(os.path.join(out_path, 'normalmap_ps.png'),jax.numpy.uint8(0.5*(normals_global*jax.numpy.asarray([1,-1,-1])+1)*mask_global[:,:,None]*255))
    iio.imwrite(os.path.join(out_path, 'albedomap_ps.png'),jax.numpy.uint8(jax.numpy.clip(rho_global/jax.numpy.quantile(rho_global[mask_global], 0.99),0,1)*255))
    iio.imwrite(os.path.join(out_path, 'validity_ps.png'), jax.numpy.uint8(jax.numpy.clip(validity_global,0,1)*255))
    iio.imwrite(os.path.join(out_path, 'mask_ps.png'), mask_global)

    del ps_values, points, normals, pixels, images, mask, shapes, output, optimizer,  parameters, data, losses, steps, rho_global, normals_global, mask_global


