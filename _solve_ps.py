import glob
import pipeline.estimate_light.preprocessing as preprocessing
import pipeline.estimate_light.optim_steps as optim_steps
import pipeline.estimate_light.outputs as outputs
import optax
import numpy
import jax
import imageio.v3 as iio
import os

view_id, mesh_id =3, 3
ps_images_paths = sorted(glob.glob(f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/PS_{view_id:02d}/DSC_*.NEF'))
out_path = f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/PS_{view_id:02d}/light_v3_PS'
step = 4
project_path = f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/meshroom/{mesh_id:02d}'
light_path = f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/PS_{view_id:02d}/light_v3_LED/light.npz'

shape = iio.improps(ps_images_paths[0]).shape

with jax.default_device(jax.devices("cpu")[0]):
    rho_global, normals_global, mask_global = jax.numpy.zeros(shape+(3,)), jax.numpy.zeros(shape+(3,)), jax.numpy.zeros(shape, dtype=bool)
    with numpy.load(light_path) as light:
        values = {k: v for k, v in light.items() if k not in {'losses'}}
    for i in range(step*step):
        u, v = jax.numpy.unravel_index(i, (step, step))
        points, normals, pixels, images, validity_mask, mask, shapes, output, optimizer = preprocessing.preprocess(ps_images_paths, sliced=(slice(u,None, step), slice(v,None, step)), threshold=(0.5,0.5), meshroom_project=project_path)

        iterations = {'PS':3000}
        parameters, data, losses, steps = optim_steps.solve_ps(values, points, normals, images, pixels, shapes, output, optimizer, mask, validity_mask, iterations, chunck_number=100)
        ps_values = parameters | data
        x, y = ps_values['pixels'][:,0], ps_values['pixels'][:,1]
        rho_global, normals_global, mask_global= rho_global.at[y, x].set(ps_values['rho']), normals_global.at[y, x].set(ps_values['normals']), mask_global.at[y, x].set(True)

    numpy.savez(os.path.join(out_path, 'photo_stereo.npz'), normalmap=normals_global, albedomap=rho_global, mask=mask_global)
    iio.imwrite(os.path.join(out_path, 'normalmap_ps.png'),jax.numpy.uint8(0.5*(normals_global*jax.numpy.asarray([1,-1,-1])+1)*mask_global[:,:,None]*255))
    iio.imwrite(os.path.join(out_path, 'albedomap_ps.png'),jax.numpy.uint8(jax.numpy.clip(rho_global/jax.numpy.quantile(rho_global[mask_global], 0.99),0,1)*255))
    iio.imwrite(os.path.join(out_path, 'mask_ps.png'), mask_global)

#TODO : tester image PNG, avec ou sans distorsion, mesh ou sphere
