import glob
import pipeline.estimate_light.preprocessing as preprocessing
import pipeline.estimate_light.optim_steps as optim_steps
import pipeline.estimate_light.outputs as outputs



mode = 'LED'

view_id, mesh_id =3, 3
ps_images_paths = sorted(glob.glob(f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/PS_{view_id:02d}/DSC_*.NEF'))
out_path = f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/PS_{view_id:02d}/light_v3_{mode}'
step = 10
project_path = f'/media/bcoupry/T7 Shield/Chauvet_1203_matin/meshroom/{mesh_id:02d}'
points, normals, pixels, images, validity_mask, mask, shapes, output, optimizer = preprocessing.preprocess(ps_images_paths, step, threshold=(0.3,0.2), meshroom_project=project_path)

# ps_images_paths = sorted(glob.glob(f'/media/bcoupry/T7 Shield/MSR_PROJECT/data/venus/multi_view_photo_stereo/VIEW_00000/raw/PS_*.NEF'))
# project_path = '/media/bcoupry/T7 Shield/MSR_side/venus'
# out_path = '/media/bcoupry/T7 Shield/MSR_side/venus/light_v3'
# aligned_image_path = '/media/bcoupry/T7 Shield1/MSR_PROJECT/data/venus/multi_view_photo_stereo/VIEW_00000/raw/PS_00000.NEF'
# mesh_path = '/media/bcoupry/T7 Shield/MSR_PROJECT/data/venus/multi_view/mesh/mesh.obj'
# step=8
#points, normals, pixels, images, validity_mask, mask, shapes, output, optimizer = preprocessing.preprocess(ps_images_paths, step, threshold=(0.3,0.2), meshroom_project=project_path, aligned_image_path=aligned_image_path, mesh_path=mesh_path, pose_path=None)


# ps_images_paths = sorted(glob.glob(f'/media/bcoupry/T7 Shield/Dome/RGB/DSC_*.png'))
# geometry_path = '/media/bcoupry/T7 Shield/Dome/pyllipse2/results/result.npz'
# out_path = '/media/bcoupry/T7 Shield/Dome/pyllipse2/results/LED_getter'
# step = 1
#points, normals, pixels, images, validity_mask, mask, shapes, output, optimizer = preprocessing.preprocess(ps_images_paths, step, threshold=(0.3,0.2), geometry_path=geometry_path)


if mode=='grid':
    iterations = {'lambertian':1000}
    parameters, losses = optim_steps.estimate_grid_light(points, normals, images, pixels, shapes, output, optimizer, mask, validity_mask, iterations, 800)
else:
    iterations = {'directional' : 400, 'rail':300, 'punctual':4000, 'LED' : 4000, 'specular':3000}
    parameters, losses = optim_steps.estimate_point_light(points, normals, images, shapes, output, optimizer, mask, validity_mask, iterations)

outputs.export_results(out_path, parameters, points, normals, pixels, images, validity_mask, mask, losses, ps_images_paths)



#TODO : tester image PNG, avec ou sans distorsion, mesh ou sphere
