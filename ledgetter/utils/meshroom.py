import numpy
import glob
import os
import json
import open3d

def format_intrinsic(intrinsic):
    width = float(intrinsic['width'])
    height = float(intrinsic['height'])

    # Get focal length
    sensor_width = float(intrinsic['sensorWidth'])
    sensor_height = float(intrinsic['sensorHeight'])
    focal_length = float(intrinsic['focalLength'])
    fx = focal_length * width / sensor_width
    fy = focal_length * height / sensor_height

    # Get principal point
    cx = width / 2 + float(intrinsic['principalPoint'][0])
    cy = height / 2 + float(intrinsic['principalPoint'][1])

    # Get intrinsics matrix
    K = numpy.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=numpy.float32)

    distorsion = numpy.array(intrinsic['distortionParams'], dtype=numpy.float32)

    return K, int(width), int(height), distorsion

def format_pose(pose):
    # Get rotation matrix and center in OpenGL convention
    R = numpy.array(pose['pose']['transform']['rotation'], dtype=numpy.float32).reshape([3,3])
    t = numpy.array(pose['pose']['transform']['center'], dtype=numpy.float32)
    return R, t

def unpack_sfm(sfm):
    poses = {pose['poseId'] : pose for pose in sfm['poses']}
    views = {view['viewId'] : view for view in sfm['views']}
    intrinsics = {intrinsic['intrinsicId'] : intrinsic for intrinsic in sfm['intrinsics']}
    names_ids = {os.path.realpath(view['path']) : view['viewId'] for view in sfm['views']}
    return poses, views, intrinsics, names_ids

def get_sfm_path(project_path):
    sfm_path = glob.glob(os.path.join(project_path,'MeshroomCache','StructureFromMotion','*','cameras.sfm'))[0]
    return sfm_path

def get_mesh_path(project_path):
    mesh_path = glob.glob(os.path.join(project_path,'MeshroomCache','MeshFiltering','*','mesh.obj'))[0]
    return mesh_path

def get_pose(project_path, aligned_image_path):
    sfm_path = get_sfm_path(project_path)
    with open(sfm_path, 'r') as f:
        sfm = json.load(f)
    poses, views, intrinsics, names_ids = unpack_sfm(sfm)
    if type(aligned_image_path) is list:
        valid_path = next(filter(lambda name : os.path.realpath(name) in aligned_image_path, names_ids.keys()))
        view_id = names_ids[os.path.realpath(valid_path)]
    else:
        view_id = names_ids[os.path.realpath(aligned_image_path)]
    view = views[view_id]
    pose_id, instrinsic_id = view['poseId'], view['intrinsicId']
    pose, intrinsic = poses[pose_id], intrinsics[instrinsic_id]
    K, width, height, distorsion = format_intrinsic(intrinsic)
    R, t = format_pose(pose)
    pose_dict = {'K':K, 'R':R, 't':t, 'width':width, 'height':height, 'distorsion' : distorsion}
    return pose_dict

def get_mesh(path):
    if os.path.isdir(path):
        mesh_path = get_mesh_path(path)
    else:
        mesh_path = path
    mesh = open3d.t.io.read_triangle_mesh(mesh_path).transform(numpy.diag([1,-1,-1,1]))
    return mesh