import numpy
import jax
import glob
import os
import ledgetter.utils.files as files
import ledgetter.image.camera as camera

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
    distorsion = jax.numpy.asarray(intrinsic['distortionParams'], dtype=numpy.float32)
    K = camera.build_K_matrix(fx, cx, cy, fy)
    intrinsic_dict = {'K':K, 'distorsion':distorsion, 'width':int(width), 'height':int(height)}
    return intrinsic_dict

def format_extrinsic(extrinsic):
    R = jax.numpy.reshape(jax.numpy.array(extrinsic['pose']['transform']['rotation'], dtype=numpy.float32), (3,3))
    t = jax.numpy.array(extrinsic['pose']['transform']['center'], dtype=numpy.float32)
    extrinsic_dict = {'R':R, 't':t}
    return extrinsic_dict

def unpack_sfm(sfm):
    """Unpacks the structure-from-motion (SfM) data.

    Args:
        sfm (dict): Dictionary containing the SfM data.

    Returns:
        Tuple:
            - dict: Mapping of pose IDs to pose data.
            - dict: Mapping of view IDs to view data.
            - dict: Mapping of intrinsic IDs to intrinsic data.
            - dict: Mapping of image file paths to view IDs.
    """
    extrinsics = {pose['poseId'] : pose for pose in sfm['poses']}
    views = {view['viewId'] : view for view in sfm['views']}
    intrinsics = {intrinsic['intrinsicId'] : intrinsic for intrinsic in sfm['intrinsics']}
    paths_ids = {view['path'] : view['viewId'] for view in sfm['views']}
    return extrinsics, views, intrinsics, paths_ids

def get_sfm_path(project_path):
    """Finds the path to the cameras.sfm file in a Meshroom project.

    Args:
        project_path (str): Path to the Meshroom project directory.

    Returns:
        str: Path to the cameras.sfm file.
    """
    sfm_path = max(glob.glob(os.path.join(project_path,'MeshroomCache','StructureFromMotion','*','cameras.sfm')), key=os.path.getmtime)
    return sfm_path

def get_mesh_path(project_path):
    """Finds the path to the mesh.obj file in a Meshroom project.

    Args:
        project_path (str): Path to the Meshroom project directory.

    Returns:
        str: Path to the mesh.obj file.
    """
    mesh_path = max(glob.glob(os.path.join(project_path,'MeshroomCache','MeshFiltering','*','mesh.obj')), key=os.path.getmtime)
    return mesh_path

def get_pose_dict(sfm, view_id):
    extrinsics, views, intrinsics, _ = unpack_sfm(sfm)
    view = views[view_id]
    extrinsic_id, instrinsic_id = view['poseId'], view['intrinsicId']
    extrinsic, intrinsic = extrinsics[extrinsic_id], intrinsics[instrinsic_id]
    pose_dict = format_extrinsic(extrinsic) | format_intrinsic(intrinsic)
    return pose_dict

def get_view_id(sfm, image_path):
    """Finds the view ID corresponding to a given image path.

    Args:
        sfm (dict): Structure-from-motion data.
        image_path (str or list): Path to the image file or a list of paths.

    Returns:
        int: View ID corresponding to one of the image paths.
    """
    _, _, _, paths_ids = unpack_sfm(sfm)
    file_matches = files.find_similar_path(list(paths_ids.keys()), image_path)[0]
    valid_path = file_matches[0][0] if len(file_matches)==1 else None
    view_id = paths_ids[valid_path]
    return view_id
