import numpy
import glob
import os
import ledgetter.utils.files as files


def format_meshroom_intrinsic(meshroom_intrinsic):
    """Formats camera intrinsic parameters into a matrix.

    Args:
        intrinsic (dict): Dictionary containing camera intrinsics.

    Returns:
        Tuple:
            - Array 3, 3: Camera intrinsic matrix.
            - int: Image width.
            - int: Image height.
            - Array N,: Distortion parameters.
    """
    width = float(meshroom_intrinsic['width'])
    height = float(meshroom_intrinsic['height'])

    # Get focal length
    sensor_width = float(meshroom_intrinsic['sensorWidth'])
    sensor_height = float(meshroom_intrinsic['sensorHeight'])
    focal_length = float(meshroom_intrinsic['focalLength'])
    fx = focal_length * width / sensor_width
    fy = focal_length * height / sensor_height

    # Get principal point
    cx = width / 2 + float(meshroom_intrinsic['principalPoint'][0])
    cy = height / 2 + float(meshroom_intrinsic['principalPoint'][1])

    # Get intrinsics matrix
    K = numpy.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=numpy.float32)

    distorsion = numpy.array(meshroom_intrinsic['distortionParams'], dtype=numpy.float32)

    return K, int(width), int(height), distorsion

def format_meshroom_extrinsic(meshroom_extrinsic):
    """Extracts the rotation matrix and translation vector from a pose dictionary.

    Args:
        pose (dict): Dictionary containing camera pose information.

    Returns:
        Tuple:
            - Array 3, 3: Rotation matrix.
            - Array 3,: Translation vector.
    """
    # Get rotation matrix and center in OpenGL convention
    R = numpy.array(meshroom_extrinsic['pose']['transform']['rotation'], dtype=numpy.float32).reshape([3,3])
    t = numpy.array(meshroom_extrinsic['pose']['transform']['center'], dtype=numpy.float32)
    return R, t

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

def get_pose(sfm, view_id):
    """Retrieves the camera pose and intrinsic parameters for a given view.

    Args:
        sfm (dict): Structure-from-motion data.
        view_id (int): ID of the view.

    Returns:
        dict: Dictionary containing:
            - 'K' (Array 3, 3): Camera intrinsic matrix.
            - 'R' (Array 3, 3): Rotation matrix.
            - 't' (Array 3,): Translation vector.
            - 'width' (int): Image width.
            - 'height' (int): Image height.
            - 'distorsion' (Array N,): Distortion parameters.
    """
    extrinsics, views, intrinsics, _ = unpack_sfm(sfm)
    view = views[view_id]
    extrinsic_id, instrinsic_id = view['poseId'], view['intrinsicId']
    extrinsic, intrinsic = extrinsics[extrinsic_id], intrinsics[instrinsic_id]
    K, width, height, distorsion = format_meshroom_intrinsic(intrinsic)
    R, t = format_meshroom_extrinsic(extrinsic)
    pose_dict = {'K':K, 'R':R, 't':t, 'width':width, 'height':height, 'distorsion' : distorsion}
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
