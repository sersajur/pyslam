import numpy as np
import os

from config import Config

from visual_odometry import VisualOdometryEducational
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset_factory import dataset_factory
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_tracker_configs import FeatureTrackerConfigs

from slam_viewer import SlamViewer

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kResultsFolder = kRootFolder + '/results'

def frame_generator(cfg):
    dataset = dataset_factory(cfg)
    image_id = 0
    while dataset.isOk():
        image_timestamp = dataset.getTimestamp()          # get current timestamp
        image = dataset.getImageColor(image_id)
        image_depth = dataset.getDepth(image_id)
        if image is None:
            break

        yield image_id, image_timestamp, image, image_depth
        image_id+=1


if __name__ == "__main__":

    config = Config()

    ground_truth = groundtruth_factory(config.dataset_settings)

    camera = PinholeCamera(config)
        
    # select your tracker configuration (see the file feature_tracker_configs.py)
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, LIGHTGLUE, XFEAT, XFEAT_XFEAT, LOFTR
    tracker_config = FeatureTrackerConfigs.LK_SHI_TOMASI
    tracker_config['num_features'] = 2000
    feature_tracker = feature_tracker_factory(**tracker_config)

    # create visual odometry object
    vo = VisualOdometryEducational(camera, ground_truth, feature_tracker)

    # init viewer of traced data
    viewer = SlamViewer()
    viewer.set_camera_config(camera)

    for img_id, timestamp, img, depth in frame_generator(config):
        # main VO function
        vo.track(img, depth, img_id, timestamp)

        # start drawing from the third image (when everything is initialized and flows in a normal way)
        if len(vo.traj3d_est)<=1:
            continue

        # log camera frame and estimated camera pose
        viewer.log_camera_frame(img_id, vo.draw_img)
        viewer.log_camera_pose(img_id, vo.poses[-1])

        # log camera trajectory: estimated, ground truth and their diff
        viewer.log_camera_trajectory_estimated(img_id, vo.traj3d_est)
        viewer.log_camera_trajectory_reference(img_id, vo.traj3d_gt)
        viewer.log_camera_trajectory_error(img_id, np.abs(vo.traj3d_est[-1] - vo.traj3d_gt[-1]))

        # log key points statistics
        viewer.log_camera_frame_key_points_reference(img_id, vo.kps_ref.shape[0])
        viewer.log_camera_frame_key_points_optically_matched(img_id, vo.num_matched_kps)
        viewer.log_camera_frame_key_points_geometrically_matched(img_id, vo.num_inliers)
