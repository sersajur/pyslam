import numpy as np
import cv2
import os
import math
import rerun as rr
import rerun.blueprint as rrb

from config import Config

from visual_odometry import VisualOdometryEducational
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset_factory import dataset_factory
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_tracker_configs import FeatureTrackerConfigs

from rerun_interface import Rerun

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

    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5*traj_img_size)
    draw_scale = 1

    Rerun.init_vo()

    for img_id, timestamp, img, depth in frame_generator(config):

        # main VO function
        vo.track(img, depth, img_id, timestamp)

        # start drawing from the third image (when everything is initialized and flows in a normal way)
        if len(vo.traj3d_est)<=1:
            continue

        x, y, z = vo.traj3d_est[-1]
        gt_x, gt_y, gt_z = vo.traj3d_gt[-1]
        draw_x, draw_y = int(draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
        draw_gt_x, draw_gt_y = int(draw_scale*gt_x) + half_traj_img_size, half_traj_img_size - int(draw_scale*gt_z)

        cv2.circle(traj_img, (draw_x, draw_y), 1,(img_id*255/4540, 255-img_id*255/4540, 0), 1)   # estimated from green to blue
        cv2.circle(traj_img, (draw_gt_x, draw_gt_y), 1,(0, 0, 255), 1)  # ground_truth in red
        # write text on traj_img
        cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
        cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        # show
        Rerun.log_img_seq('trajectory_img/2d', img_id, traj_img)
        Rerun.log_2d_seq_scalar('trajectory_error/err_x', img_id, math.fabs(gt_x-x))
        Rerun.log_2d_seq_scalar('trajectory_error/err_y', img_id, math.fabs(gt_y-y))
        Rerun.log_2d_seq_scalar('trajectory_error/err_z', img_id, math.fabs(gt_z-z))

        Rerun.log_2d_seq_scalar('trajectory_stats/num_matches', img_id, vo.num_matched_kps)
        Rerun.log_2d_seq_scalar('trajectory_stats/num_inliers', img_id, vo.num_inliers)

        Rerun.log_3d_camera_img_seq(img_id, vo.draw_img, None, camera, vo.poses[-1])
        Rerun.log_3d_trajectory(img_id, vo.traj3d_est, 'estimated', color=[0,0,255])
        Rerun.log_3d_trajectory(img_id, vo.traj3d_gt, 'ground_truth', color=[255,0,0])

    if not os.path.exists(kResultsFolder):
        os.makedirs(kResultsFolder, exist_ok=True)
    print(f'saving {kResultsFolder}/map.png')
    cv2.imwrite(f'{kResultsFolder}/map.png', traj_img)
                
    cv2.destroyAllWindows()
