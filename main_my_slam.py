import cv2
import time
import json
import numpy as np

from config import Config
from frame import are_map_points_visible, are_map_points_visible_in_frame

from slam import Slam
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset_factory import dataset_factory
from dataset_types import SensorType, DatasetEnvironmentType

from utils_sys import Printer, force_kill_all_and_exit
from utils_serialization import SerializableEnumEncoder

from feature_tracker_configs import FeatureTrackerConfigs
from loop_detector_configs import LoopDetectorConfigs
from slam_viewer import SlamViewer


def frame_generator(dataset):
    image_id = 0
    while dataset.isOk():
        image_timestamp = dataset.getTimestamp()          # get current timestamp
        next_image_timestamp = dataset.getNextTimestamp()
        image_duration = next_image_timestamp-image_timestamp if (image_timestamp is not None and next_image_timestamp is not None) else -1
        image = dataset.getImageColor(image_id)
        image_depth = dataset.getDepth(image_id)
        if image is None:
            break

        yield image_id, image_timestamp, image_duration, image, image_depth
        image_id+=1

class SlamStateData():
    def __init__(self):
        self.cur_frame_id = None
        self.cur_pose = None
        self.cur_pose_timestamp = None
        self.poses = []
        self.pose_timestamps = []
        self.points = []
        self.colors = []
        self.points_visible_on_frame = []
        self.points_can_be_visible_on_frame = []

def get_slam_state_data(slam):
    state = SlamStateData()

    # get frame id (not always available?)
    state.cur_frame_id = slam.tracking.f_cur.id if slam.tracking.f_cur is not None else -1

    # get estimated pose and timestamp (frame or frame+predicted motion?)
    if slam.map.num_frames() > 0:
        state.cur_pose = slam.map.get_frame(-1).Twc.copy()
        state.cur_pose_timestamp = slam.map.get_frame(-1).timestamp

    # get estimated history of poses and timestamps
    num_map_keyframes = slam.map.num_keyframes()
    keyframes = slam.map.get_keyframes()
    state.poses = np.empty((num_map_keyframes, 4, 4), dtype=float)
    state.pose_timestamps = np.empty((num_map_keyframes,), dtype=float)
    for i, kf in enumerate(keyframes):
        state.poses[i] = kf.Twc
        state.pose_timestamps[i] = kf.timestamp

    # get estimated map points
    num_map_points = slam.map.num_points()
    map_points = slam.map.get_points()
    state.points = np.empty((num_map_points, 3), dtype=float)
    state.colors = np.empty((num_map_points, 3), dtype=float)
    for i, p in enumerate(map_points):
        state.points[i] = p.pt
        state.colors[i] = np.flip(p.color) / 256.

    state.points_visible_on_frame = np.array([p.pt for p in slam.tracking.f_cur.get_matched_points()])

    possibly_visible_points_mask, _, _, _ = slam.tracking.f_cur.are_visible(map_points)
    state.points_can_be_visible_on_frame = state.points[possibly_visible_points_mask == 1]

    return state

if __name__ == "__main__":
    config = Config()

    # TODO: add interactive control:
    # - specify SLAM config
    # - load map

    dataset = dataset_factory(config)
    groundtruth = groundtruth_factory(config.dataset_settings)
    camera = PinholeCamera(config)

    # Select your tracker configuration (see the file feature_tracker_configs.py) 
    # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, KEYNET, SUPERPOINT, CONTEXTDESC, LIGHTGLUE, XFEAT, XFEAT_XFEAT
    # WARNING: At present, SLAM does not support LOFTR and other "pure" image matchers (further details in the commenting notes about LOFTR in feature_tracker_configs.py).
    feature_tracker_config = FeatureTrackerConfigs.ORB2
        
    # Select your loop closing configuration (see the file loop_detector_configs.py). Set it to None to disable loop closing. 
    # LoopDetectorConfigs: DBOW2, DBOW2_INDEPENDENT, DBOW3, DBOW3_INDEPENDENT, IBOW, OBINDEX2, VLAD, HDC_DELF, SAD, ALEXNET, NETVLAD, COSPLACE, EIGENPLACES  etc.
    # NOTE: under mac, the boost/text deserialization used by DBOW2 and DBOW3 may be very slow.
    loop_detection_config = LoopDetectorConfigs.DBOW3
        
    Printer.green('feature_tracker_config: ',json.dumps(feature_tracker_config, indent=4, cls=SerializableEnumEncoder))          
    Printer.green('loop_detection_config: ',json.dumps(loop_detection_config, indent=4, cls=SerializableEnumEncoder))

    # create SLAM object
    assert dataset.sensorType() == SensorType.MONOCULAR
    assert dataset.environmentType() == DatasetEnvironmentType.OUTDOOR
    slam = Slam(camera, feature_tracker_config, 
                loop_detection_config, dataset.sensorType(),
                environment_type=dataset.environmentType(),
                config=config,
                headless=True)
    slam.set_viewer_scale(dataset.scale_viewer_3d)

    # init viewer of traced data
    viewer = SlamViewer()
    viewer.set_camera_config(camera)
    viewer.set_world_scale(1)

    for img_id, timestamp, frame_duration, img, depth in frame_generator(dataset):
        print(f'image: {img_id}, timestamp: {timestamp}, duration: {frame_duration}')

        time_start = time.time()

        slam.track(img, None, None, img_id, timestamp)  # main SLAM function

        duration = time.time() - time_start
        if duration < frame_duration:
            # probably needed because internal algos rely on system time
            time.sleep(frame_duration - duration)

        ss = get_slam_state_data(slam)

        # logging
        # - camera frame with detected key points
        viewer.log_camera_frame(img_id, slam.map.draw_feature_trails(img))

        # - camera current pose
        if ss.cur_pose is not None:
            viewer.log_camera_pose(ss.cur_frame_id, ss.cur_pose)

        # - camera estimated trajectory:
        viewer.log_camera_pose_trajectory_estimated(ss.cur_frame_id, ss.poses)

        # - map of feature points
        viewer.log_map_points(ss.cur_frame_id, ss.points, ss.colors)
        viewer.log_map_points_currently_visible(ss.cur_frame_id, ss.points_visible_on_frame)
        viewer.log_map_points_can_be_currently_visible(ss.cur_frame_id, ss.points_can_be_visible_on_frame)
        if len(ss.points_can_be_visible_on_frame) > 0:
            viewer.log_stat_map_points_visible_to_observable_ratio(ss.cur_frame_id, len(ss.points_visible_on_frame) / len(ss.points_can_be_visible_on_frame))

        # TODO: add logging:
        # - image display with feature trails (img_draw = slam.map.draw_feature_trails(img))
        # - trajectory (online_trajectory_writer.write_trajectory(slam.tracking.cur_R, slam.tracking.cur_t, timestamp))
        # - a lot of things from plot_drawer.draw(img_id) TODO: specify details
        # - map (viewer3D.draw_slam_map(slam))
        # - dense map (viewer3D.draw_dense_map(slam))
        # - SLAM state? (how many times the state is SlamState.LOST)
        # - comparison with ground truth with align_with_scale (eval_ate)

        # TODO: add interactive control:
        # - step by step mode
        # - reset
        # - save map
        # - do bundle adjustment
        # - exit

    # TODO: add interactive control:
    # - save map
    # - save trajectory
    # - save stats

    # close stuff 
    slam.quit()
    cv2.destroyAllWindows()
    force_kill_all_and_exit(verbose=False) # just in case