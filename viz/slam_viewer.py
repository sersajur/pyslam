import rerun as rr
import rerun.blueprint as rrb
import cv2
import numpy as np
from numpy import unsignedinteger


class SlamViewer:
    _PTH_WORLD = "/world"
    _PTH_WORLD_ESTIMATED = "/world/estimated"
    _PTH_WORLD_REFERENCE = "/world/reference"
    _PTH_TRAJECTORY_ESTIMATED = "/world/estimated/trajectory"
    _PTH_TRAJECTORY_REFERENCE = "/world/reference/trajectory"
    _PTH_MAP_ESTIMATED = "/world/estimated/map"
    _PTH_MAP_ESTIMATED_VISIBLE = "/world/estimated/map/visible"
    _PTH_MAP_ESTIMATED_POTENTIALLY_VISIBLE = "/world/estimated/map/potentially_visible"
    _PTH_CAMERA = "/world/estimated/camera"
    _PTH_CAMERA_IMAGE = "/world/estimated/camera/image"
    _PTH_STAT_TRAJECTORY_ERROR = "/statistics/trajectory_error"
    _PTH_STAT_TRAJECTORY_ERROR_DISTANCE = "/statistics/trajectory_error/distance"
    _PTH_STAT_TRAJECTORY_ERROR_ABS_X = "/statistics/trajectory_error/abs_x"
    _PTH_STAT_TRAJECTORY_ERROR_ABS_Y = "/statistics/trajectory_error/abs_y"
    _PTH_STAT_TRAJECTORY_ERROR_ABS_Z = "/statistics/trajectory_error/abs_z"
    _PTH_STAT_FRAME_KEY_POINTS = "/statistics/frame_key_points"
    _PTH_STAT_FRAME_KEY_POINTS_REFERENCE = "/statistics/frame_key_points/reference"
    _PTH_STAT_FRAME_KEY_POINTS_OPTICALLY_MATCHED = "/statistics/frame_key_points/optically_matched"
    _PTH_STAT_FRAME_KEY_POINTS_GEOMETRICALLY_MATCHED = "/statistics/frame_key_points/geometrically_matched"
    _PTH_STAT_MAP_POINTS_FRAME_VISIBLE_TO_OBSERVABLE_RATIO = "/statistics/map/frame_visible_to_observable_ratio"
    def __init__(self):
        self._init_rerun()

    def _init_rerun(self):
        # predefine window layout
        blueprint = rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Scene", origin=self._PTH_WORLD),
                rrb.Spatial2DView(name="Camera", origin=self._PTH_CAMERA),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="Trajectory Error", origin=self._PTH_STAT_TRAJECTORY_ERROR),
                rrb.TimeSeriesView(name="Key Points", origin=self._PTH_STAT_FRAME_KEY_POINTS),
            )
        )

        # spawn window with predefined layout
        rr.init("Slam Viewer", spawn=True, default_blueprint=blueprint)

        # set axis directions, draw axis
        rr.log(self._PTH_WORLD, rr.ViewCoordinates.RDF, static=True)  # X=Right, Y=Down, Z=Forward
        self.set_world_scale(1)

        # configure statistics plots
        # - Trajectory Error
        rr.log(self._PTH_STAT_TRAJECTORY_ERROR_DISTANCE, rr.SeriesLines(colors=[255, 255, 255], widths=2), static=True)
        rr.log(self._PTH_STAT_TRAJECTORY_ERROR_ABS_X, rr.SeriesLines(colors=[255, 0, 0], widths=1), static=True)
        rr.log(self._PTH_STAT_TRAJECTORY_ERROR_ABS_Y, rr.SeriesLines(colors=[0, 255, 0], widths=1), static=True)
        rr.log(self._PTH_STAT_TRAJECTORY_ERROR_ABS_Z, rr.SeriesLines(colors=[0, 0, 255], widths=1), static=True)
        # - Key Points Number
        rr.log(self._PTH_STAT_FRAME_KEY_POINTS_REFERENCE, rr.SeriesLines(colors=[255, 255, 255], widths=2), static=True)
        rr.log(self._PTH_STAT_FRAME_KEY_POINTS_OPTICALLY_MATCHED, rr.SeriesLines(colors=[0, 255, 255], widths=1), static=True)
        rr.log(self._PTH_STAT_FRAME_KEY_POINTS_GEOMETRICALLY_MATCHED, rr.SeriesLines(colors=[0, 255, 0], widths=1), static=True)
        # - Map Points statistics
        rr.log(self._PTH_STAT_MAP_POINTS_FRAME_VISIBLE_TO_OBSERVABLE_RATIO, rr.SeriesLines(colors=[255, 255, 255], widths=2), static=True)

    def set_camera_config(self, camera):
        rr.log(self._PTH_CAMERA, rr.Pinhole(
            resolution=[camera.width, camera.height],
            focal_length=[camera.fx, camera.fy],
            principal_point=[camera.cx, camera.cy],
            image_plane_distance=1), static=True)

    def set_world_scale(self, scale):
        rr.log(self._PTH_WORLD_ESTIMATED, rr.Transform3D(scale=[scale, scale, scale], from_parent=False, axis_length=1), static=True)

    def log_camera_frame(self, frame_id: int, frame) -> None:
        rr.set_time("frame_id", duration=frame_id)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log(self._PTH_CAMERA_IMAGE, rr.Image(rgb))

    def log_camera_rotation_and_translation(self, frame_id: int, camera_R, camera_t) -> None:
        rr.set_time("frame_id", duration=frame_id)

        rr.log(self._PTH_CAMERA, rr.Transform3D(translation=camera_t, mat3x3=camera_R, from_parent=False))

    def log_camera_pose(self, frame_id: int, camera_pose) -> None:
        rr.set_time("frame_id", duration=frame_id)

        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        self.log_camera_rotation_and_translation(frame_id, R, t)

    def log_camera_trajectory_estimated(self, frame_id: int, points) -> None:
        rr.set_time("frame_id", duration=frame_id)

        points = np.array(points).reshape(-1, 3)
        rr.log(self._PTH_TRAJECTORY_ESTIMATED, rr.LineStrips3D([points], colors=[200, 0, 0]))

    def log_camera_pose_trajectory_estimated(self, frame_id: int, poses) -> None:
        camera_positions = poses[:, :3, 3]
        self.log_camera_trajectory_estimated(frame_id, camera_positions)

    def log_camera_trajectory_reference(self, frame_id: int, points) -> None:
        rr.set_time("frame_id", duration=frame_id)

        points = np.array(points).reshape(-1, 3)
        rr.log(self._PTH_TRAJECTORY_REFERENCE, rr.LineStrips3D([points], colors=[0, 200, 0]))

    def log_map_points(self, frame_id: int, points, colors) -> None:
        rr.set_time("frame_id", duration=frame_id)
        points = np.array(points).reshape(-1, 3)
        colors = np.array(colors).reshape(-1, 3)
        rr.log(self._PTH_MAP_ESTIMATED, rr.Points3D(points, colors=colors, radii=0.05))

    def log_map_points_currently_visible(self, frame_id: int, points) -> None:
        rr.set_time("frame_id", duration=frame_id)
        points = np.array(points).reshape(-1, 3)

        rr.log(self._PTH_MAP_ESTIMATED_VISIBLE, rr.Points3D(points, colors=[0, 255, 0], radii=0.05))

    def log_map_points_can_be_currently_visible(self, frame_id: int, points) -> None:
        rr.set_time("frame_id", duration=frame_id)
        points = np.array(points).reshape(-1, 3)

        rr.log(self._PTH_MAP_ESTIMATED_POTENTIALLY_VISIBLE, rr.Points3D(points, colors=[0, 0, 255], radii=0.05))

    def log_stat_map_points_visible_to_observable_ratio(self, frame_id: int, ratio) -> None:
        rr.set_time("frame_id", duration=frame_id)

        rr.log(self._PTH_STAT_MAP_POINTS_FRAME_VISIBLE_TO_OBSERVABLE_RATIO, rr.Scalars(ratio))

    def log_camera_trajectory_error(self, frame_id: int, abs_xyz_error) -> None:
        rr.set_time("frame_id", duration=frame_id)

        rr.log(self._PTH_STAT_TRAJECTORY_ERROR_ABS_X, rr.Scalars(abs_xyz_error[0]))
        rr.log(self._PTH_STAT_TRAJECTORY_ERROR_ABS_Y, rr.Scalars(abs_xyz_error[1]))
        rr.log(self._PTH_STAT_TRAJECTORY_ERROR_ABS_Z, rr.Scalars(abs_xyz_error[2]))
        rr.log(self._PTH_STAT_TRAJECTORY_ERROR_DISTANCE, rr.Scalars(np.linalg.norm(abs_xyz_error)))

    def log_camera_frame_key_points_reference(self, frame_id: int, key_points_count: unsignedinteger) -> None:
        rr.set_time("frame_id", duration=frame_id)

        rr.log(self._PTH_STAT_FRAME_KEY_POINTS_REFERENCE, rr.Scalars(key_points_count))

    def log_camera_frame_key_points_optically_matched(self, frame_id: int, key_points_count: unsignedinteger) -> None:
        rr.set_time("frame_id", duration=frame_id)

        rr.log(self._PTH_STAT_FRAME_KEY_POINTS_OPTICALLY_MATCHED, rr.Scalars(key_points_count))

    def log_camera_frame_key_points_geometrically_matched(self, frame_id: int, key_points_count: unsignedinteger) -> None:
        rr.set_time("frame_id", duration=frame_id)

        rr.log(self._PTH_STAT_FRAME_KEY_POINTS_GEOMETRICALLY_MATCHED, rr.Scalars(key_points_count))