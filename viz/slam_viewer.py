import rerun as rr
import rerun.blueprint as rrb
import cv2
import numpy as np
from numpy import unsignedinteger


class SlamViewer:
    def __init__(self):
        self._init_rerun()

    @staticmethod
    def _init_rerun():
        # predefine window layout
        blueprint = rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Scene", origin="/world"),
                rrb.Spatial2DView(name="Camera", origin="/world/camera"),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="Trajectory Error", origin="/statistics/trajectory_error"),
                rrb.TimeSeriesView(name="Key Points", origin="/statistics/frame_key_points"),
            )
        )

        # spawn window with predefined layout
        rr.init("Slam Viewer", spawn=True, default_blueprint=blueprint)

        # set axis directions, draw axes and XZ grid
        rr.log("world", rr.ViewCoordinates.RDF, static=True)  # X=Right, Y=Down, Z=Forward
        SlamViewer._draw_3d_grid_plane()
        SlamViewer._draw_axes()

        # configure statistics plots
        # - Trajectory Error
        rr.log("statistics/trajectory_error/distance", rr.SeriesLines(colors=[255, 255, 255], widths=2), static=True)
        rr.log("statistics/trajectory_error/abs_x", rr.SeriesLines(colors=[255, 0, 0], widths=1), static=True)
        rr.log("statistics/trajectory_error/abs_y", rr.SeriesLines(colors=[0, 255, 0], widths=1), static=True)
        rr.log("statistics/trajectory_error/abs_z", rr.SeriesLines(colors=[0, 0, 255], widths=1), static=True)
        # - Key Points Number
        rr.log("statistics/frame_key_points/reference", rr.SeriesLines(colors=[255, 255, 255], widths=2), static=True)
        rr.log("statistics/frame_key_points/optically_matched", rr.SeriesLines(colors=[0, 255, 255], widths=1), static=True)
        rr.log("statistics/frame_key_points/geometrically_matched", rr.SeriesLines(colors=[0, 255, 0], widths=1), static=True)


    @staticmethod
    def _draw_3d_grid_plane(num_divs=30, div_size=10):
        # Plane parallel to x-z at origin with normal -y
        minx = -num_divs * div_size
        minz = -num_divs * div_size
        maxx = num_divs * div_size
        maxz = num_divs * div_size
        lines = []
        for n in range(2 * num_divs):
            lines.append([[minx + div_size * n, 0, minz], [minx + div_size * n, 0, maxz]])
            lines.append([[minx, 0, minz + div_size * n], [maxx, 0, minz + div_size * n]])
        rr.log(
            "world/grid",
            rr.LineStrips3D(lines, radii=0.01, colors=[0.7 * 255, 0.7 * 255, 0.7 * 255]),
            static=True,
        )

    @staticmethod
    def _draw_axes():
        rr.log(
            "world/axes",
            rr.Arrows3D(vectors=[[10, 0, 0], [0, 10, 0], [0, 0, 10]], labels=["X", "Y", "Z"],
                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]], radii=0.05),
            static=True,
        )

    def set_camera_config(self, camera):
        rr.log("world/camera", rr.Pinhole(
            resolution=[camera.width, camera.height],
            focal_length=[camera.fx, camera.fy],
            principal_point=[camera.cx, camera.cy],), static=True)

    def log_camera_frame(self, frame_id: int, frame) -> None:
        rr.set_time("frame_id", duration=frame_id)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log("world/camera", rr.Image(rgb))

    def log_camera_rotation_and_translation(self, frame_id: int, camera_R, camera_t) -> None:
        rr.set_time("frame_id", duration=frame_id)

        rr.log("world/camera", rr.Transform3D(translation=camera_t, mat3x3=camera_R, from_parent=False))

    def log_camera_pose(self, frame_id: int, camera_pose) -> None:
        rr.set_time("frame_id", duration=frame_id)

        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        self.log_camera_rotation_and_translation(frame_id, R, t)

    def log_camera_trajectory_estimated(self, frame_id: int, points) -> None:
        rr.set_time("frame_id", duration=frame_id)

        points = np.array(points).reshape(-1, 3)
        rr.log("world/camera_trajectory_estimated", rr.LineStrips3D([points], radii=0.1, colors=[200, 0, 0]))

    def log_camera_pose_trajectory_estimated(self, frame_id: int, poses) -> None:
        camera_positions = poses[:, :3, 3]
        self.log_camera_trajectory_estimated(frame_id, camera_positions)

    def log_camera_trajectory_reference(self, frame_id: int, points) -> None:
        rr.set_time("frame_id", duration=frame_id)

        points = np.array(points).reshape(-1, 3)
        rr.log("world/camera_trajectory_reference", rr.LineStrips3D([points], radii=0.1, colors=[0, 200, 0]))

    def log_map_points(self, frame_id: int, points, colors) -> None:
        rr.set_time("frame_id", duration=frame_id)
        points = np.array(points).reshape(-1, 3)
        colors = np.array(colors).reshape(-1, 3)
        rr.log("world/map_points", rr.Points3D(points, colors=colors))

    def log_camera_trajectory_error(self, frame_id: int, abs_xyz_error) -> None:
        rr.set_time("frame_id", duration=frame_id)

        rr.log("statistics/trajectory_error/abs_x", rr.Scalars(abs_xyz_error[0]))
        rr.log("statistics/trajectory_error/abs_y", rr.Scalars(abs_xyz_error[1]))
        rr.log("statistics/trajectory_error/abs_z", rr.Scalars(abs_xyz_error[2]))
        rr.log("statistics/trajectory_error/distance", rr.Scalars(np.linalg.norm(abs_xyz_error)))

    def log_camera_frame_key_points_reference(self, frame_id: int, key_points_count: unsignedinteger) -> None:
        rr.set_time("frame_id", duration=frame_id)

        rr.log("statistics/frame_key_points/reference", rr.Scalars(key_points_count))

    def log_camera_frame_key_points_optically_matched(self, frame_id: int, key_points_count: unsignedinteger) -> None:
        rr.set_time("frame_id", duration=frame_id)

        rr.log("statistics/frame_key_points/optically_matched", rr.Scalars(key_points_count))

    def log_camera_frame_key_points_geometrically_matched(self, frame_id: int, key_points_count: unsignedinteger) -> None:
        rr.set_time("frame_id", duration=frame_id)

        rr.log("statistics/frame_key_points/geometrically_matched", rr.Scalars(key_points_count))