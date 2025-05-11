import rerun as rr
import rerun.blueprint as rrb
import cv2


class RerunLogger:
    def __init__(self):
        self._init_rerun()

    @staticmethod
    def _init_rerun():
        # predefine window layout
        blueprint = rrb.Horizontal(
            rrb.Spatial3DView(name="Scene", origin="/world")
        )

        # spawn window with predefined layout
        rr.init("Visual Odometry", spawn=True, default_blueprint=blueprint)

        # set axis directions
        rr.log("world", rr.ViewCoordinates.RDF, static=True)  # X=Right, Y=Down, Z=Forward

        RerunLogger._draw_3d_grid_plane()
        RerunLogger._draw_axes()

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

    def log_camera_pose(self, frame_id: int, img, camera, camera_pose) -> None:
        rr.set_time("frame_id", duration=frame_id)

        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rr.log("world/camera", rr.Transform3D(translation=t, mat3x3=R, from_parent=False))
        rr.log("world/camera",
               rr.Pinhole(
                   resolution=[camera.width, camera.height],
                   focal_length=[camera.fx, camera.fy],
                   principal_point=[camera.cx, camera.cy],
                   # image_plane_distance=20,
               ), )
        rr.log("world/camera", rr.Image(rgb))