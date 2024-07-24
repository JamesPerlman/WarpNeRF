from pathlib import Path
from typing import List
import numpy as np

class BundlerSFMCameraData:
    f: float
    k1: float
    k2: float
    R: np.ndarray
    t: np.ndarray

class BundlerSFMViewData:
    camera_index: int
    sift_index: int
    x: float
    y: float

class BundlerSFMPointData:
    position: np.ndarray
    color: np.ndarray
    view_list: List[BundlerSFMViewData]

class BundlerSFMData:
    cameras: List[BundlerSFMCameraData]
    points: List[BundlerSFMPointData]


def read_bundler_sfm_data(path: Path, read_points: bool = False) -> BundlerSFMData:
    with path.open("r") as file:
        data = BundlerSFMData()

        data.cameras = []
        data.points = []

        header = file.readline().strip()

        num_cameras, num_points = map(int, file.readline().split(" "))

        for _ in range(num_cameras):
            camera = BundlerSFMCameraData()
            camera.f, camera.k1, camera.k2 = map(float, file.readline().split(" "))
            camera.R = np.array([list(map(float, file.readline().split(" "))) for _ in range(3)])
            camera.t = np.array(list(map(float, file.readline().split(" "))))
        
            data.cameras.append(camera)
        
        if not read_points:
            return data
        
        for _ in range(num_points):
            point = BundlerSFMPointData()
            point.position = np.array(list(map(float, file.readline().split(" "))))
            point.color = np.array(list(map(int, file.readline().split(" "))))
            point.view_list = []

            view_lists = file.readline().split(" ")[1:]

            # split view_list into groups of 4
            view_lists = [view_lists[i:i+4] for i in range(0, len(view_lists), 4)]

            for view_list in view_lists:
                view = BundlerSFMViewData()
                view.camera_index, view.sift_index = map(int, view_list[:2])
                view.x, view.y = map(float, view_list[2:])
                point.view_list.append(view)
            
            data.points.append(point)

        return data
