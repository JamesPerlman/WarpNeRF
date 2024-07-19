from dataclasses import dataclass
from pathlib import Path
import subprocess as sp
import tyro

def run_reality_capture(input: Path, output: Path):
    """ Perform Structure from Motion using RealityCapture. """
    
    rc_path = Path("C:/Program Files/Epic Games/RealityCapture/AppProxy.exe")

    input_images_path = input
    output_project_path = output

    output_registration_path = output_project_path / "registration.out"
    output_undistorted_images_path = output_project_path / "undistorted_images"
    output_sparse_point_cloud_path = output_project_path / "sparse_point_cloud.ply"

    output_project_path.mkdir(parents=True, exist_ok=True)
    output_undistorted_images_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(rc_path),
        "-headless",
        "-newScene",
        "-addFolder", str(input_images_path),
        "-align",
        "-exportRegistration", str(output_registration_path),
        "-exportSparsePointCloud", str(output_sparse_point_cloud_path),
        "-quit"
    ]

    sp.run(cmd, check=True)

def main():
    tyro.extras.subcommand_cli_from_dict(
        {
            "sfm": run_reality_capture,
            "train": None,
        }
    )
