from dataclasses import dataclass
from pathlib import Path
from threading import Thread
import subprocess as sp
import time
import tyro

def run_reality_capture(input: Path, output: Path):
    """ Perform Structure from Motion using RealityCapture. """
    
    rc_path = Path("C:/Program Files/Capturing Reality/RealityCapture/RealityCapture.exe")

    input_images_path = input
    output_project_path = output

    output_registration_path = output_project_path / "registration.out"
    output_undistorted_images_path = output_project_path / "undistorted_images"
    output_sparse_point_cloud_path = output_project_path / "sparse_point_cloud.ply"

    output_project_path.mkdir(parents=True, exist_ok=True)
    output_undistorted_images_path.mkdir(parents=True, exist_ok=True)

    rc_cmd = [
        str(rc_path),
        "-headless",
        "-newScene",
        "-addFolder", str(input_images_path),
        "-align",
        "-exportRegistration", str(output_registration_path),
        "-exportSparsePointCloud", str(output_sparse_point_cloud_path),
        "-quit"
    ]

    rc_proc = sp.Popen(rc_cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True)

    # Read and print each line of output
    while rc_proc.poll() is None:
        out = rc_proc.stdout.read(1)
        if out:
            print(out.decode("utf-8"), end="")

    # check for any remaining output
    out, err = rc_proc.communicate()

    if err:
        print(err.decode("utf-8"))


def main():
    tyro.extras.subcommand_cli_from_dict(
        {
            "sfm": run_reality_capture,
            "train": None,
        }
    )

# nk sfm --input /e/2022/nerf-library/testdata/lego/train/ --output /e/nerfs/test/dozer/