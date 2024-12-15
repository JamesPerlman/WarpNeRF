from pathlib import Path
import tyro

from warpnerf.utils.bundler_sfm import create_bundler_sfm_data_from_path
from warpnerf.utils.reality_capture import run_reality_capture
from warpnerf.utils.structure_from_motion import run_sfm
from warpnerf.demos.hello_fvdb import hello_fvdb

def open_bundler_file(path: Path):
    data = create_bundler_sfm_data_from_path(path)

def main():
    tyro.extras.subcommand_cli_from_dict(
        {
            "rc": run_reality_capture,
            "sfm": run_sfm,
            "train": None,
            "open": open_bundler_file,
            "hellofvdb": hello_fvdb,
        }
    )
