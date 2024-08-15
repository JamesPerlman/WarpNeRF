from pathlib import Path
import tyro

from warpnerf.utils.bundler_sfm import read_bundler_sfm_data
from warpnerf.utils.reality_capture import run_reality_capture


def open_bundler_file(path: Path):
    data = read_bundler_sfm_data(path)

def main():
    tyro.extras.subcommand_cli_from_dict(
        {
            "sfm": run_reality_capture,
            "train": None,
            "open": open_bundler_file
        }
    )
