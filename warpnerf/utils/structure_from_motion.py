from pathlib import Path
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image
import pycolmap

import torch

def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix in ['.jpg', '.png', '.jpeg']

class SfMPipeline:
    def __init__(self):
        self.extractor = ALIKED()
        self.matcher = LightGlue(features='aliked')

    def features_for_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.extractor.extract(image)

    def match_features(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        return self.matcher({ 'image0': features1, 'image1': features2 })

    def run(self, project_path: Path):

        features = []
        images = []
       
        for i, path in enumerate(project_path.iterdir()):
            if not is_image(path):
                continue

            image = load_image(path)

            images[i] = image
            features[i] = self.features_for_image(image)

        matches = []

        # windowed matching
        def arr_key(i, j):
            return (i, j) if i < j else (j, i)
        
        window_size = 15
        for i in range(len(images)):
            for j in range(i + 1, i + window_size):
                j = j % len(images)
                key_ij = arr_key(i, j)
                if not key_ij in matches:
                    matches[key_ij] = self.match_features(features[i], features[j])
        
        db = pycolmap.Database(project_path / "colmap.db")

def run_sfm(path: Path):
    pipeline = SfMPipeline()
    pipeline.run(path)
