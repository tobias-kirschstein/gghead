from pathlib import Path

import torch
import tyro
from eg3d.visualizer import Visualizer

from gghead.model_manager.gghead_model_manager import GGHeadModelFolder


def main():
    with torch.no_grad():
        viz = Visualizer(capture_dir=None)

        # List pickles.
        pretrained = [
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhq512-128.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/afhqcats512-128.pkl',
        ]

        gghead_models_path = GGHeadModelFolder().get_location()
        gghead_model_snapshot_paths = sorted([str(path) for path in Path(gghead_models_path).rglob("*.pkl")])
        pretrained.extend(gghead_model_snapshot_paths)

        # Populate recent pickles list with pretrained model URLs.
        for url in pretrained:
            viz.add_recent_pickle(url)

        # Run.
        while not viz.should_close():
            viz.draw_frame()
        viz.close()

if __name__ == '__main__':
    tyro.cli(main)
