from dataclasses import dataclass
from typing import Optional, List

from elias.config import Config

from gghead.env import GGH_RENDERINGS_PATH
from gghead.model_manager.finder import find_model_folder
from gghead.util.name_builder import NameBuilder


@dataclass
class InterpolationRenderingConfig(Config):
    run_name: str
    checkpoint: int = -1
    n_persons: int = 5
    resolution: int = 512
    seeds: Optional[List[int]] = None
    n_frames: int = 1000
    n_circles: int = 5
    move_z: Optional[float] = None
    load_ema: bool = True
    truncation_psi: float = 0.7


class InterpolationRenderingManager:
    def __init__(self, config: InterpolationRenderingConfig):
        self._config = config

    def get_rendering_path(self):
        model_folder = find_model_folder(self._config.run_name)
        prefix = model_folder.get_prefix()
        c = self._config
        c_default = InterpolationRenderingConfig(None)
        output_folder = f"{GGH_RENDERINGS_PATH}/interpolations/{prefix}"
        prefix = f"{c.run_name}_ckpt-{c.checkpoint}"

        nb = NameBuilder(c, c_default, prefix=prefix, suffix='.mp4')
        nb.add('n_persons', 'pers', always=True)
        nb.add('resolution', 'res')
        nb.add('n_frames', 'frames')
        nb.add('n_circles', 'circ')
        nb.add('move_z')
        nb.add('load_ema', 'ema')
        nb.add('truncation_psi')
        nb.add('seeds')
        name = nb.get()

        output_path = f"{output_folder}/{name}"
        return output_path
