from dataclasses import dataclass
from enum import auto

from elias.config import StringEnum, Config, implicit


@dataclass
class GaussianAttributeConfig(Config):
    use_rodriguez_rotation: bool = False  # If set, predict rodriguez rotation in [0, 2pi] instead of quaternions
    sh_degree: int = 0
    n_color_channels: int = implicit(3)

class GaussianAttribute(StringEnum):
    POSITION = auto()
    SCALE = auto()
    ROTATION = auto()
    OPACITY = auto()
    COLOR = auto()

    def get_n_channels(self, config: GaussianAttributeConfig) -> int:
        if self == self.POSITION:
            return 3
        elif self == self.SCALE:
            return 3
        elif self == self.ROTATION:
            if config.use_rodriguez_rotation:
                return 3
            else:
                return 4
        elif self == self.OPACITY:
            return 1
        elif self == self.COLOR:
            n_channels_per_color = (config.sh_degree + 1) ** 2
            return n_channels_per_color * config.n_color_channels
        else:
            raise ValueError(f"Cannot get n_channels of {self}")
