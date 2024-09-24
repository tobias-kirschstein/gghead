from dataclasses import dataclass
from typing import Union, Optional

from elias.config import Config
from elias.manager.model import _ModelType

from gghead.dataset.image_folder_dataset import GGHeadImageFolderDatasetConfig
from gghead.model_manager.base_model_manager import GGHeadTrainSetup, GGHeadGANOptimizerConfig, GGHeadGANConfig, \
    BaseModelManager, \
    BaseModelFolder


@dataclass
class GGHeadExperimentConfig(Config):
    model_config: GGHeadGANConfig
    dataset_config: GGHeadImageFolderDatasetConfig
    optimizer_config: GGHeadGANOptimizerConfig
    train_setup: GGHeadTrainSetup


class GGHeadModelManager(BaseModelManager):

    def __init__(self, run_name: str):
        super().__init__('gghead', run_name)

    def _load_checkpoint(self, checkpoint_file_name: Union[str, int], load_ema: bool = False, **kwargs) -> _ModelType:
        model = super()._load_checkpoint(checkpoint_file_name, load_ema, **kwargs)

        # Backward compatibility
        if not hasattr(model, '_n_uv_channels_background'):
            model._n_uv_channels_background = 0

        if model._config.use_initial_scales and not hasattr(model, '_initial_gaussian_scales_head'):
            model.register_buffer("_initial_gaussian_scales_head", model._initial_gaussian_scales, persistent=False)

        if not hasattr(model, '_n_uv_channels_per_shell'):
            # n_uv_channels was renamed into n_uv_channels_per_shell
            model._n_uv_channels_per_shell = model._n_uv_channels

        if not hasattr(model, '_n_uv_channels_decoded'):
            model._n_uv_channels_decoded = model._n_uv_channels

        if not hasattr(model, '_n_uv_channels_per_shell_decoded'):
            model._n_uv_channels_per_shell_decoded = model._n_uv_channels_per_shell

        if not hasattr(model._config, 'template_update_attributes'):
            # template_update_attributes was added to config and used in forward pass
            model._config.template_update_attributes = []

        if (model._config.super_resolution_config.use_superresolution
                and model._config.super_resolution_config.superresolution_version == 2
                and not hasattr(model.super_resolution, 'n_downsampling_layers')):
            # Number of downsampling layers was made variable
            model.super_resolution.n_downsampling_layers = 1

        return model


class GGHeadModelFolder(BaseModelFolder[GGHeadModelManager]):
    def __init__(self):
        super().__init__('gghead', 'GGHEAD')

    def new_run(self, name: Optional[str] = None) -> GGHeadModelManager:
        return super().new_run(name)

    def open_run(self, run_name_or_id: Union[str, int]) -> GGHeadModelManager:
        return super().open_run(run_name_or_id)
