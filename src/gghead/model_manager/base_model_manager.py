import pickle
from dataclasses import fields, dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Literal

from elias.config import Config, implicit
from elias.folder import ModelFolder, Folder
from elias.folder.run import _RunManagerType
from elias.manager import ModelManager
from elias.manager.model import _ModelConfigType, _OptimizationConfigType, _ModelType
from silberstral.silberstral import create_linked_type_var

from gghead.dataset.image_folder_dataset import GGHeadImageFolderDatasetConfig
from gghead.eg3d.loss import GGHStyleGAN2LossConfig
from gghead.env import GGH_MODELS_PATH
from gghead.models.gaussian_discriminator import GaussianDiscriminator, GaussianDiscriminatorConfig
from gghead.models.gghead_model import GGHeadConfig
from gghead.util.name_builder import NameBuilder


@dataclass
class OptimizerConfig(Config):
    lr: float = 0.002
    beta1: float = 0
    beta2: float = 0.99
    eps: float = 1e-8

    n_phases: int = 1


@dataclass
class GGHeadTrainSetup(Config):
    project_name: str = 'generative-gaussian-heads'
    group_name: str = 'generative-gaussian-heads'

    gpus: int = 1

    slurm_jobid: Optional[int] = None
    slurm_nodelist: Optional[str] = None

    accumulate_metrics: Optional[int] = None  # Average metrics over that many steps
    metrics: List[str] = field(default_factory=lambda: ['fid100'])
    resume_run: Optional[str] = None
    reset_cur_nimg: bool = False
    resume_checkpoint: int = -1
    total_kimg: Optional[int] = 25000
    wandb_run_id: str = implicit()


@dataclass
class GGHeadGANOptimizerConfig(Config):
    generator_optimizer_config: OptimizerConfig
    discriminator_optimizer_config: OptimizerConfig

    loss_config: GGHStyleGAN2LossConfig

    batch_size: int = 16
    separate_lr_template_offsets: Optional[
        float] = None  # Optionally, specify a different learning rate for learnable template offsets
    template_offsets_gradient_accumulation: int = 1
    freeze_generator: bool = False


GGHGeneratorType = Literal['gaussians', 'GSM', 'triplanes']


@dataclass
class GGHeadGANConfig(Config):
    generator_config: GGHeadConfig
    # generator_config: TriPlaneGaussianGeneratorConfig
    discriminator_config: GaussianDiscriminatorConfig
    generator_type: GGHGeneratorType = 'gaussians'


@dataclass
class GGHeadEvaluationConfig(Config):
    checkpoint: int = -1
    load_ema: bool = False


@dataclass
class GGHeadEvaluationResult(Config):
    fid100: Optional[float] = None
    fid1k: Optional[float] = None
    fid5k: Optional[float] = None
    fid10k: Optional[float] = None
    fid50k_full: Optional[float] = None

    def get_fid(self, count: int) -> Optional[float]:
        if count == 100:
            return self.fid100
        elif count == 1000:
            return self.fid1k
        elif count == 5000:
            return self.fid5k
        elif count == 10000:
            return self.fid10k
        elif count == 50000:
            return self.fid50k_full
        else:
            raise ValueError(f"Unsupported fid count {count}")


class BaseModelManager(
    ModelManager[
        None, GGHeadGANConfig, GGHeadGANOptimizerConfig, GGHeadImageFolderDatasetConfig, GGHeadTrainSetup, GGHeadEvaluationConfig, GGHeadEvaluationResult]):
    def __init__(self, model_type: str, run_name: str, checkpoints_sub_folder: Optional[str] = "checkpoints",
                 checkpoint_name_format: str = "checkpoint-$.pkl"):
        super().__init__(f"{GGH_MODELS_PATH}/{model_type}", run_name,
                         checkpoints_sub_folder=checkpoints_sub_folder, checkpoint_name_format=checkpoint_name_format)
        self._evaluations_sub_folder = 'evaluations'

    def get_wandb_folder(self) -> str:
        return f"{self._location}/wandb"

    def get_metric_path(self, metric_name: str) -> str:
        return f"{self._location}/metric-{metric_name}.jsonl"

    # ----------------------------------------------------------
    # Model & Checkpoints
    # ----------------------------------------------------------

    def _build_model(self, model_config: _ModelConfigType,
                     optimization_config: Optional[_OptimizationConfigType] = None, **kwargs) -> _ModelType:
        pass

    def _store_checkpoint(self, model: _ModelType, checkpoint_file_name: str, **kwargs):
        pass

    def _load_checkpoint(self, checkpoint_file_name: Union[str, int], load_ema: bool = False, **kwargs) -> _ModelType:
        checkpoint_path = f"{self._checkpoints_folder}/{checkpoint_file_name}"
        with open(checkpoint_path, "rb") as f:
            snapshot = pickle.load(f)

        if load_ema:
            model = snapshot['G_ema']
        else:
            model = snapshot['G']

        return model

    def load_discriminator(self, checkpoint_name_or_id: Union[str, int]) -> GaussianDiscriminator:
        checkpoint_id = self._resolve_checkpoint_id(checkpoint_name_or_id)
        checkpoint_file_name = self._checkpoints_folder.get_file_name_by_numbering(self._checkpoint_name_format,
                                                                                   checkpoint_id)

        checkpoint_path = f"{self._checkpoints_folder}/{checkpoint_file_name}"
        with open(checkpoint_path, "rb") as f:
            snapshot = pickle.load(f)

        discriminator = snapshot['D']
        return discriminator

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------

    def _get_evaluation_path(self, evaluation_config: GGHeadEvaluationConfig, relative: bool = False) -> str:
        if relative:
            evaluations_folder = self._evaluations_sub_folder
        else:
            evaluations_folder = f"{self._location}/{self._evaluations_sub_folder}"

        c = evaluation_config
        c_default = GGHeadEvaluationConfig()
        prefix = f"evaluation_ckpt-{c.checkpoint}"

        nb = NameBuilder(c, c_default, prefix=prefix, suffix='.json')
        nb.add('load_ema', 'ema')
        name = nb.get()

        output_path = f"{evaluations_folder}/{name}"
        return output_path

    def store_evaluation_result(self, evaluation_config: GGHeadEvaluationConfig,
                                evaluation_result: GGHeadEvaluationResult, overwrite: bool = True):
        full_evaluation_path = self._get_evaluation_path(evaluation_config)
        if Path(full_evaluation_path).exists() and not overwrite:
            print(f"Evaluation at {full_evaluation_path} already exists. Try merging...")
            existing_evaluation_result = self.load_evaluation_result(evaluation_config)

            merged_values = dict()
            for field_data in fields(evaluation_result):
                field_name = field_data.name
                new_value = getattr(evaluation_result, field_name)
                existing_value = getattr(existing_evaluation_result, field_name)
                # assert new_value is None or existing_value is None, \
                #     f"Cannot merge evaluations due to duplicate key: {field_name}"

                if new_value is not None:
                    value = new_value
                elif existing_value is not None:
                    value = existing_value
                else:
                    value = None
                merged_values[field_name] = value

            evaluation_result = GGHeadEvaluationResult(**merged_values)

        evaluation_path = self._get_evaluation_path(evaluation_config, relative=True)
        super().store_evaluation_result(evaluation_result, evaluation_path)

    def load_evaluation_result(self, evaluation_config: GGHeadEvaluationConfig) -> GGHeadEvaluationResult:
        evaluation_path = self._get_evaluation_path(evaluation_config, relative=True)
        return super().load_evaluation_result(evaluation_path)

    def has_evaluation_result(self, evaluation_config: GGHeadEvaluationConfig) -> bool:
        evaluation_path = self._get_evaluation_path(evaluation_config)
        return Path(evaluation_path).exists()

    def list_evaluated_checkpoint_ids(self) -> List[int]:
        evaluation_folder = Folder(f"{self._location}/{self._evaluations_sub_folder}")
        checkpoint_ids = evaluation_folder.list_file_numbering("evaluation_ckpt-$.json", return_only_numbering=True)

        return checkpoint_ids


_BaseModelManagerType = create_linked_type_var(_RunManagerType, bound=BaseModelManager)


class BaseModelFolder(ModelFolder[_BaseModelManagerType]):
    def __init__(self, model_type: str, prefix: str):
        super().__init__(f"{GGH_MODELS_PATH}/{model_type}", prefix, localize_via_run_name=True)
        self._name_format = f"{self._prefix}-$[_*]"  # Overwrite default name format s.t. there is a underscore after run name

    def new_run(self, name: Optional[str] = None) -> _BaseModelManagerType:
        return super().new_run(name)

    def open_run(self, run_name_or_id: Union[str, int]) -> _BaseModelManagerType:
        return super().open_run(run_name_or_id)
