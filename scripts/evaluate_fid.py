from pathlib import Path
from typing import Union

import torch
import tyro
from eg3d.metrics.metric_main import calc_metric, register_metric

from gghead.env import GGHEAD_DATA_PATH
from gghead.model_manager.base_model_manager import GGHeadEvaluationConfig, GGHeadEvaluationResult
from gghead.model_manager.finder import find_model_manager
from gghead.util.metrics import fid100, fid1k, fid50k_full, fid5k, fid10k


def main(run_names: str,
         /,
         fid: int = 50000,                  # How many samples should be generated to compute FID score
         load_ema: bool = True,             # Whether to evaluate the exponentially moving average version of the generator
         checkpoint: Union[int, str] = -1,  # Which checkpoint to evaluate. 'all' will evaluate all checkpoints. 'remaining' will evaluate all that have not been evaluated yet, -1 will evaluate the latest checkpoint, otherwise will evaluate the specified checkpoint iteration
         local: bool = False,               # Useful when locally evaluating a model trained somewhere else: Replaces the path to the dataset with the local data path
         gpus: int = 1):
    # set_start_method('spawn') is essential for the combination of DataLoader and zip Dataset to work
    # Otherwise, get random errors on Unix systems where fork is the default start method since parallel reading of zipfiles doesn't work there
    torch.multiprocessing.set_start_method('spawn')

    for run_name in run_names.split(','):

        try:
            model_manager = find_model_manager(run_name)
            if checkpoint == 'all':
                checkpoint_ids = model_manager.list_checkpoint_ids()
            elif checkpoint == 'remaining':
                candidate_checkpoint_ids = model_manager.list_checkpoint_ids()
                checkpoint_ids = []
                for checkpoint_id in candidate_checkpoint_ids:
                    evaluation_config = GGHeadEvaluationConfig(checkpoint=checkpoint_id, load_ema=load_ema)
                    if not model_manager.has_evaluation_result(evaluation_config):
                        checkpoint_ids.append(checkpoint_id)
                    else:
                        evaluation_result = model_manager.load_evaluation_result(evaluation_config)
                        if evaluation_result.get_fid(fid) is None:
                            checkpoint_ids.append(checkpoint_id)
            else:
                if isinstance(checkpoint, int):
                    checkpoint_ids = [model_manager._resolve_checkpoint_id(checkpoint)]
                else:
                    checkpoint_ids = [model_manager._resolve_checkpoint_id(int(ckpt)) for ckpt in checkpoint.split(',')]

            for checkpoint_id in checkpoint_ids:
                model = model_manager.load_checkpoint(checkpoint_id, load_ema=load_ema).cuda()
                dataset_config = model_manager.load_dataset_config()

                if local:
                    dataset_config.path = f"{GGHEAD_DATA_PATH}/{Path(dataset_config.path).name}"

                register_metric(fid100)
                register_metric(fid1k)
                register_metric(fid50k_full)
                if fid == 100:
                    fid_metric = fid100
                elif fid == 1000:
                    fid_metric = fid1k
                elif fid == 5000:
                    fid_metric = fid5k
                elif fid == 10000:
                    fid_metric = fid10k
                elif fid == 50000:
                    fid_metric = fid50k_full
                else:
                    raise ValueError(f"Wrong FID count {fid}")

                result_dict = calc_metric(metric=fid_metric.__name__, G=model,
                                          dataset_kwargs=dataset_config.get_eval_dict(), num_gpus=gpus, rank=0, device='cuda')

                evaluation_config = GGHeadEvaluationConfig(checkpoint=checkpoint_id, load_ema=load_ema)
                evaluation_result = GGHeadEvaluationResult(**result_dict['results'])

                print("===========================")
                print(f"Evaluation for {run_name} - checkpoint {checkpoint_id}")
                print("===========================")
                print(evaluation_result)
                model_manager.store_evaluation_result(evaluation_config, evaluation_result, overwrite=False)
        except Exception as e:
            print(f"Skipping {run_name} due to {e}")


if __name__ == '__main__':
    tyro.cli(main)
