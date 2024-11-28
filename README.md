# GGHead: Fast and Generalizable 3D Gaussian Heads

[Paper](https://tobias-kirschstein.github.io/gghead/static/GGHead_paper.pdf) | [Video](https://youtu.be/1iyC74neQXc) | [Project Page](https://tobias-kirschstein.github.io/gghead/)

![](static/teaser.gif)

[Tobias Kirschstein](tobias-kirschstein.github.io), [Simon Giebenhain](https://simongiebenhain.github.io/), [Jiapeng Tang](https://tangjiapeng.github.io/), [Markos Georgopoulos](https://scholar.google.com/citations?user=id7vw0UAAAAJ&hl=en),
and [Matthias Nießner](https://www.niessnerlab.org/)  
**Siggraph Asia 2024**

# 1. Setup

## 1.1. Dependencies

1. Create conda environment `gghead` with newest PyTorch and CUDA 11.8:
    ```bash
    conda env create -f environment.yml
    ```
2. Ensure that `nvcc.exe` is taken from the conda environment and includes can be found:
    1. *[Linux]*
        ```bash
        conda activate gghead
        conda env config vars set CUDA_HOME=$CONDA_PREFIX
        conda activate base
        conda activate gghead
        ```
    2. *[Windows]*
        ```bash
        conda activate gghead
        conda env config vars set CUDA_HOME=$Env:CONDA_PREFIX
        conda env config vars set NVCC_PREPEND_FLAGS="-I$Env:CONDA_PREFIX\Library\include"
        conda activate base
        conda activate gghead
        ```
3. Check whether the correct `nvcc` can be found on the path via:
    ```bash
    nvcc --version
    ```
   which should say something like `release 11.8`.
4. Install Gaussian Splatting (which upon installation will compile CUDA kernels with `nvcc`):
    ```bash
    pip install gaussian_splatting@git+https://github.com/tobias-kirschstein/gaussian-splatting.git
    ```
    1. *[Optional]* If you compile the CUDA kernels on a different machine than the one you use for running code, you may need to manually specify the target
       GPU compute architecture for the compilation process via the `TORCH_CUDA_ARCH_LIST` environment variable:
       ```bash
       TORCH_CUDA_ARCH_LIST="8.0" pip install gaussian_splatting@git+https://github.com/tobias-kirschstein/gaussian-splatting.git
       ```
       Choose the correct compute architecture(s) that match your setup.
       Consult [this website](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) if unsure about the compute architecture of
       your graphics card.
    2. *[Troubleshooting]*
       On a Linux machine, if you run into
       ```
       gcc: fatal error: cannot execute 'cc1plus': posix_spawnp: No such file or directory
       ``` 
       or
       ```
       x86_64-conda_cos6-linux-gnu-cc: error trying to exec 'cc1plus': execvp: No such file or directory
       ```
       try
       ```
       conda install gxx_linux-64 gcc_linux-64
       ```
5. Finally install the `gghead` module via:
   ```bash
   pip install -e .
   ```

## 1.2. Environment Paths

All paths to data / models / renderings are defined by environment variables.  
Please create a file in your home directory in `~/.config/gghead/.env` with the following content:

```python
GGHEAD_DATA_PATH = "..."
GGHEAD_MODELS_PATH = "..."
GGHEAD_RENDERINGS_PATH = "..."
```

Replace the ... with the locations where data / models / renderings should be located on your machine.

- `GGHEAD_DATA_PATH`: Location of the FFHQ dataset and foreground masks. Only needed for training. See [Section 2](#2-data) for how to obtain the datasets.
- `GGHEAD_MODELS_PATH`: During training, model checkpoints and configs will be saved here. See [Section 4](#4-downloads) for downloading pre-trained models.
- `GGHEAD_RENDERINGS_PATH`: Video renderings of trained models will be stored here

If you do not like creating a config file in your home directory, you can instead hard-code the paths in the [env.py](src/gghead/env.py).

# 2. Data

Only needed for training. Download the zip files of the respective datasets and put them into `${GGHEAD_DATA_PATH}`.

| Dataset   | Images + Cameras                                                                  | Masks                                                                                                       |
|-----------|-----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| FFHQ-512  | [FFHQ_png_512.zip](https://kaldir.vc.in.tum.de/gghead/FFHQ_png_512.zip)           | [FFHQ_png_512_masks_modnet.zip](https://kaldir.vc.in.tum.de/gghead/FFHQ_png_512_masks_modnet.zip)           |
| FFHQ-1024 | [FFHQ_png_1024.zip](https://kaldir.vc.in.tum.de/gghead/FFHQ_png_1024.zip)         | [FFHQ_png_1024_masks_modnet.zip](https://kaldir.vc.in.tum.de/gghead/FFHQ_png_1024_masks_modnet.zip)         |
| AFHQ-512  | [afhq_v2_processed.zip](https://kaldir.vc.in.tum.de/gghead/afhq_v2_processed.zip) | [afhq_v2_processed_masks_modnet.zip](https://kaldir.vc.in.tum.de/gghead/afhq_v2_processed_masks_modnet.zip) |

# 3. Usage

## 3.1. Training

### 3.1.1. FFHQ 512

#### Stage 1 [256x256 pre-training]

```shell
BW_IMPLEMENTATION=1 python scripts/train_gghead.py ffhq FFHQ_png_512.zip 1 32 --kimg 6400
```

will start training GGHead on 1 GPU with a batch-size of 32 for 6400k images. To speed up training, you can use more GPUs.
E.g., `python scripts/train_gghead.py ffhq FFHQ_png_512.zip 4 32` will train on 4 GPUs with a batch size of 8 per GPU instead.  
Assets produced during training:
 - Loss curves will be logged to [weights&biases](https://wandb.ai) into a project called `generative-gaussian-heads`
 - Generated images will be logged periodically to `${GGHEAD_MODELS_PATH}/gghead/GGHEAD-xxx` every 200k train images
 - Checkpoints are stored in `${GGHEAD_MODELS_PATH}/gghead/GGHEAD-xxx/checkpoints` every 200k train images
 - Evaluation result with FID scores are stored in `${GGHEAD_MODELS_PATH}/gghead/GGHEAD-xxx/evaluations` every 200k train images

#### Stage 2 [512x512 training]

```shell
BW_IMPLEMENTATION=1 python scripts/train_gghead.py ffhq FFHQ_png_512.zip 1 32 --kimg 25000 --resume_run GGHEAD-xxx --overwrite_resolution 512 --overwrite_n_uniform_flame_vertices 512 --overwrite_lambda_tv_uv_rendering 100 --overwrite_lambda_beta_loss 1
```

Replace `GGHEAD-xxx` with the name of the run from stage 1, e.g., `GGHEAD-4` (everything after the number can be omitted).

#### Useful flags

- `use_vis_window`: for local debugging. Opens a dearpygui window showing live training progress
- `image_snapshot_ticks`: How often debug images will be stored during training. Default: every 50 ticks = every 200k train images (1 tick = 4k images)
- `metrics`: Which FID scores to compute during training. Default: `fid100,fid1k,fid10k`. Always computing FID score with 50k generated samples is expensive during training. In our experience, only generating 10k images is already enough to assess which run performs better.

### 3.1.2. FFHQ 1024

```shell
BW_IMPLEMENTATION=1 python scripts/train_gghead.py ffhq FFHQ_png_1024.zip 1 32 --kimg 27000 --resume_run GGHEAD-xxx --overwrite_resolution 1024
```

Replace `GGHEAD-xxx` with the name of the run from stage 2.

### 3.1.3. AFHQ

```shell
BW_IMPLEMENTATION=1 python scripts/train_gghead.py ffhq afhq_v2_processed.zip 1 32 --kimg 30000 --resume_run GGHEAD-xxx
```

Replace `GGHEAD-xxx` with the name of the run from stage 2.

## 3.2. Rendering

### 3.2.1. Sampling 3D heads

From a trained model `GGHEAD-xxx`, render short videos of randomly sampled 3D heads via:

```shell
python scripts/sample_heads.py GGHEAD-xxx
```

Replace `xxx` with the actual ID of the model.  
The generated videos will be placed into `${GGHEAD_RENDERINGS_PATH}/sampled_heads/`  
![GGHead Visualizer Showcase](static/example_sampled_head.gif)

### 3.2.2. Interpolations

From a trained model `GGHEAD-xxx`, render interpolation videos that morph between randomly sampled 3D heads via:

```shell
python scripts/render_interpolation.py GGHEAD-xxx
```

Replace `xxx` with the actual ID of the model.  
The generated videos will be placed into `${GGHEAD_RENDERINGS_PATH}/interpolations/`  
![GGHead Visualizer Showcase](static/example_interpolation.gif)

## 3.3. Evaluation

```shell
python scripts/evaluate_fid.py GGHEAD-xxx
```

Calculates FID score between generated images and the dataset images for the model `GGHEAD-xxx` (Replace `xxx` with the specific run ID that you want to evaluate).  
The default number of generated samples for FID calculation is 50000 which can be changed via `--fid`.  
The evaluation result will be printed in the terminal and also stored as a JSON file in `${GGHEAD_MODELS_PATH}/gghead/GGHEAD-xxx/evaluations`.

## 3.5. Example Notebooks

The [notebooks folder](notebooks) contains minimal examples on how to:

- Load a trained model, generate a 3D head and render it from an arbitrary viewpoint ([inference.ipynb](notebooks/inference.ipynb))

## 3.6. Visualizer

You can start the excellent GUI from EG3D and StyleGAN by running:

```shell
python visualizer.py
```

In the visualizer, you can select all checkpoints found in `${GGHEAD_MODELS_PATH}/gghead` and freely explore the generated heads in 3D.
![GGHead Visualizer Showcase](static/example_visualizer.gif)

# 4. Downloads

## 4.1. Pre-trained models

Put pre-trained models into `${GGHEAD_MODELS_PATH}/gghead`.

| Dataset   | GGHead model                                                                             |
|-----------|------------------------------------------------------------------------------------------|
| FFHQ-512  | [GGHEAD-1_ffhq512](https://nextcloud.tobias-kirschstein.de/index.php/s/49pojneNNMMmew4)  |
| FFHQ-1024 | [GGHEAD-2_ffhq1024](https://nextcloud.tobias-kirschstein.de/index.php/s/49pojneNNMMmew4) |
| AFHQ-512  | [GGHEAD-3-afhq512](https://nextcloud.tobias-kirschstein.de/index.php/s/49pojneNNMMmew4)  |

<hr>

```bibtex
@article{kirschstein2024gghead,
  title={GGHead: Fast and Generalizable 3D Gaussian Heads},
  author={Kirschstein, Tobias and Giebenhain, Simon and Tang, Jiapeng and Georgopoulos, Markos and Nie{\ss}ner, Matthias},
  journal={arXiv preprint arXiv:2406.09377},
  year={2024}
}
```

Contact [Tobias Kirschstein](mailto:tobias.kirschstein@tum.de) for questions, comments and reporting bugs, or open a GitHub issue.

