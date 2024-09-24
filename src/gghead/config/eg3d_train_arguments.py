from dataclasses import field, dataclass
from typing import Optional, Literal, List


@dataclass
class EG3DTrainArguments:
    # Required.
    outdir: str  # Where to save the results
    cfg: str  # Base configuration
    data: str  # Training data [ZIP|DIR]
    gpus: int  # Number of GPUs to use
    batch: int  # Total batch size
    gamma: float  # R1 regularization weight

    # Optional features.
    cond: bool = True  # Train conditional models
    mirror: bool = False  # Enable dataset x-flips
    aug: Literal['noaug', 'ada', 'fixed'] = 'noaug'  # Augmentation mode
    resume: Optional[str] = None  # Resume from given network pickle [PATH|URL]
    freezed: int = 0  # Freeze first layers of D
    use_calibration: bool = False  # Whether to use poses from dataset_calibration_fitted.json

    # Misc settings.
    desc: Optional[str] = None  # String to include in result dir name
    metrics: List[str] = field(default_factory=lambda: ['fid50k_full'])  # Quality metrics
    kimg: int = 25000  # Total training duration
    tick: int = 4  # How often to print progress [KIMG]
    snap: int = 50  # How often to save snapshots [TICKS]
    seed: int = 0  # Random seed
    nobench: bool = False  # Disable cuDNN benchmarking
    workers: int = 3  # DataLoader worker processes
    dry_run: bool = False  # Print training options and exit

    # Generator Pose Conditioning
    gen_pose_cond: bool = False  # If true, enable generator pose conditioning
    c_scale: float = 1  # Scale factor for generator pose conditioning
    c_noise: float = 0  # Add noise for generator pose conditioning
    gpc_reg_prob: float = 0.5  # Strength of swapping regularization. None means no generator pose conditioning, i.e. condition with zeros
    gpc_reg_fade_kimg: int = 1000  # Length of swapping prob fade

    disc_c_noise: float = 0  # Strength of discriminator pose conditioning regularization, in standard deviations

    # fp16
    sr_num_fp16_res: int = 4  # Number of fp16 layers in superresolution
    g_num_fp16_res: int = 0  # Number of fp16 layers in generator
    d_num_fp16_res: int = 4  # Number of fp16 layers in discriminator

    # Neural Rendering
    neural_rendering_resolution_initial: int = 64  # Resolution to render at
    neural_rendering_resolution_final: Optional[int] = None  # Final resolution to render at, if blending
    neural_rendering_resolution_fade_kimg: int = 1000  # Kimg to blend resolution over
    blur_fade_kimg: int = 200  # Blur over how many
    resume_blur: bool = False  # Enable to blur even on resume

    # TriPlane density regularization
    density_reg: float = 0.25  # Density regularization strength
    density_reg_every: int = 4  # lazy density reg
    density_reg_p_dist: float = 0.004  # density regularization strength
    reg_type: Literal['l1', 'l1-alt', 'monotonic-detach', 'monotonic-fixed', 'total-variation'] = 'l1'  # Type of regularization

    # Super-resolution
    sr_noise_mode: Literal['random', 'none'] = 'none'  # Type of noise for superresolution
    sr_first_cutoff: int = 2  # First cutoff for AF superresolution
    sr_first_stopband: float = 2 ** 2.1  # First cutoff for AF superresolution
    style_mixing_prob: float = 0  # Style-mixing regularization probability for training
    sr_module: Optional[str] = None  # Superresolution module override

    decoder_lr_mul: float = 1  # decoder learning rate multiplier

    # Misc hyperparameters.
    p: float = 0.2  # Probability for --aug=fixed
    target: float = 0.6  # Target value for --aug=ada
    batch_gpu: Optional[int] = None  # Limit batch size per GPU
    cbase: int = 32768  # Capacity multiplier
    cmax: int = 512  # Max. feature maps
    glr: Optional[float] = None  # G learning rate  [default: varies]
    dlr: float = 0.002  # D learning rate
    map_depth: int = 2  # Mapping network depth  [default: varies]
    mbstd_group: int = 4  # Minibatch std group size

    # Gaussian specific
    use_gaussians: bool = False