from eg3d.metrics import frechet_inception_distance
from eg3d.metrics.metric_main import register_metric
from gghead.dataset.image_folder_dataset import GGHeadMaskImageFolderDataset, GGHeadImageFolderDatasetConfig


@register_metric
def fid100(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.dataset = GGHeadMaskImageFolderDataset(GGHeadImageFolderDatasetConfig(**opts.dataset_kwargs))
    fid = frechet_inception_distance.compute_fid(opts, max_real=100, num_gen=100)
    return dict(fid100=fid)


@register_metric
def fid1k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.dataset = GGHeadMaskImageFolderDataset(GGHeadImageFolderDatasetConfig(**opts.dataset_kwargs))
    fid = frechet_inception_distance.compute_fid(opts, max_real=1000, num_gen=1000)
    return dict(fid1k=fid)

# @register_metric
# def fid1k_broken(opts):
#     opts.dataset_kwargs.update(max_size=None, xflip=False)
#     opts.dataset = GGHMaskImageFolderDataset(GGHImageFolderDatasetConfig(**opts.dataset_kwargs))
#     fid = frechet_inception_distance.compute_fid(opts, max_real=250, num_gen=1000)
#     return dict(fid1k=fid)


@register_metric
def fid5k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.dataset = GGHeadMaskImageFolderDataset(GGHeadImageFolderDatasetConfig(**opts.dataset_kwargs))
    fid = frechet_inception_distance.compute_fid(opts, max_real=5000, num_gen=5000)
    return dict(fid5k=fid)


@register_metric
def fid10k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.dataset = GGHeadMaskImageFolderDataset(GGHeadImageFolderDatasetConfig(**opts.dataset_kwargs))
    fid = frechet_inception_distance.compute_fid(opts, max_real=10000, num_gen=10000)
    return dict(fid10k=fid)


@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.dataset = GGHeadMaskImageFolderDataset(GGHeadImageFolderDatasetConfig(**opts.dataset_kwargs))
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)
