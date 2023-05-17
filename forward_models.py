import torch


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('../logs/loss_functions.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def cumprod_exclusive(tensor, dim=-2):
    cumprod = torch.cumprod(tensor, dim)
    cumprod = torch.roll(cumprod, 1, dim)
    cumprod[..., 0, :] = 1.0
    return cumprod


def compute_transmittance_weights(pred_sigma, t_intervals):
    pred_alpha = 1.-torch.exp(-torch.relu(pred_sigma)*t_intervals)
    pred_weights = pred_alpha * cumprod_exclusive(1.-pred_alpha+1e-10, dim=-2)
    return pred_weights


def compute_tomo_radiance(pred_weights, pred_rgb):
    pred_rgb_pos = pred_rgb.clone()
    pred_pixel_samples = torch.sum(pred_rgb_pos*pred_weights, dim=-2)
    return pred_pixel_samples


def compute_transmittance_weights_piecewise(pred_sigma, t_intervals, ncuts=32):
    shape_in = pred_sigma.shape
    logger.debug(f"shape_in: {shape_in}") 
    *others, nsamples_per_ray, ndims = shape_in
    logger.debug(f'nsamples_per_ray: {nsamples_per_ray}')
    logger.debug(f'ndims: {ndims}')
    ncuts_per_ray = ncuts
    logger.debug(f'num_cuts_per_ray: {ncuts_per_ray}')
    nsamples_per_cut = nsamples_per_ray//ncuts_per_ray
    logger.debug(f'nsamples_per_cut: {nsamples_per_cut}')

    logger.debug(f'pred_sigma.shape before reshape: {pred_sigma.shape}')
    pred_sigma = pred_sigma.reshape(-1, ncuts_per_ray, nsamples_per_cut, ndims)
    logger.debug(f'pred_sigma.shape after reshape: {pred_sigma.shape}')

    logger.debug(f't_intervals.shape before reshape: {t_intervals.shape}')
    t_intervals = t_intervals.reshape(-1, ncuts_per_ray, nsamples_per_cut, ndims)
    logger.debug(f't_intervals.shape after reshape: {t_intervals.shape}')

    pred_sigma_mean = torch.relu(torch.mean(pred_sigma, dim=-2))
    t_intervals_sum = torch.sum(t_intervals, dim=-2)

    pred_alpha = 1.-torch.exp(-pred_sigma_mean*t_intervals_sum)
    pred_weights = pred_alpha * cumprod_exclusive(1.-pred_alpha+1e-10, dim=-2)

    return pred_weights


def compute_tomo_radiance_piecewise(pred_weights, pred_rgb, ncuts_per_ray=32):
    shape_in = pred_rgb.shape
    *others, nsamples_per_ray, ndims = shape_in

    nsamples_per_cut = nsamples_per_ray//ncuts_per_ray
    pred_rgb = pred_rgb.reshape(-1, ncuts_per_ray, nsamples_per_cut, ndims)
    pred_rgb = torch.sigmoid(torch.mean(pred_rgb, dim=-2))

    pred_pixel_samples = torch.sum(pred_rgb*pred_weights, dim=-2)

    return pred_pixel_samples


def compute_rgb_integral(pred_rgb):
    pred_pixel_samples = torch.sum(pred_rgb, dim=-2)  # line integral
    return pred_pixel_samples


def compute_tomo_depth(pred_weights, zs):
    pred_depth = torch.sum(pred_weights*zs, dim=-2)
    return pred_depth


def compute_disp_from_depth(pred_depth, pred_weights):
    pred_disp = 1. / torch.max(torch.tensor(1e-10).to(pred_depth.device),
                               pred_depth / torch.sum(pred_weights, -2))
    return pred_disp
