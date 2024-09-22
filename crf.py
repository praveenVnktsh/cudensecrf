import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from bilateral import bilateral_filter_torch_triton

class CRF(nn.Module):
    """
    Class for learning and inference in conditional random field model using mean field approximation
    and convolutional approximation in pairwise potentials term.

    Parameters
    ----------
    n_spatial_dims : int
        Number of spatial dimensions of input tensors.
    filter_size : int or sequence of ints
        Size of the gaussian filters in message passing.
        If it is a sequence its length must be equal to ``n_spatial_dims``.
    n_iter : int
        Number of iterations in mean field approximation.
    requires_grad : bool
        Whether or not to train CRF's parameters.
    returns : str
        Can be 'logits', 'proba', 'log-proba'.
    smoothness_weight : float
        Initial weight of smoothness kernel.
    smoothness_theta : float or sequence of floats
        Initial bandwidths for each spatial feature in the gaussian smoothness kernel.
        If it is a sequence its length must be equal to ``n_spatial_dims``.
    """

    def __init__(self, n_spatial_dims, filter_size=11, n_iter=5, requires_grad=True,
                 returns='logits', smoothness_weight=1, smoothness_theta=1):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.n_iter = n_iter
        self.filter_size = np.broadcast_to(filter_size, n_spatial_dims)
        self.returns = returns
        self.requires_grad = requires_grad

        self._set_param('smoothness_weight', smoothness_weight)
        self._set_param('inv_smoothness_theta', 1 / np.broadcast_to(smoothness_theta, n_spatial_dims))

    def _set_param(self, name, init_value):
        setattr(self, name, nn.Parameter(torch.tensor(init_value, dtype=torch.float, requires_grad=self.requires_grad)))

    def forward(self, x, img, spatial_spacings=None, verbose=False):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with negative unary potentials, e.g. the CNN's output.
        spatial_spacings : array of floats or None
            Array of shape ``(batch_size, len(spatial))`` with spatial spacings of tensors in batch ``x``.
            None is equivalent to all ones. Used to adapt spatial gaussian filters to different inputs' resolutions.
        verbose : bool
            Whether to display the iterations using tqdm-bar.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``
            with logits or (log-)probabilities of assignment to each class.
        """
        batch_size, n_classes, *spatial = x.shape
        assert len(spatial) == self.n_spatial_dims

        # binary segmentation case
        if n_classes == 1:
            x = torch.cat([x, torch.zeros(x.shape).to(x)], dim=1)

        if spatial_spacings is None:
            spatial_spacings = np.ones((batch_size, self.n_spatial_dims))

        negative_unary = x.clone()

        for i in tqdm(range(self.n_iter), disable=not verbose):
            # normalizing
            q = F.softmax(x, dim=1)

            # message passing
            messages = self.smoothness_weight * self._smoothing_filter(q, spatial_spacings)

            filtered_potential = bilateral_filter_torch_triton(
                    img=img,          # RGB image for batch b
                    spatial_sigma=1,
                    range_sigma=1,
                    kernel_radius=11,
                )

            # Store the filtered result
            filtered_messages = messages + filtered_potential.unsqueeze(0)

            # compatibility transform
            x = self._compatibility_transform(filtered_messages)

            # adding unary potentials
            x = negative_unary - x

        if self.returns == 'logits':
            output = x
        elif self.returns == 'proba':
            output = F.softmax(x, dim=1)
        elif self.returns == 'log-proba':
            output = F.log_softmax(x, dim=1)
        else:
            raise ValueError("Attribute ``returns`` must be 'logits', 'proba' or 'log-proba'.")

        if n_classes == 1:
            output = output[:, 0] - output[:, 1] if self.returns == 'logits' else output[:, 0]
            output.unsqueeze_(1)

        return output

    def _smoothing_filter(self, x, spatial_spacings):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with negative unary potentials, e.g. logits.
        spatial_spacings : torch.tensor or None
            Tensor of shape ``(batch_size, len(spatial))`` with spatial spacings of tensors in batch ``x``.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        return torch.stack([self._single_smoothing_filter(x[i], spatial_spacings[i]) for i in range(x.shape[0])])

    @staticmethod
    def _pad(x, filter_size):
        padding = []
        for fs in filter_size:
            padding += 2 * [fs // 2]

        return F.pad(x, list(reversed(padding)))  # F.pad pads from the end

    def _single_smoothing_filter(self, x, spatial_spacing):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(n, *spatial)``.
        spatial_spacing : sequence of len(spatial) floats

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(n, *spatial)``.
        """
        x = self._pad(x, self.filter_size)
        for i, dim in enumerate(range(1, x.ndim)):
            # reshape to (-1, 1, x.shape[dim])
            x = x.transpose(dim, -1)
            shape_before_flatten = x.shape[:-1]
            x = x.flatten(0, -2).unsqueeze(1)

            # 1d gaussian filtering
            kernel = self._create_gaussian_kernel1d(self.inv_smoothness_theta[i], spatial_spacing[i],
                                                   self.filter_size[i]).view(1, 1, -1).to(x)
            x = F.conv1d(x, kernel)

            # reshape back to (n, *spatial)
            x = x.squeeze(1).view(*shape_before_flatten, x.shape[-1]).transpose(-1, dim)

        return x

    @staticmethod
    def _create_gaussian_kernel1d(inverse_theta, spacing, filter_size):
        """
        Parameters
        ----------
        inverse_theta : torch.tensor
            Tensor of shape ``(,)``
        spacing : float
        filter_size : int

        Returns
        -------
        kernel : torch.tensor
            Tensor of shape ``(filter_size,)``.
        """
        distances = spacing * torch.arange(-(filter_size // 2), filter_size // 2 + 1).to(inverse_theta)
        kernel = torch.exp(-(distances * inverse_theta) ** 2 / 2)
        zero_center = torch.ones(filter_size).to(kernel)
        zero_center[filter_size // 2] = 0
        return kernel * zero_center

    def _compatibility_transform(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape ``(batch_size, n_classes, *spatial)``.

        Returns
        -------
        output : torch.tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        labels = torch.arange(x.shape[1])
        compatibility_matrix = self._compatibility_function(labels, labels.unsqueeze(1)).to(x)

        return torch.einsum('ij..., jk -> ik...', x, compatibility_matrix)

    @staticmethod
    def _compatibility_function(label1, label2):
        """
        Input tensors must be broadcastable.

        Parameters
        ----------
        label1 : torch.Tensor
        label2 : torch.Tensor

        Returns
        -------
        compatibility : torch.Tensor
        """
        return -(label1 == label2).float()
    

def load_tile(root):
    import glob
    stacked = []
    H = 512
    for path in glob.glob(root + '*.tiff'):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        img = cv2.resize(img, (H, H))
        stacked.append(img)
    return np.stack(stacked, axis=0)

def dump_image(img, path):
    # img = img
    img = F.softmax(img, dim=0).argmax(dim=0).detach().cpu().numpy()
    # img = (img * 255).astype(np.uint8)
    new_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    new_img[..., 1] = img * 255 / 12
    cv2.imwrite(path, new_img)

if __name__ == "__main__":
    import cv2
    import time
    model = CRF(n_spatial_dims=1).cuda()

    batch_size, n_channels, spatial = 1, 3, (512, 512)
    x = torch.zeros(batch_size, n_channels, *spatial)
    # profiler = cProfile.Profile()
    # profiler.enable()

    starttime = time.time()
    tile = np.load('data/tile.npy')
    anno = np.load('data/intensity.npy')
    anno = torch.from_numpy(anno).float().cuda()
    anno = F.interpolate(anno.unsqueeze(0).unsqueeze(0), spatial, mode='bilinear').squeeze(0).squeeze(0)
    tile = torch.from_numpy(tile).float().cuda()
    
    log_proba = model(tile, anno)
    print(f'Time: {time.time() - starttime}')
    dump_image(log_proba, 'outputs/log_proba.png')
