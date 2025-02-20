from typing import List

import math
import torch
import torchvision
import numpy as np

from PIL import Image
from diffusers.utils import CONFIG_NAME
from diffusers.image_processor import PipelineImageInput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from transformers import ImageProcessingMixin

from .torch_dct import dct_2d, idct_2d


def rgb_to_ycbcr_tensor(image: torch.Tensor) -> torch.Tensor:
    img = image.float() / 255
    y = (img[:,:,:,0] * 0.299) + (img[:,:,:,1] * 0.587) + (img[:,:,:,2] * 0.114)
    cb = 0.5 + (img[:,:,:,0] * -0.168935) + (img[:,:,:,1] * -0.331665) + (img[:,:,:,2] * 0.50059)
    cr = 0.5 + (img[:,:,:,0] * 0.499813) + (img[:,:,:,1] * -0.418531) + (img[:,:,:,2] * -0.081282)
    ycbcr = torch.stack([y,cb,cr], dim=1)
    ycbcr = (ycbcr - 0.5) * 2
    return ycbcr


def ycbcr_tensor_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    ycbcr_img = (ycbcr / 2) + 0.5
    y = ycbcr_img[:,0,:,:]
    cb = ycbcr_img[:,1,:,:] - 0.5
    cr = ycbcr_img[:,2,:,:] - 0.5

    r = y + (cr * 1.402525)
    g = y + (cb * -0.343730) + (cr * -0.714401)
    b = y + (cb * 1.769905) + (cr * 0.000013)
    rgb = torch.stack([r,g,b], dim=-1).clamp(0,1)
    rgb = (rgb*255).to(torch.uint8)
    return rgb


def encode_single_channel_dct_2d(img: torch.Tensor, block_size: int=16, norm: str='ortho') -> torch.Tensor:
    batch_size, height, width = img.shape
    h_blocks = int(height//block_size)
    w_blocks = int(width//block_size)

    dct_tensor = torch.zeros((batch_size, h_blocks, w_blocks, block_size, block_size), device=img.device, dtype=torch.float32)
    for h in range(h_blocks):
        for w in range(w_blocks):
            dct_tensor[:, h,w] = img[:, h*block_size:(h+1)*block_size, w*block_size:(w+1)*block_size]
    dct_tensor = dct_2d(dct_tensor, norm=norm).reshape(batch_size, h_blocks, w_blocks, block_size*block_size)
    dct_tensor = dct_tensor.transpose(3,1).transpose(2,3)

    return dct_tensor


def decode_single_channel_dct_2d(img: torch.Tensor, norm: str='ortho') -> torch.Tensor:
    batch_size, combined_block_size, h_blocks, w_blocks = img.shape
    block_size = int(math.sqrt(combined_block_size))
    height = int(h_blocks*block_size)
    width = int(w_blocks*block_size)

    idct_tensor = torch.zeros((batch_size, h_blocks, w_blocks, block_size, block_size), device=img.device, dtype=torch.float32)
    for h in range(h_blocks):
        for w in range(w_blocks):
            idct_tensor[:, h:h+1, w:w+1] = img[:, :,h,w].reshape(batch_size, 1, 1, block_size, block_size)
    idct_tensor = idct_2d(idct_tensor, norm=norm)

    img_tensor = torch.zeros((batch_size, height, width), device=img.device, dtype=torch.float32)
    for h in range(h_blocks):
        for w in range(w_blocks):
            img_tensor[:, h*block_size:(h+1)*block_size, w*block_size:(w+1)*block_size] = idct_tensor[:, h,w]

    return img_tensor


def encode_jpeg_tensor(img: torch.Tensor, block_size: int=16, cbcr_downscale: int=2, norm: str='ortho') -> torch.Tensor:
    img = img[:, :, :(img.shape[-2]//block_size)*block_size, :(img.shape[-1]//block_size)*block_size] # crop to a multiply of block_size
    _, _, height, width = img.shape
    downsample = torchvision.transforms.Resize((height//cbcr_downscale, width//cbcr_downscale), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    down_img = downsample(img[:, 1:,:,:])
    y = encode_single_channel_dct_2d(img[:, 0, :,:], block_size=block_size, norm=norm)
    cb = encode_single_channel_dct_2d(down_img[:, 0, :,:], block_size=block_size//cbcr_downscale, norm=norm)
    cr = encode_single_channel_dct_2d(down_img[:, 1, :,:], block_size=block_size//cbcr_downscale, norm=norm)
    return torch.cat([y,cb,cr], dim=1)


def decode_jpeg_tensor(jpeg_img: torch.Tensor, block_size: int=16, cbcr_downscale: int=2, norm: str='ortho') -> torch.Tensor:
    _, _, h_blocks, w_blocks = jpeg_img.shape
    y_block_size = block_size*block_size
    cbcr_block_size = int((block_size//cbcr_downscale)*(block_size//cbcr_downscale))
    y = jpeg_img[:, :y_block_size]
    cb = jpeg_img[:, y_block_size:y_block_size+cbcr_block_size]
    cr = jpeg_img[:, y_block_size+cbcr_block_size:]
    y = decode_single_channel_dct_2d(y, norm=norm)
    cb = decode_single_channel_dct_2d(cb, norm=norm)
    cr = decode_single_channel_dct_2d(cr, norm=norm)
    upsample = torchvision.transforms.Resize((h_blocks*block_size, w_blocks*block_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    cb = upsample(cb)
    cr = upsample(cr)
    return torch.stack([y,cb,cr], dim=1)


class JPEGEncoder(ImageProcessingMixin, ConfigMixin):

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        block_size: int = 16,
        cbcr_downscale: int = 2,
        norm: str = "ortho",
        latents_std: List[float] = None,
        latents_mean: List[float] = None,
    ):
        self.block_size = block_size
        self.cbcr_downscale = cbcr_downscale
        self.norm = norm
        self.latents_std = latents_std
        self.latents_mean = latents_mean


    def encode(self, images: PipelineImageInput, device: str="cpu") -> torch.Tensor:
        """
        Encode RGB 0-255 image to JPEG Latents.

        Args:
            image (`PIL.Image.Image`, `np.ndarray` or `torch.Tensor`):
                The image input, can be a PIL image, numpy array or pytorch tensor.
                Must be an RGB image or a list of RGB images with 0-255 range and (batch_size, height, width, channels) shape.

        Returns:
            `torch.Tensor`:
                The encoded JPEG Latents.
        """

        if isinstance(images, list):
            combined_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    img = torch.from_numpy(np.asarray(img).copy()).unsqueeze(0)
                    combined_images.append(img)
                elif isinstance(img, np.ndarray):
                    if len(img.shape) == 3:
                        img = img.unsqueeze(0)
                    img = torch.from_numpy(img)
                    combined_images.append(img)
                elif isinstance(img, torch.Tensor):
                    if len(img.shape) == 3:
                        img = img.unsqueeze(0)
                    combined_images.append(img)
                else:
                    raise RuntimeError(f"Invalid input! Given: {type(img)} should be in ('torch.Tensor', 'np.ndarray', 'PIL.Image.Image')")
            combined_images = torch.cat(combined_images, dim=0)
        elif isinstance(images, Image.Image):
            combined_images = torch.from_numpy(np.asarray(images).copy()).unsqueeze(0)
        elif isinstance(images, np.ndarray):
            combined_images = torch.from_numpy(images)
            if len(combined_images.shape) == 3:
                combined_images = combined_images.unsqueeze(0)
        elif isinstance(images, torch.Tensor):
            combined_images = images
            if len(combined_images.shape) == 3:
                combined_images = combined_images.unsqueeze(0)
        else:
            raise RuntimeError(f"Invalid input! Given: {type(images)} should be in ('torch.Tensor', 'np.ndarray', 'PIL.Image.Image')")

        combined_images = combined_images.to(device)
        latents = rgb_to_ycbcr_tensor(combined_images)
        latents = encode_jpeg_tensor(latents, block_size=self.block_size, cbcr_downscale=self.cbcr_downscale, norm=self.norm)

        if self.latents_mean is not None:
            latents = latents - torch.tensor(self.latents_mean, device=device, dtype=torch.float32).view(1,-1,1,1)
        if self.latents_std is not None:
            latents = latents / torch.tensor(self.latents_std, device=device, dtype=torch.float32).view(1,-1,1,1)

        return latents

    def decode(self, latents: torch.Tensor, return_type: str="pil") -> PipelineImageInput:
        latents = latents.to(dtype=torch.float32)
        if self.latents_std is not None:
            latents = latents * torch.tensor(self.latents_std, device=latents.device, dtype=torch.float32).view(1,-1,1,1)
        if self.latents_mean is not None:
            latents = latents + torch.tensor(self.latents_mean, device=latents.device, dtype=torch.float32).view(1,-1,1,1)

        images = decode_jpeg_tensor(latents, block_size=self.block_size, cbcr_downscale=self.cbcr_downscale, norm=self.norm)
        images = ycbcr_tensor_to_rgb(images)

        if return_type == "pt":
            return images
        elif return_type == "np":
            return images.detach().cpu().numpy()
        elif return_type == "pil":
            image_list = []
            for i in range(images.shape[0]):
                image_list.append(Image.fromarray(images[i].detach().cpu().numpy()))
            return image_list
        else:
            raise RuntimeError(f"Invalid return_type! Given: {return_type} should be in ('pt', 'np', 'pil')")
