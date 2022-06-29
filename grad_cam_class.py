from typing import Optional

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.transforms.functional import resize

from app.dataset import get_val_transform


class ViTGradCam:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.grad_cam = self._get_grad_cam()

    @staticmethod
    def reshape_transform(tensor, height=32, width=32):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    @staticmethod
    def image_to_input(image_path: str) -> torch.Tensor:
        transform = get_val_transform(512)
        img = Image.open(image_path).convert("L")
        inp = transform(img).unsqueeze(0)
        return inp

    def _get_grad_cam(self) -> GradCAMPlusPlus:
        target_layer = [self.model.blocks[-1].norm1]
        grad_cam = GradCAMPlusPlus(
            model=self.model,
            target_layers=target_layer,
            reshape_transform=self.reshape_transform,
            use_cuda=True,
        )
        return grad_cam

    def get_cam(self, image_path: str, label: Optional[int] = None) -> np.ndarray:
        inp = self.image_to_input(image_path)
        if label is not None:
            label = [ClassifierOutputTarget(label)]
        return self.grad_cam(inp, label, aug_smooth=True)

    def visualize(self, image_path: str, label: Optional[int] = None):
        cam = self.get_cam(image_path, label)[0]

        img = Image.open(image_path).convert("RGB")
        img = resize(img, (512, 512))
        img = np.asarray(img)
        img = np.float32(img) / 255
        vis = show_cam_on_image(img, cam, use_rgb=True)
        return Image.fromarray(vis)

    __call__ = visualize


class DenseNetGradCam(ViTGradCam):
    def _get_grad_cam(self) -> GradCAMPlusPlus:
        target_layer = [self.model.features[-1]]
        grad_cam = GradCAMPlusPlus(
            model=self.model,
            target_layers=target_layer,
            use_cuda=True,
        )
        return grad_cam
