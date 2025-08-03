from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL.Image import Image

from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import make_visible_image


def cv2_has_opengl() -> bool:
    return hasattr(cv2, "ogl") and hasattr(cv2.ogl, "Texture2D")


# Assuming you have a GPU tensor with shape [H, W, C] or [C, H, W]
# and values normalized between 0-1 or 0-255
def show_gpu_tensor(label: str, tensor: torch.Tensor, wait: bool = True) -> None:
    assert cv2_has_opengl()
    # Ensure tensor is in correct format [H, W, C]
    if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
        tensor = tensor.permute(1, 2, 0)

    # Create OpenGL window
    cv2.namedWindow(label, cv2.WINDOW_OPENGL)

    # Create OpenGL texture
    tex = cv2.ogl.Texture2D()

    # Register CUDA-OpenGL interop
    cuda_gl_interop = cv2.cuda_GpuMat()
    cuda_gl_interop.mapDevice(tex.texId())

    # Copy tensor data directly to OpenGL texture
    # Note: This assumes tensor is in CUDA memory
    cuda_gl_interop.upload(tensor.contiguous())

    # Display the texture
    cv2.ogl.render(tex)
    cv2.waitKey(1)  # Update display


def show_image(
    label: str,
    img: Union[str, torch.Tensor],
    wait: bool = True,
    enable_resizing: Union[bool, float, None] = None,
    scale: Optional[float] = None,
):
    if enable_resizing is None and scale is not None:
        enable_resizing = scale
    else:
        assert scale is None
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, Image):
        img = np.array(img)
    elif isinstance(img, StreamTensor):
        img = img.get()
    if img.ndim == 2:
        # grayscale
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    if img.ndim == 4:
        for i in img:
            cv2.imshow(
                label,
                make_visible_image(
                    i,
                    # scale_elements=255.0,
                    enable_resizing=enable_resizing,
                ),
            )
            cv2.waitKey(1 if not wait else 0)
    else:
        cv2.imshow(
            label,
            make_visible_image(
                img,
                # scale_elements=255.0,
                enable_resizing=enable_resizing,
                force_numpy=True,
            ),
        )
        cv2.waitKey(1 if not wait else 0)
