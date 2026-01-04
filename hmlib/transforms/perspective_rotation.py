from typing import Tuple, Union

import torch
from mmengine.registry import TRANSFORMS

from hmlib.bbox.box_functions import center, height, width
from hmlib.utils.distributions import ImageHorizontalGaussianDistribution
from hmlib.utils.gpu import StreamTensorBase
from hmlib.utils.image import image_height, image_width, rotate_image_batch, to_float_image


def _slow_to_tensor(tensor: Union[torch.Tensor, StreamTensorBase]) -> torch.Tensor:
    """
    Give up on the stream and get the sync'd tensor
    """
    if isinstance(tensor, StreamTensorBase):
        tensor._verbose = True
        return tensor.wait()
    return tensor


def image_wh(image: torch.Tensor):
    if image.shape[-1] in [3, 4]:
        return torch.tensor(
            [image.shape[-2], image.shape[-3]], dtype=torch.float, device=image.device
        )
    assert image.shape[1] in [3, 4]
    return torch.tensor([image.shape[-1], image.shape[-2]], dtype=torch.float, device=image.device)


@TRANSFORMS.register_module()
class HmPerspectiveRotation:

    def __init__(
        self,
        enabled: bool = True,
        fixed_edge_rotation_angle: Union[float, Tuple[float, float]] = 0.0,
        dtype: torch.dtype = torch.float,
        pre_clip: bool = False,
        image_label: str = "img",
        bbox_label: str = "camera_box",
    ):
        self._enabled = enabled
        self._pre_clip = pre_clip
        self._dtype = dtype
        self._fixed_edge_rotation_angle = fixed_edge_rotation_angle
        self._image_label = image_label
        self._bbox_label = bbox_label
        self._horizontal_image_gaussian_distribution = None
        self._zero_uint8 = None

    def __call__(self, results):
        if not self._enabled or self._fixed_edge_rotation_angle == 0:
            return results
        online_im = results.pop(self._image_label)
        current_box = results.pop(self._bbox_label)
        online_im = _slow_to_tensor(online_im)
        online_im = to_float_image(
            online_im,
            non_blocking=True,
            dtype=self._dtype,
        )
        src_image_width = image_width(online_im)
        rotated_images = []
        current_boxes = []
        for img, bbox in zip(online_im, current_box):
            rotation_point = center(bbox)
            width_center = src_image_width / 2
            if rotation_point[0] < width_center:
                mult = -1
            else:
                mult = 1

            gaussian = 1 - self._get_gaussian(src_image_width).get_gaussian_y_from_image_x_position(
                rotation_point[0],
                wide=True,
            )

            fixed_edge_rotation_angle = self._fixed_edge_rotation_angle
            if isinstance(fixed_edge_rotation_angle, (list, tuple)):
                assert len(fixed_edge_rotation_angle) == 2
                if rotation_point[0] < src_image_width // 2:
                    fixed_edge_rotation_angle = int(self._fixed_edge_rotation_angle[0])
                else:
                    fixed_edge_rotation_angle = int(self._fixed_edge_rotation_angle[1])

            angle = torch.ones((), dtype=torch.float, device=img.device) * (
                fixed_edge_rotation_angle - fixed_edge_rotation_angle * gaussian
            )
            angle *= mult

            # BEGIN PERFORMANCE HACK
            #
            # Chop off edges of image that won't be visible after a final crop
            # before we rotate in order to reduce the computation necessary
            # for the rotation (as well as other subsequent operations)
            #
            if self._pre_clip:
                pre_size = image_width(img), image_height(img)
                img, bbox = self.crop_working_image_width(image=img, current_box=bbox)
                post_size = image_width(img), image_height(img)
                if pre_size != post_size:
                    results["rotation_crop"] = {"from": pre_size, "to": post_size}
                rotation_point = center(bbox)
            #
            # END PERFORMANCE HACK
            #

            img = rotate_image_batch(
                img=img.unsqueeze(0),
                angle=angle,
                rotation_point=rotation_point,
            ).squeeze(0)
            rotated_images.append(img)
            current_boxes.append(bbox)
        del online_im
        results[self._image_label] = torch.stack(rotated_images)
        results[self._bbox_label] = torch.stack(current_boxes)

        return results

    def crop_working_image_width(self, image: torch.Tensor, current_box: torch.Tensor):
        """
        We try to only retain enough image to supply an arbitrary rotation
        about the center of the given bounding box with pixels,
        and offset that bounding box to be relative to the new (hopefully smaller)
        image
        """
        if self._zero_uint8 is None:
            self._zero_uint8 = torch.tensor(0, dtype=torch.uint8, device=image.device)

        bbox_w = width(current_box)
        bbox_h = height(current_box)
        bbox_c = center(current_box)
        assert bbox_w > 10  # Sanity
        assert bbox_h > 10  # Sanity
        # make sure we're the expected (albeit arbitrary) channels first
        assert image.shape[-1] in [3, 4]
        img_wh = image_wh(image)
        min_width_per_side = torch.sqrt(torch.square(bbox_w) + torch.square(bbox_h)) / 2
        clip_left = torch.max(self._zero_uint8, bbox_c[0] - min_width_per_side).to(
            torch.int64, non_blocking=True
        )
        clip_right = torch.min(img_wh[0] - 1, bbox_c[0] + min_width_per_side).to(
            torch.int64, non_blocking=True
        )
        if image.ndim == 3:
            # no batch dimension
            image = image[:, clip_left:clip_right, :]
        else:
            # with batch dimension
            image = image[:, :, clip_left:clip_right, :]
        if clip_left.device != current_box.device:
            clip_left = clip_left.to(current_box.device)
        current_box[0] -= clip_left
        current_box[2] -= clip_left

        return image, current_box

    def _get_gaussian(self, image_width: int):
        if self._horizontal_image_gaussian_distribution is None:
            self._horizontal_image_gaussian_distribution = ImageHorizontalGaussianDistribution(
                image_width, invert=True, show=False
            )
        else:
            assert image_width == self._horizontal_image_gaussian_distribution.width
        return self._horizontal_image_gaussian_distribution
