from typing import Optional, Tuple, Union

import torch
from mmengine.registry import TRANSFORMS

from hmlib.bbox.box_functions import center, height, width
from hmlib.utils.distributions import ImageHorizontalGaussianDistribution
from hmlib.utils.gpu import StreamTensorBase
from hmlib.utils.image import image_height, image_width, rotate_image, to_float_image


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


def soft_new_zeros(t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(shape, dtype=t.dtype, device=t.device)


@TRANSFORMS.register_module()
class HmPerspectiveRotation:
    """Optional pre-rotation crop around a camera box.

    When ``pre_clip`` is True, this transform crops a vertical strip centered on
    the box center before rotating. The per-side crop radius defaults to the
    rectangle's circumscribed-circle radius (``sqrt(w^2 + h^2) / 2``) optionally
    scaled by ``crop_half_width_scale``. This is sufficient to contain the box
    for any in-plane rotation angle; using ``fixed_crop_half_width=True`` caches
    that radius from the first frame so the working image width stays constant.
    """

    def __init__(
        self,
        fixed_edge_rotation: bool = True,
        fixed_edge_rotation_angle: Union[float, Tuple[float, float]] = 0.0,
        dtype: torch.dtype = torch.float,
        pre_clip: bool = False,
        image_label: str = "img",
        bbox_label: str = "camera_box",
        fixed_crop_half_width: bool = True,
        crop_half_width_scale: float = 1.0,
    ):
        self._fixed_edge_rotation = fixed_edge_rotation
        self._pre_clip = pre_clip
        self._dtype = dtype
        self._fixed_edge_rotation_angle = fixed_edge_rotation_angle
        self._image_label = image_label
        self._bbox_label = bbox_label
        self._horizontal_image_gaussian_distribution = None
        self._crop_half_width: Optional[torch.Tensor] = None
        self._working_crop_width: Optional[int] = None
        self._image_width: Optional[int] = None
        self._image_height: Optional[int] = None
        self._fixed_crop_half_width = fixed_crop_half_width
        self._crop_half_width_scale = float(crop_half_width_scale)
        assert self._crop_half_width_scale >= 1.0

    def __call__(self, results):
        if not self._fixed_edge_rotation or self._fixed_edge_rotation_angle == 0:
            return results
        online_im = results.pop(self._image_label)
        current_box = results.pop(self._bbox_label)
        online_im = _slow_to_tensor(online_im)
        online_im = to_float_image(
            online_im,
            non_blocking=True,
            dtype=self._dtype,
        )
        src_image_width = int(image_width(online_im))
        if self._image_width is None:
            self._image_width = src_image_width
        else:
            # Expect a fixed source width across calls in a given pipeline.
            assert src_image_width == self._image_width
        rotated_images = []
        current_boxes = []
        for img, bbox in zip(online_im, current_box):
            rotation_point = [int(i) for i in center(bbox)]
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

            angle = fixed_edge_rotation_angle - fixed_edge_rotation_angle * gaussian
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
            assert angle.device.type == "cpu"
            img = rotate_image(
                img=img,
                angle=angle.item(),
                rotation_point=rotation_point,
            )
            rotated_images.append(img)
            current_boxes.append(bbox)
        del online_im
        # online_im = torch.stack(rotated_images)
        # results[self._image_label] = online_im
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
        bbox_w = width(current_box)
        bbox_h = height(current_box)
        bbox_c = center(current_box)
        assert bbox_w > 10  # Sanity
        assert bbox_h > 10  # Sanity
        # make sure we're the expected (albeit arbitrary) channels-last
        assert image.shape[-1] in [3, 4]
        img_wh = image_wh(image)
        img_width = img_wh[0]

        # Base half-width per side: radius of the rectangle's circumscribed circle,
        # optionally scaled to retain additional margin.
        base_half_width = torch.sqrt(torch.square(bbox_w) + torch.square(bbox_h)) / 2
        base_half_width = base_half_width * self._crop_half_width_scale

        if self._fixed_crop_half_width:
            # Cache a fixed half-width (per side) from the first frame so the
            # working image width remains constant across frames.
            if self._crop_half_width is None:
                self._crop_half_width = base_half_width.detach()
            min_width_per_side = self._crop_half_width.to(device=current_box.device)
        else:
            min_width_per_side = base_half_width

        crop_width = self._resolve_working_crop_width(
            min_width_per_side=min_width_per_side, img_width=img_width
        )
        half_width = crop_width * 0.5
        clip_left = torch.floor(bbox_c[0] - half_width)
        clip_right = clip_left + crop_width

        pad_left = torch.clamp(-clip_left, min=0)
        slice_left = torch.clamp(clip_left, min=0)
        slice_right = torch.clamp(clip_right, max=img_width)

        slice_left_int = int(slice_left.item())
        slice_right_int = int(slice_right.item())
        pad_left_int = int(pad_left.item())
        assert self._working_crop_width is not None
        target_width_int = self._working_crop_width

        if image.ndim == 3:
            # no batch dimension
            sliced = image[:, slice_left_int:slice_right_int, :]
            image = soft_new_zeros(image, (image.shape[0], target_width_int, image.shape[2]))
            image[:, pad_left_int : pad_left_int + sliced.shape[1], :] = sliced
        else:
            # with batch dimension
            sliced = image[:, :, slice_left_int:slice_right_int, :]
            image = soft_new_zeros(
                image, (image.shape[0], image.shape[1], target_width_int, image.shape[3])
            )
            image[:, :, pad_left_int : pad_left_int + sliced.shape[2], :] = sliced

        offset = (slice_left - pad_left).to(device=current_box.device, dtype=current_box.dtype)
        current_box[0] -= offset
        current_box[2] -= offset

        return image, current_box

    def _get_gaussian(self, image_width: int):
        if self._horizontal_image_gaussian_distribution is None:
            self._horizontal_image_gaussian_distribution = ImageHorizontalGaussianDistribution(
                image_width, invert=True, show=False
            )
        else:
            assert image_width == self._horizontal_image_gaussian_distribution.width
        return self._horizontal_image_gaussian_distribution

    def _resolve_working_crop_width(
        self, min_width_per_side: torch.Tensor, img_width: torch.Tensor
    ) -> torch.Tensor:
        """
        Pick a fixed working width for the pre-clip crop so every frame shares the
        same shape. We cap it at the source width to avoid expanding the work area.
        """
        if self._working_crop_width is None:
            width = torch.ceil(min_width_per_side * 2.0)
            width = torch.clamp(width, min=2.0, max=img_width)
            self._working_crop_width = int(width.item())
        assert self._working_crop_width is not None
        return min_width_per_side.new_tensor(float(self._working_crop_width))
