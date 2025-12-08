import copy

import torch

from hmlib.bbox.box_functions import center, height, width
from hmlib.transforms.perspective_rotation import HmPerspectiveRotation, _slow_to_tensor, image_wh
from hmlib.utils.distributions import ImageHorizontalGaussianDistribution
from hmlib.utils.image import image_height, image_width, rotate_image, to_float_image


class LegacyHmPerspectiveRotation:
    """
    Reference implementation matching the original HmPerspectiveRotation semantics.

    Used in tests to assert that the current implementation produces equivalent
    outputs for the default configuration.
    """

    def __init__(
        self,
        fixed_edge_rotation: bool = False,
        fixed_edge_rotation_angle=0.0,
        dtype: torch.dtype = torch.float,
        pre_clip: bool = False,
        image_label: str = "img",
        bbox_label: str = "camera_box",
    ):
        self._fixed_edge_rotation = fixed_edge_rotation
        self._pre_clip = pre_clip
        self._dtype = dtype
        self._fixed_edge_rotation_angle = fixed_edge_rotation_angle
        self._image_label = image_label
        self._bbox_label = bbox_label
        self._horizontal_image_gaussian_distribution = None
        self._zero_uint8 = None
        self._crop_half_width = None

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
        src_image_width = image_width(online_im)
        rotated_images = []
        current_boxes = []
        for img, bbox in zip(online_im, current_box):
            rotation_point = [int(i) for i in center(bbox)]
            width_center = src_image_width / 2
            mult = -1 if rotation_point[0] < width_center else 1

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

            if self._pre_clip:
                pre_size = image_width(img), image_height(img)
                img, bbox = self.crop_working_image_width(image=img, current_box=bbox)
                post_size = image_width(img), image_height(img)
                if pre_size != post_size:
                    results["rotation_crop"] = {"from": pre_size, "to": post_size}
                rotation_point = center(bbox)

            img = rotate_image(
                img=img,
                angle=angle,
                rotation_point=rotation_point,
            )
            rotated_images.append(img)
            current_boxes.append(bbox)

        current_box = torch.stack(current_boxes)
        results[self._image_label] = rotated_images
        results[self._bbox_label] = current_box
        return results

    def crop_working_image_width(self, image: torch.Tensor, current_box: torch.Tensor):
        if self._zero_uint8 is None:
            self._zero_uint8 = torch.tensor(0, dtype=torch.uint8, device=image.device)

        bbox_w = width(current_box)
        bbox_h = height(current_box)
        bbox_c = center(current_box)
        assert bbox_w > 10
        assert bbox_h > 10
        assert image.shape[-1] in [3, 4]
        img_wh = image_wh(image)

        if self._crop_half_width is None:
            min_width_per_side = torch.sqrt(torch.square(bbox_w) + torch.square(bbox_h)) / 2
            self._crop_half_width = min_width_per_side.detach()
        else:
            min_width_per_side = self._crop_half_width.to(device=current_box.device)

        clip_left = torch.max(self._zero_uint8, bbox_c[0] - min_width_per_side).to(
            torch.int64, non_blocking=True
        )
        clip_right = torch.min(img_wh[0] - 1, bbox_c[0] + min_width_per_side).to(
            torch.int64, non_blocking=True
        )
        if image.ndim == 3:
            image = image[:, clip_left:clip_right, :]
        else:
            image = image[:, :, clip_left:clip_right, :]
        if clip_left.device != current_box.device:
            clip_left = clip_left.to(current_box.device)
        current_box[0] -= clip_left
        current_box[2] -= clip_left
        return image, current_box

    def _get_gaussian(self, image_width_val: int):
        if self._horizontal_image_gaussian_distribution is None:
            self._horizontal_image_gaussian_distribution = ImageHorizontalGaussianDistribution(
                image_width_val, invert=True, show=False
            )
        else:
            assert image_width_val == self._horizontal_image_gaussian_distribution.width
        return self._horizontal_image_gaussian_distribution


def _make_test_batch(num_frames: int = 3):
    torch.manual_seed(0)
    H, W = 180, 320
    img = torch.rand(num_frames, H, W, 3, dtype=torch.float32)
    boxes = []
    for i in range(num_frames):
        x1 = 20 + 10 * i
        y1 = 30
        x2 = min(W - 20, x1 + 80 + 10 * i)
        y2 = min(H - 20, y1 + 60)
        boxes.append([x1, y1, x2, y2])
    camera_box = torch.tensor(boxes, dtype=torch.float32)
    return img, camera_box


def _make_edge_boxes(H: int, W: int):
    return torch.tensor(
        [
            [5.0, 40.0, 85.0, 120.0],  # near left edge
            [W - 90.0, 30.0, W - 5.0, 110.0],  # near right edge
            [W / 2 - 40.0, 20.0, W / 2 + 40.0, 100.0],  # centered
        ],
        dtype=torch.float32,
    )


def _run_pair(transform_cls, legacy_cls, *, pre_clip: bool, angle, boxes: torch.Tensor):
    img = torch.rand(boxes.shape[0], 180, 320, 3, dtype=torch.float32)

    new_tf = transform_cls(
        fixed_edge_rotation=True,
        fixed_edge_rotation_angle=angle,
        pre_clip=pre_clip,
        image_label="img",
        bbox_label="camera_box",
    )
    legacy_tf = legacy_cls(
        fixed_edge_rotation=True,
        fixed_edge_rotation_angle=angle,
        pre_clip=pre_clip,
        image_label="img",
        bbox_label="camera_box",
    )

    new_results = {"img": img.clone(), "camera_box": boxes.clone()}
    legacy_results = {"img": img.clone(), "camera_box": boxes.clone()}

    new_results = new_tf(copy.deepcopy(new_results))
    legacy_results = legacy_tf(copy.deepcopy(legacy_results))

    new_imgs = new_results["img"]
    legacy_imgs = legacy_results["img"]
    assert isinstance(new_imgs, list)
    assert isinstance(legacy_imgs, list)
    assert len(new_imgs) == len(legacy_imgs) == boxes.shape[0]

    for ni, li in zip(new_imgs, legacy_imgs):
        assert ni.shape == li.shape
        assert torch.allclose(ni, li, atol=1e-5, rtol=1e-5)

    assert torch.allclose(
        new_results["camera_box"], legacy_results["camera_box"], atol=1e-5, rtol=1e-5
    )


def should_perspective_rotation_match_legacy_without_pre_clip():
    img, camera_box = _make_test_batch(num_frames=3)
    _run_pair(HmPerspectiveRotation, LegacyHmPerspectiveRotation, pre_clip=False, angle=10.0, boxes=camera_box)  # type: ignore[arg-type]


def should_perspective_rotation_match_legacy_with_pre_clip():
    img, _ = _make_test_batch(num_frames=3)
    H, W = img.shape[1], img.shape[2]
    boxes = _make_edge_boxes(H, W)
    _run_pair(HmPerspectiveRotation, LegacyHmPerspectiveRotation, pre_clip=True, angle=10.0, boxes=boxes)  # type: ignore[arg-type]


def should_perspective_rotation_keep_fixed_width_when_configured():
    img, boxes = _make_test_batch(num_frames=3)
    H, W = img.shape[1], img.shape[2]

    transform = HmPerspectiveRotation(
        fixed_edge_rotation=True,
        fixed_edge_rotation_angle=10.0,
        pre_clip=True,
        image_label="img",
        bbox_label="camera_box",
        fixed_crop_half_width=True,
        crop_half_width_scale=1.0,
    )

    results = {"img": img.clone(), "camera_box": boxes.clone()}
    results = transform(results)
    rotated_imgs = results["img"]

    assert isinstance(rotated_imgs, list)
    widths = {int(image_width(t)) for t in rotated_imgs}
    heights = {int(image_height(t)) for t in rotated_imgs}
    assert len(widths) == 1
    assert len(heights) == 1

