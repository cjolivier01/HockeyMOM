# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Tuple, Union

import cv2  # kept for optional window display only (not used for drawing now)
import mmcv  # kept for color parsing compatibility
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.registry import VISUALIZERS
from mmpose.structures import PoseDataSample

from hmlib.vis.pt_text import draw_text as torch_draw_text

# New torch-only drawing utilities
from hmlib.vis.pt_visualization import draw_box as torch_draw_box
from hmlib.vis.pt_visualization import draw_circle as torch_draw_circle
from hmlib.vis.pt_visualization import draw_line as torch_draw_line

from .pytorch_backend_visualizer import PytorchBackendVisualizer

try:
    from mmpose.visualization.simcc_vis import SimCCVisualizer
except Exception:
    SimCCVisualizer = None


def _get_adaptive_scales(
    areas: Union[np.ndarray, torch.Tensor], min_area: int = 800, max_area: int = 30000
) -> Union[np.ndarray, torch.Tensor]:
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``min_area``, the scale is 0.5 while the area is larger than
    ``max_area``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Defaults to 800.
        max_area (int): Upper bound areas for adaptive scales.
            Defaults to 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    if isinstance(areas, torch.Tensor):
        scales = 0.5 + (areas - min_area) / (max_area - min_area)
        return scales.clamp(0.5, 1.0)
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


@VISUALIZERS.register_module()
class PytorchPoseLocalVisualizer(PytorchBackendVisualizer):
    """MMPose Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to ``None``
        vis_backends (list, optional): Visual backend config list. Defaults to
            ``None``
        save_dir (str, optional): Save file dir for all storage backends.
            If it is ``None``, the backend storage will not save any data.
            Defaults to ``None``
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to ``'green'``
        kpt_color (str, tuple(tuple(int)), optional): Color of keypoints.
            The tuple of color should be in BGR order. Defaults to ``'red'``
        link_color (str, tuple(tuple(int)), optional): Color of skeleton.
            The tuple of color should be in BGR order. Defaults to ``None``
        line_width (int, float): The width of lines. Defaults to 1
        radius (int, float): The radius of keypoints. Defaults to 4
        show_keypoint_weight (bool): Whether to adjust the transparency
            of keypoints according to their score. Defaults to ``False``
        alpha (int, float): The transparency of bboxes. Defaults to ``1.0``

    Examples:
        >>> import numpy as np
        >>> from mmengine.structures import InstanceData
        >>> from mmpose.structures import PoseDataSample
        >>> from mmpose.visualization import PytorchPoseLocalVisualizer

        >>> pose_local_visualizer = PytorchPoseLocalVisualizer(radius=1)
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_instances = InstanceData()
        >>> gt_instances.keypoints = np.array([[[1, 1], [2, 2], [4, 4],
        ...                                          [8, 8]]])
        >>> gt_pose_data_sample = PoseDataSample()
        >>> gt_pose_data_sample.gt_instances = gt_instances
        >>> dataset_meta = {'skeleton_links': [[0, 1], [1, 2], [2, 3]]}
        >>> pose_local_visualizer.set_dataset_meta(dataset_meta)
        >>> pose_local_visualizer.add_datasample('image', image,
        ...                         gt_pose_data_sample)
        >>> pose_local_visualizer.add_datasample(
        ...                       'image', image, gt_pose_data_sample,
        ...                        out_file='out_file.jpg')
        >>> pose_local_visualizer.add_datasample(
        ...                        'image', image, gt_pose_data_sample,
        ...                         show=True)
        >>> pred_instances = InstanceData()
        >>> pred_instances.keypoints = np.array([[[1, 1], [2, 2], [4, 4],
        ...                                       [8, 8]]])
        >>> pred_instances.score = np.array([0.8, 1, 0.9, 1])
        >>> pred_pose_data_sample = PoseDataSample()
        >>> pred_pose_data_sample.pred_instances = pred_instances
        >>> pose_local_visualizer.add_datasample('image', image,
        ...                         gt_pose_data_sample,
        ...                         pred_pose_data_sample)
    """

    def __init__(
        self,
        name: str = "visualizer",
        image: Optional[np.ndarray] = None,
        vis_backends: Optional[Dict] = None,
        save_dir: Optional[str] = None,
        bbox_color: Optional[Union[str, Tuple[int]]] = "green",
        kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = "red",
        link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
        text_color: Optional[Union[str, Tuple[int]]] = (255, 255, 255),
        skeleton: Optional[Union[List, Tuple]] = None,
        line_width: Union[int, float] = 1,
        radius: Union[int, float] = 3,
        show_keypoint_weight: bool = False,
        backend: str = "pytorch",
        alpha: float = 1.0,
    ):

        warnings.filterwarnings(
            "ignore", message=".*please provide the `save_dir` argument.*", category=UserWarning
        )

        super().__init__(
            name=name, image=image, vis_backends=vis_backends, save_dir=save_dir, backend=backend
        )

        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.line_width = line_width
        self.text_color = text_color
        self.skeleton = skeleton
        self.radius = radius
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight
        # Set default value. When calling
        # `PytorchPoseLocalVisualizer().set_dataset_meta(xxx)`,
        # it will override the default value.
        self.dataset_meta = {}

    def set_dataset_meta(self, dataset_meta: Dict, skeleton_style: str = "mmpose"):
        """Assign dataset_meta to the visualizer. The default visualization
        settings will be overridden.

        Args:
            dataset_meta (dict): meta information of dataset.
        """
        if skeleton_style == "openpose":
            dataset_name = dataset_meta["dataset_name"]
            if dataset_name == "coco":
                dataset_meta = parse_pose_metainfo(
                    dict(from_file="configs/_base_/datasets/coco_openpose.py")
                )
            elif dataset_name == "coco_wholebody":
                dataset_meta = parse_pose_metainfo(
                    dict(from_file="configs/_base_/datasets/" "coco_wholebody_openpose.py")
                )
            else:
                raise NotImplementedError(
                    f"openpose style has not been " f"supported for {dataset_name} dataset"
                )

        if isinstance(dataset_meta, dict):
            self.dataset_meta = dataset_meta.copy()
            self.bbox_color = dataset_meta.get("bbox_color", self.bbox_color)
            self.kpt_color = dataset_meta.get("keypoint_colors", self.kpt_color)
            self.link_color = dataset_meta.get("skeleton_link_colors", self.link_color)
            self.skeleton = dataset_meta.get("skeleton_links", self.skeleton)

            if isinstance(self.bbox_color[0], np.ndarray):
                self.bbox_color = [tuple(c.tolist()) for c in self.bbox_color]
            if isinstance(self.kpt_color[0], np.ndarray):
                self.kpt_color = [tuple(c.tolist()) for c in self.kpt_color]
            if isinstance(self.link_color[0], np.ndarray):
                self.link_color = [tuple(c.tolist()) for c in self.link_color]

        # sometimes self.dataset_meta is manually set, which might be None.
        # it should be converted to a dict at these times
        if self.dataset_meta is None:
            self.dataset_meta = {}

    # ----------------------------------------------------------------------------------
    # Torch-only helper utilities
    # ----------------------------------------------------------------------------------
    def _ensure_torch_image(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Ensure image is a torch uint8 tensor on its current / default device.

        Accepts (H,W,C) or (C,H,W). Returns (C,H,W) on same device (GPU if already) without copying
        if possible.
        """
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                # Accept (C,H,W) or (H,W,C)
                if image.shape[0] in (1, 3, 4):
                    # Assume channel-first already
                    return image
                else:
                    # Assume channel-last
                    return image.permute(2, 0, 1)
            elif image.ndim == 4:
                # Assume batch of images; take first
                return image[0]
            else:
                raise ValueError("Unsupported tensor image shape")
        # numpy -> torch
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("Expected numpy image (H,W,3|4)")
        tensor = torch.from_numpy(image)
        tensor = tensor.permute(2, 0, 1).contiguous()
        return tensor

    def _color_to_rgb_tuple(self, color) -> Tuple[int, int, int]:
        """Normalize color definitions to RGB int tuple.

        - Accepts str (parsed via mmcv), tuple/list of length 3.
        - Source meta may provide BGR ordering; we detect via attribute comment
          expectation (dataset provides BGR). If "looks like" BGR and not grayscale,
          we reverse to RGB for internal torch drawing (which assumes RGB).
        """
        if isinstance(color, str):
            bgr = mmcv.color_val(color)  # returns BGR
            return (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        if isinstance(color, (tuple, list)) and len(color) == 3:
            # Heuristic: assume provided in BGR as per original OpenCV usage
            return (int(color[2]), int(color[1]), int(color[0]))
        raise ValueError(f"Unsupported color format: {color}")

    def _draw_label(
        self,
        image: torch.Tensor,
        text: str,
        x: int,
        y: int,
        scale: float,
        color: Tuple[int, int, int],
    ):
        # Calibrate font size: original used ~13 * scale in OpenCV; our torch_draw_text internally *10
        desired = max(8, int(13 * scale))
        font_size = max(1, desired // 10)  # inverse of *10 in implementation
        image = torch_draw_text(
            image=image,
            x=int(x),
            y=int(y),
            text=text,
            font_size=font_size,
            color=color,
            position_is_text_bottom=True,
        )
        return image

    # ----------------------------------------------------------------------------------
    # Torch drawing implementations
    # ----------------------------------------------------------------------------------
    def _draw_instances_bbox(
        self, image: Union[np.ndarray, torch.Tensor], instances: InstanceData
    ) -> torch.Tensor:
        """Draw bounding boxes and corresponding labels of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        torch_image = self._ensure_torch_image(image)
        device = torch_image.device

        if "bboxes" not in instances:
            return torch_image

        bboxes = instances.bboxes
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).to(device)
        else:
            bboxes = bboxes.to(device)

        # Draw each bbox
        for box in bboxes:
            tlx, tly, brx, bry = [
                int(v.item()) if isinstance(v, torch.Tensor) else int(v) for v in box
            ]
            color_rgb = self._color_to_rgb_tuple(
                self.bbox_color
                if not isinstance(self.bbox_color, (list, tuple))
                else tuple(self.bbox_color)
            )
            torch_image = torch_draw_box(
                image=torch_image,
                tlbr=(tlx, tly, brx, bry),
                color=color_rgb,
                thickness=int(self.line_width),
                alpha=int(self.alpha * 255),
                filled=False,
            )

        if "labels" in instances and self.text_color is not None:
            classes = self.dataset_meta.get("classes", None)
            labels = instances.labels
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).to(device)
            positions = bboxes[:, :2]
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)
            txt_color = self._color_to_rgb_tuple(
                self.text_color
                if not isinstance(self.text_color, (list, tuple))
                else tuple(self.text_color)
            )
            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_int = int(label.item() if isinstance(label, torch.Tensor) else int(label))
                label_text = classes[label_int] if classes is not None else f"class {label_int}"
                x, y = int(pos[0].item()), int(pos[1].item())
                scale_val = float(
                    scales[i].item() if isinstance(scales, torch.Tensor) else scales[i]
                )
                torch_image = self._draw_label(torch_image, label_text, x, y, scale_val, txt_color)

        return torch_image

    def _draw_instances_kpts(
        self,
        image: Union[np.ndarray, torch.Tensor],
        instances: InstanceData,
        kpt_thr: float = 0.3,
        show_kpt_idx: bool = False,
        skeleton_style: str = "mmpose",
    ):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            skeleton_style (str): Skeleton style selection. Defaults to
                ``'mmpose'``

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        if skeleton_style == "openpose":
            # (Optional) Not yet converted to torch-only; fallback (could be implemented similarly)
            return self._draw_instances_kpts_openpose(image, instances, kpt_thr)

        torch_image = self._ensure_torch_image(image)
        H, W = torch_image.shape[1], torch_image.shape[2]

        if "keypoints" not in instances:
            return torch_image

        keypoints = instances.get("transformed_keypoints", instances.keypoints)
        if isinstance(keypoints, np.ndarray):
            keypoints = torch.from_numpy(keypoints)
        if "keypoints_visible" in instances:
            keypoints_visible = instances.keypoints_visible
            if isinstance(keypoints_visible, np.ndarray):
                keypoints_visible = torch.from_numpy(keypoints_visible)
        else:
            keypoints_visible = torch.ones(keypoints.shape[:-1])

        # Build color lists
        for kpts, visible in zip(keypoints, keypoints_visible):
            # Colors per keypoint
            if self.kpt_color is None or isinstance(self.kpt_color, str):
                kpt_color_list = [self.kpt_color] * len(kpts)
            elif len(self.kpt_color) == len(kpts):
                kpt_color_list = self.kpt_color
            else:
                raise ValueError(
                    f"the length of kpt_color ({len(self.kpt_color)}) does not match keypoints ({len(kpts)})"
                )

            # Draw skeleton links
            if self.skeleton is not None and self.link_color is not None:
                if self.link_color is None or isinstance(self.link_color, str):
                    link_color_list = [self.link_color] * len(self.skeleton)
                elif len(self.link_color) == len(self.skeleton):
                    link_color_list = self.link_color
                else:
                    raise ValueError(
                        f"the length of link_color ({len(self.link_color)}) does not match skeleton ({len(self.skeleton)})"
                    )
                for sk_id, sk in enumerate(self.skeleton):
                    pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                    pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                    if (
                        pos1[0] <= 0
                        or pos1[0] >= W
                        or pos1[1] <= 0
                        or pos1[1] >= H
                        or pos2[0] <= 0
                        or pos2[0] >= W
                        or pos2[1] <= 0
                        or pos2[1] >= H
                        or visible[sk[0]] < kpt_thr
                        or visible[sk[1]] < kpt_thr
                        or link_color_list[sk_id] is None
                    ):
                        continue  # skip the link that should not be drawn
                    color = link_color_list[sk_id]
                    if not isinstance(color, str):
                        color = self._color_to_rgb_tuple(color)
                    else:
                        color = self._color_to_rgb_tuple(color)
                    # draw_line expects scalar ints & color tuple
                    torch_image = torch_draw_line(
                        image=torch_image,
                        x1=pos1[0],
                        y1=pos1[1],
                        x2=pos2[0],
                        y2=pos2[1],
                        color=color,
                        thickness=int(self.line_width),
                    )

            # Draw keypoints
            for kid, kpt in enumerate(kpts):
                if visible[kid] < kpt_thr or kpt_color_list[kid] is None:
                    continue
                color = kpt_color_list[kid]
                if not isinstance(color, str):
                    color = self._color_to_rgb_tuple(color)
                else:
                    color = self._color_to_rgb_tuple(color)
                radius = int(self.radius)
                torch_image = torch_draw_circle(
                    image=torch_image,
                    center_x=float(kpt[0]),
                    center_y=float(kpt[1]),
                    radius=float(radius),
                    color=color,
                    thickness=10,
                    fill=True,
                )
                if show_kpt_idx:
                    torch_image = torch_draw_text(
                        image=torch_image,
                        x=int(kpt[0]) + radius,
                        y=int(kpt[1]) - radius,
                        text=str(kid),
                        font_size=max(1, int(self.radius)),
                        color=color,
                        position_is_text_bottom=True,
                    )
        return torch_image

    def _draw_instances_kpts_openpose(
        self, image: Union[np.ndarray, torch.Tensor], instances: InstanceData, kpt_thr: float = 0.3
    ) -> torch.Tensor:
        """Draw keypoints with OpenPose-style skeleton rendering using torch operations."""

        torch_image = self._ensure_torch_image(image)
        device = torch_image.device
        height, width = torch_image.shape[1], torch_image.shape[2]

        if "keypoints" not in instances:
            return torch_image

        keypoints = instances.get("transformed_keypoints", instances.keypoints)
        if isinstance(keypoints, np.ndarray):
            keypoints = torch.from_numpy(keypoints)
        keypoints = keypoints.to(device=device, dtype=torch.float32)
        if "keypoints_visible" in instances:
            keypoints_visible = instances.keypoints_visible
            if isinstance(keypoints_visible, np.ndarray):
                keypoints_visible = torch.from_numpy(keypoints_visible)
            keypoints_visible = keypoints_visible.to(device=device, dtype=torch.float32)
        else:
            keypoints_visible = torch.ones(keypoints.shape[:-1], device=device, dtype=torch.float32)

        keypoints_info = torch.cat((keypoints, keypoints_visible[..., None]), dim=-1)
        neck = keypoints_info[:, [5, 6]].mean(dim=1)
        neck[:, 2:3] = (
            (keypoints_info[:, 5, 2:3] > kpt_thr) & (keypoints_info[:, 6, 2:3] > kpt_thr)
        ).float()
        keypoints_info = torch.cat(
            [keypoints_info[:, :17], neck.unsqueeze(1), keypoints_info[:, 17:]], dim=1
        )

        mmpose_idx = torch.tensor(
            [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3], device=device, dtype=torch.long
        )
        openpose_idx = torch.tensor(
            [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17], device=device, dtype=torch.long
        )
        keypoints_info = keypoints_info.clone()
        keypoints_info[:, openpose_idx] = keypoints_info[:, mmpose_idx]

        keypoints = keypoints_info[..., :2]
        keypoints_visible = keypoints_info[..., 2]

        for kpts, visible in zip(keypoints, keypoints_visible):
            if self.kpt_color is None or isinstance(self.kpt_color, str):
                kpt_color_list = [self.kpt_color] * len(kpts)
            elif len(self.kpt_color) == len(kpts):
                kpt_color_list = self.kpt_color
            else:
                raise ValueError(
                    f"the length of kpt_color ({len(self.kpt_color)}) does not matches that of keypoints ({len(kpts)})"
                )

            if self.skeleton is not None and self.link_color is not None:
                if self.link_color is None or isinstance(self.link_color, str):
                    link_color_list = [self.link_color] * len(self.skeleton)
                elif len(self.link_color) == len(self.skeleton):
                    link_color_list = self.link_color
                else:
                    raise ValueError(
                        f"the length of link_color ({len(self.link_color)}) does not matches that of skeleton ({len(self.skeleton)})"
                    )

                for sk_id, sk in enumerate(self.skeleton):
                    pos1 = kpts[sk[0]]
                    pos2 = kpts[sk[1]]
                    if (
                        pos1[0] <= 0
                        or pos1[0] >= width
                        or pos1[1] <= 0
                        or pos1[1] >= height
                        or pos2[0] <= 0
                        or pos2[0] >= width
                        or pos2[1] <= 0
                        or pos2[1] >= height
                        or visible[sk[0]] < kpt_thr
                        or visible[sk[1]] < kpt_thr
                        or link_color_list[sk_id] is None
                    ):
                        continue

                    color = link_color_list[sk_id]
                    color_tuple = self._color_to_rgb_tuple(color)
                    transparency = float(self.alpha)
                    if self.show_keypoint_weight:
                        transparency *= float(
                            0.5 * (visible[sk[0]] + visible[sk[1]]).clamp(0, 1).item()
                        )
                    if sk_id <= 16:
                        transparency = min(transparency, 0.6)
                    thickness = max(1, int(self.line_width if sk_id <= 16 else 2))

                    x_pair = torch.tensor([pos1[0], pos2[0]], device=device)
                    y_pair = torch.tensor([pos1[1], pos2[1]], device=device)

                    def _draw_line(
                        img: torch.Tensor,
                        x_pair=x_pair,
                        y_pair=y_pair,
                        color_tuple=color_tuple,
                        thickness=thickness,
                    ) -> torch.Tensor:
                        return torch_draw_line(
                            image=img,
                            x1=int(x_pair[0].item()),
                            y1=int(y_pair[0].item()),
                            x2=int(x_pair[1].item()),
                            y2=int(y_pair[1].item()),
                            color=color_tuple,
                            thickness=thickness,
                        )

                    torch_image = self._apply_draw(
                        torch_image, _draw_line, max(0.0, min(1.0, transparency))
                    )

            for kid, kpt in enumerate(kpts):
                if visible[kid] < kpt_thr or kpt_color_list[kid] is None:
                    continue
                color = kpt_color_list[kid]
                if not isinstance(color, str):
                    if isinstance(color, torch.Tensor):
                        if float(color.sum().item()) == 0:
                            continue
                    elif isinstance(color, np.ndarray):
                        if float(color.sum()) == 0:
                            continue
                    else:
                        if sum(color) == 0:
                            continue
                color_tuple = self._color_to_rgb_tuple(color)
                transparency = float(self.alpha)
                if self.show_keypoint_weight:
                    transparency *= float(torch.clamp(visible[kid], 0, 1).item())
                radius = self.radius // 2 if kid > 17 else self.radius

                def _draw_circle(
                    img: torch.Tensor, kpt=kpt, color_tuple=color_tuple, radius=radius
                ) -> torch.Tensor:
                    return torch_draw_circle(
                        image=img,
                        center_x=float(kpt[0].item()),
                        center_y=float(kpt[1].item()),
                        radius=float(radius),
                        color=color_tuple,
                        thickness=1,
                        fill=True,
                    )

                torch_image = self._apply_draw(
                    torch_image, _draw_circle, max(0.0, min(1.0, transparency))
                )

        return torch_image

    def _draw_instance_heatmap(
        self,
        fields: PixelData,
        overlaid_image: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """Draw heatmaps of GT or prediction.

        Args:
            fields (:obj:`PixelData`): Data structure for
                pixel-level annotations or predictions.
            overlaid_image (np.ndarray): The image to draw.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        if "heatmaps" not in fields:
            return None
        heatmaps = fields.heatmaps
        if isinstance(heatmaps, np.ndarray):
            heatmaps = torch.from_numpy(heatmaps)
        heatmaps = heatmaps.to(dtype=torch.float32)
        if heatmaps.dim() == 3:
            # (K,H,W) -> aggregate max
            heatmaps, _ = heatmaps.max(dim=0)
        # Normalize 0..1
        hmin = heatmaps.min()
        hmax = heatmaps.max()
        if (hmax - hmin) > 1e-6:
            norm = (heatmaps - hmin) / (hmax - hmin)
        else:
            norm = torch.zeros_like(heatmaps)
        # Simple colormap: jet-like (approx) using ramps
        r = torch.clamp(1.5 - torch.abs(4 * norm - 3), 0, 1)
        g = torch.clamp(1.5 - torch.abs(4 * norm - 2), 0, 1)
        b = torch.clamp(1.5 - torch.abs(4 * norm - 1), 0, 1)
        heat_rgb = torch.stack([r, g, b], dim=0)  # (3,H,W)
        heat_rgb = (heat_rgb * 255).to(torch.uint8)
        if overlaid_image is not None:
            base = self._ensure_torch_image(overlaid_image).to(torch.uint8)
            alpha = 0.5
            heat_rgb = (
                alpha * heat_rgb.to(torch.float32) + (1 - alpha) * base.to(torch.float32)
            ).to(torch.uint8)
        return heat_rgb

    def _draw_instance_xy_heatmap(
        self,
        fields: PixelData,
        overlaid_image: Optional[np.ndarray] = None,
        n: int = 20,
    ):
        """Draw heatmaps of GT or prediction.

        Args:
            fields (:obj:`PixelData`): Data structure for
            pixel-level annotations or predictions.
            overlaid_image (np.ndarray): The image to draw.
            n (int): Number of keypoint, up to 20.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        if "heatmaps" not in fields:
            return None
        heatmaps = fields.heatmaps
        _, h, w = heatmaps.shape
        if isinstance(heatmaps, np.ndarray):
            heatmaps = torch.from_numpy(heatmaps)
        if SimCCVisualizer is None:
            return None
        out_image = SimCCVisualizer().draw_instance_xy_heatmap(heatmaps, overlaid_image, n)
        out_image = cv2.resize(out_image[:, :, ::-1], (w, h))
        return out_image

    @master_only
    def add_datasample(
        self,
        name: str,
        image: Union[np.ndarray, torch.Tensor],
        data_sample: PoseDataSample,
        draw_gt: bool = True,
        draw_pred: bool = True,
        draw_heatmap: bool = False,
        draw_bbox: bool = False,
        show_kpt_idx: bool = False,
        skeleton_style: str = "mmpose",
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        kpt_thr: float = 0.3,
        step: int = 0,
        clone_image: bool = True,
    ) -> torch.Tensor:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier
            image (np.ndarray): The image to draw
            data_sample (:obj:`PoseDataSample`, optional): The data sample
                to visualize
            draw_gt (bool): Whether to draw GT PoseDataSample. Default to
                ``True``
            draw_pred (bool): Whether to draw Prediction PoseDataSample.
                Defaults to ``True``
            draw_bbox (bool): Whether to draw bounding boxes. Default to
                ``False``
            draw_heatmap (bool): Whether to draw heatmaps. Defaults to
                ``False``
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            skeleton_style (str): Skeleton style selection. Defaults to
                ``'mmpose'``
            show (bool): Whether to display the drawn image. Default to
                ``False``
            wait_time (float): The interval of show (s). Defaults to 0
            out_file (str): Path to output file. Defaults to ``None``
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            step (int): Global step value to record. Defaults to 0
        """

        torch_image = self._ensure_torch_image(image)

        gt_img_data: Optional[torch.Tensor] = None
        pred_img_data: Optional[torch.Tensor] = None
        gt_img_heatmap: Optional[torch.Tensor] = None
        pred_img_heatmap: Optional[torch.Tensor] = None

        if draw_gt:
            gt_img_data = torch_image.clone() if clone_image else torch_image
            if "gt_instances" in data_sample:
                gt_img_data = self._draw_instances_kpts(
                    gt_img_data, data_sample.gt_instances, kpt_thr, show_kpt_idx, skeleton_style
                )
                if draw_bbox:
                    gt_img_data = self._draw_instances_bbox(gt_img_data, data_sample.gt_instances)
            if "gt_fields" in data_sample and draw_heatmap:
                gt_img_heatmap = self._draw_instance_heatmap(data_sample.gt_fields, torch_image)
                if gt_img_heatmap is not None:
                    gt_img_data = torch.cat((gt_img_data, gt_img_heatmap), dim=1)

        if draw_pred:
            pred_img_data = torch_image.clone() if clone_image else torch_image
            if "pred_instances" in data_sample:
                pred_img_data = self._draw_instances_kpts(
                    pred_img_data, data_sample.pred_instances, kpt_thr, show_kpt_idx, skeleton_style
                )
                if draw_bbox:
                    pred_img_data = self._draw_instances_bbox(
                        pred_img_data, data_sample.pred_instances
                    )
            if "pred_fields" in data_sample and draw_heatmap:
                pred_img_heatmap = self._draw_instance_heatmap(data_sample.pred_fields, torch_image)
                if pred_img_heatmap is not None:
                    pred_img_data = torch.cat((pred_img_data, pred_img_heatmap), dim=1)

        if gt_img_data is not None and pred_img_data is not None:
            # Align heatmap presence
            if gt_img_heatmap is None and pred_img_heatmap is not None:
                gt_img_data = torch.cat((gt_img_data, torch_image), dim=1)
            elif gt_img_heatmap is not None and pred_img_heatmap is None:
                pred_img_data = torch.cat((pred_img_data, torch_image), dim=1)
            drawn_img = torch.cat((gt_img_data, pred_img_data), dim=2)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data if pred_img_data is not None else torch_image

        self.set_image(drawn_img)

        cpu_image: Optional[np.ndarray] = None
        if out_file is not None or bool(self._vis_backends):
            cpu_image = self._tensor_to_numpy(drawn_img)

        if out_file is not None and cpu_image is not None:
            mmcv.imwrite(cpu_image[..., ::-1], out_file)
        elif cpu_image is not None:
            self.add_image(name, cpu_image, step)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        return drawn_img
