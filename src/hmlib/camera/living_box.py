from typing import Optional, Tuple

import torch

from hmlib.bbox.box_functions import (
    center,
    center_batch,
    clamp_box,
    get_enclosing_box,
    height,
    make_box_at_center,
    move_box_to_center,
    remove_largest_bbox,
    scale_box,
    tlwh_centers,
    tlwh_to_tlbr_single,
    width,
)
from hmlib.tracking_utils import visualization as vis
from hockeymom.core import AllLivingBoxConfig, BBox, LivingBox, WHDims


def to_bbox(tensor: torch.Tensor, is_cpp: bool) -> BBox:
    if not is_cpp:
        return tensor
    if isinstance(tensor, BBox):
        return tensor
    bbox = BBox()
    bbox.left = tensor[0].item()
    bbox.top = tensor[1].item()
    bbox.right = tensor[2].item()
    bbox.bottom = tensor[3].item()
    return bbox


def from_bbox(bbox: BBox, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    if isinstance(bbox, torch.Tensor):
        return bbox
    return torch.tensor(
        [bbox.left, bbox.top, bbox.right, bbox.bottom], dtype=torch.float, device=device
    )


class PyLivingBox(LivingBox):

    def __init__(self, *args, color: Tuple[int, int, int], thickness: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._color = color
        self._thickness = thickness

    def _draw_resizing_state(
        self,
        img: torch.Tensor,
        draw_thresholds: bool = False,
        following_box: Optional[LivingBox] = None,
    ):
        rconfig = self.resizing_config()
        rstate = self.resizing_state()
        lconfig = self.living_config()
        lstate = self.living_state()

        if rconfig.sticky_sizing:
            assert following_box is not None  # why?
            my_bbox = self.bounding_box()
            my_bbox_t: torch.Tensor = from_bbox(my_bbox)
            center_tensor = center(my_bbox_t)
            my_width = my_bbox.width()
            my_height = my_bbox.height()
            following_bbox = from_bbox(following_box.bounding_box())
            # dashed box representing the following box inscribed at our center

            if rstate.size_is_frozen:
                corner_box = scale_box(box=my_bbox_t, scale_width=0.98, scale_height=0.98)
                img = vis.draw_corner_boxes(
                    image=img, bbox=corner_box, color=(255, 255, 255), thickness=1
                )

            scaled_following_box = scale_box(
                following_bbox,
                scale_width=lconfig.scale_dest_width,
                scale_height=lconfig.scale_dest_height,
            )

            inscribed = move_box_to_center(scaled_following_box.clone(), center(my_bbox_t))
            img = vis.draw_dashed_rectangle(
                img,
                box=inscribed,
                color=(255, 255, 255) if not lstate.was_size_constrained else (0, 0, 255),
                thickness=2,
            )

            # Sizing thresholds
            if draw_thresholds:
                gs = self.get_grow_shrink_wh(my_bbox)
                grow_box = make_box_at_center(
                    center_point=center_tensor,
                    w=my_width + gs.grow_width,
                    h=my_height + gs.grow_height,
                )
                shrink_box = make_box_at_center(
                    center_point=center_tensor,
                    w=my_width - gs.shrink_width,
                    h=my_height - gs.shrink_height,
                )

                img = vis.draw_centered_lines(img, bbox=grow_box, thickness=4, color=(0, 255, 0))
                img = vis.draw_centered_lines(img, bbox=shrink_box, thickness=4, color=(0, 0, 255))
        return img

    def _draw_translation_state(
        self,
        img: torch.Tensor,
        draw_thresholds: bool = False,
        following_box: Optional[LivingBox] = None,
    ):
        tconfig = self.translation_config()
        tstate = self.translation_state()
        draw_box = from_bbox(self.bounding_box())
        img = vis.plot_rectangle(
            img,
            draw_box,
            color=self._color if not tstate.translation_is_frozen else (128, 128, 128),
            thickness=self._thickness,
            label=self.name(),
            text_scale=2,
        )
        return img

    def draw(
        self,
        img: torch.Tensor,
        draw_thresholds: bool = False,
        following_box: Optional[LivingBox] = None,
    ) -> torch.Tensor:
        img = self._draw_resizing_state(
            img, draw_thresholds=draw_thresholds, following_box=following_box
        )
        img = self._draw_translation_state(
            img, draw_thresholds=draw_thresholds, following_box=following_box
        )
        return img

    def get_size_scale(self) -> Tuple[float, float]:
        size_scale: WHDims = super().get_size_scale()
        return size_scale.width, size_scale.height
