import torch
from hmlib.tracking_utils import visualization as vis


class BoundaryLines:
    def __init__(self, upper_border_lines, lower_border_lines, original_clip_box=None):
        self._original_clip_box = original_clip_box
        if self._original_clip_box is None:
            self._original_clip_box = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        elif not isinstance(self._original_clip_box, torch.Tensor):
            self._original_clip_box = torch.tensor(
                self._original_clip_box, dtype=torch.float32
            )
        clip_upper_left = self._original_clip_box[0:2]
        if upper_border_lines:
            self._upper_borders = torch.tensor(upper_border_lines, dtype=torch.float32)
            self._upper_borders[:, 0:2] -= clip_upper_left
            self._upper_borders[:, 2:4] -= clip_upper_left
            self._upper_line_vectors = self.tlbr_to_line_vectors(self._upper_borders)
        else:
            self._upper_borders = None
            self._upper_line_vectors = None
        if lower_border_lines:
            self._lower_borders = torch.tensor(lower_border_lines, dtype=torch.float32)
            self._lower_borders[:, 0:2] -= clip_upper_left
            self._lower_borders[:, 2:4] -= clip_upper_left
            self._lower_line_vectors = self.tlbr_to_line_vectors(self._lower_borders)
        else:
            self._lower_borders = None
            self._lower_line_vectors = None

    def draw(self, img):
        if self._upper_borders is not None:
            for i in range(len(self._upper_borders)):
                vis.plot_line(
                    img,
                    self._upper_borders[i][0:2],
                    self._upper_borders[i][2:4],
                    color=(255, 0, 0),
                    thickness=1,
                )
        if self._lower_borders is not None:
            for i in range(len(self._lower_borders)):
                vis.plot_line(
                    img,
                    self._lower_borders[i][0:2],
                    self._lower_borders[i][2:4],
                    color=(0, 0, 255),
                    thickness=1,
                )

    def tlbr_to_line_vectors(self, tlbr_batch):
        # Assuming tlbr_batch shape is (N, 4) with each box as [top, left, bottom, right]

        top_vectors = torch.stack(
            (tlbr_batch[:, 1], tlbr_batch[:, 0]), dim=1
        ) - torch.stack((tlbr_batch[:, 3], tlbr_batch[:, 0]), dim=1)
        left_vectors = torch.stack(
            (tlbr_batch[:, 1], tlbr_batch[:, 0]), dim=1
        ) - torch.stack((tlbr_batch[:, 1], tlbr_batch[:, 2]), dim=1)
        bottom_vectors = torch.stack(
            (tlbr_batch[:, 3], tlbr_batch[:, 2]), dim=1
        ) - torch.stack((tlbr_batch[:, 1], tlbr_batch[:, 2]), dim=1)
        right_vectors = torch.stack(
            (tlbr_batch[:, 3], tlbr_batch[:, 2]), dim=1
        ) - torch.stack((tlbr_batch[:, 3], tlbr_batch[:, 0]), dim=1)

        # Concatenating vectors for all sides
        line_vectors = torch.stack(
            (top_vectors, left_vectors, bottom_vectors, right_vectors), dim=1
        )

        return line_vectors

    def is_point_too_high(self, point):
        return False

    def is_point_too_low(self, point):
        too_low = self.is_point_below_line(point)
        return False

    def is_point_outside(self, point):
        return self.is_point_above_line(point) or self.is_point_below_line(point)

    def _is_point_above_line(self, point, line_start, line_end):
        # Create vectors
        line_vec = line_end - line_start
        point_vec = point - line_start

        # Compute the cross product
        cross_product_z = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]

        # Check if the point is above the line
        return cross_product_z < 0

    def is_point_above_line(self, point):
        if self._upper_borders is None:
            return False
        point = point.cpu()
        x = point[0]
        # y = point[1]
        for i, high_border in enumerate(self._upper_borders):
            if x >= high_border[0] and x <= high_border[2]:
                if self._is_point_above_line(point, high_border[0:2], high_border[2:4]):
                    return True
        return False

    def point_batch_check_point_above_segents(self, points):
        lines_start = self._upper_borders[:, 0:2]
        point_vecs = points[0] - lines_start
        cross_z = (
            self._upper_line_vectors[:, 0] * point_vecs[:, 1]
            - self._upper_line_vectors[:, 1] * point_vecs[:, 0]
        )
        return cross_z < 0

    def is_point_below_line(self, point):
        if self._lower_borders is None:
            return False
        point = point.cpu()
        x = point[0]
        # y = point[1]
        for i, low_border in enumerate(self._lower_borders):
            if x >= low_border[0] and x <= low_border[2]:
                if self._is_point_below_line(point, low_border[0:2], low_border[2:4]):
                    return True
        return False

    def _is_point_below_line(self, point, line_start, line_end):
        # Create vectors
        line_vec = line_end - line_start
        point_vec = point - line_start

        # Compute the cross product
        cross_product_z = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]

        # Check if the point is below the line
        return cross_product_z > 0
