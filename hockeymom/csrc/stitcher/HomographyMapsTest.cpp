#include "hockeymom/csrc/stitcher/HomographyMaps.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

int main() {
  const std::vector<std::array<double, 2>> right_points = {
      {5.0, 5.0},
      {75.0, 5.0},
      {75.0, 55.0},
      {5.0, 55.0},
      {30.0, 20.0},
      {50.0, 40.0},
      {20.0, 50.0},
  };
  std::vector<std::array<double, 2>> left_points;
  left_points.reserve(right_points.size());
  for (const auto& point : right_points) {
    left_points.push_back({point[0] + 20.0, point[1] + 10.0});
  }
  left_points.back() = {90.0, 5.0};

  const auto result = hm::stitcher::create_homography_maps(
      left_points, right_points, 100, 80, 100, 80, 1.0, 0.999, 10000, 0);
  assert(result.canvas_width == 120);
  assert(result.canvas_height == 90);
  assert(std::abs(result.right_to_left_homography[2] - 20.0) < 1e-4);
  assert(std::abs(result.right_to_left_homography[5] - 10.0) < 1e-4);
  assert(result.image_maps[0].x_position == 0);
  assert(result.image_maps[0].y_position == 0);
  assert(result.image_maps[1].x_position == 20);
  assert(result.image_maps[1].y_position == 10);
  assert(result.image_maps[0].x_map.front() == 0);
  assert(result.image_maps[0].y_map.front() == 0);
  assert(result.image_maps[1].x_map.front() == 0);
  assert(result.image_maps[1].y_map.front() == 0);
  assert(result.inlier_mask.size() == right_points.size());
  assert(result.inlier_mask.back() == 0);

  const auto scaled = hm::stitcher::create_homography_maps(
      left_points, right_points, 100, 80, 100, 80, 1.0, 0.999, 10000, 60);
  assert(scaled.canvas_width == 60);
  assert(scaled.canvas_height == 45);
  assert(std::abs(scaled.output_scale - 0.5) < 1e-6);

  return 0;
}
