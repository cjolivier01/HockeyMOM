#pragma once

#include <cassert>
#include <cmath>
// #include <memory>
// #include <optional>
// #include <string>
// #include <variant>

namespace hm {
namespace play_tracker {

using FloatValue = float;
using IntValue = int64_t;

/**
 * __          __ _    _ _____  _
 * \ \        / /| |  | |  __ \(_)
 *  \ \  /\  / / | |__| | |  | |_ _ __ ___   ___
 *   \ \/  \/ /  |  __  | |  | | | '_ ` _ \ / __|
 *    \  /\  /   | |  | | |__| | | | | | | |\__ \
 *     \/  \/    |_|  |_|_____/|_|_| |_| |_||___/
 *
 */
struct WHDims {
  const FloatValue width;
  const FloatValue height;
};

/**
 *  _____       _       _   _____  _  __  __
 * |  __ \     (_)     | | |  __ \(_)/ _|/ _|
 * | |__) |___  _ _ __ | |_| |  | |_| |_| |_
 * |  ___// _ \| | '_ \| __| |  | | |  _|  _|
 * | |   | (_) | | | | | |_| |__| | | | | |
 * |_|    \___/|_|_| |_|\__|_____/|_|_| |_|
 *
 */
struct PointDiff {
  FloatValue dx;
  FloatValue dy;
};

/**
 *   _____  _            _____  _  __  __
 *  / ____|(_)          |  __ \(_)/ _|/ _|
 * | (___   _  ____ ___ | |  | |_| |_| |_
 *  \___ \ | ||_  // _ \| |  | | |  _|  _|
 *  ____) || | / /|  __/| |__| | | | | |
 * |_____/ |_|/___|\___||_____/|_|_| |_|
 *
 */
struct SizeDiff {
  FloatValue dw;
  FloatValue dh;
};

/**
 *  _____       _       _
 * |  __ \     (_)     | |
 * | |__) |___  _ _ __ | |_
 * |  ___// _ \| | '_ \| __|
 * | |   | (_) | | | | | |_
 * |_|    \___/|_|_| |_|\__|
 *
 */
struct Point {
  FloatValue x;
  FloatValue y;

  PointDiff operator-(const Point& pt) const {
    return PointDiff{.dx = x - pt.x, .dy = y - pt.y};
  }
};

/**
 *  ____  ____
 * |  _ \|  _ \
 * | |_) | |_) | ___ __  __
 * |  _ <|  _ < / _ \\ \/ /
 * | |_) | |_) | (_) |>  <
 * |____/|____/ \___//_/\_\
 *
 * @brief Bounding Box
 */
struct BBox {
  BBox() = default;
  BBox(FloatValue l, FloatValue t, FloatValue r, FloatValue b)
      : left(l), top(t), right(r), bottom(b) {}
  BBox(const Point& center, const WHDims& dims) {
    auto half_w = dims.width / 2;
    auto half_h = dims.height / 2;
    left = center.x - half_w;
    right = center.x + half_w;
    top = center.y - half_h;
    bottom = center.y + half_h;
  }
  constexpr FloatValue width() const {
    assert(right >= left);
    return right - left;
  }
  constexpr FloatValue height() const {
    assert(bottom >= top);
    return bottom - top;
  }
  constexpr FloatValue aspect_ratio() const {
    return width() / height();
  }
  BBox clone() const {
    return *this;
  }
  Point center() const {
    return Point{.x = (right - left) / 2, .y = (bottom - top) / 2};
  }
  BBox make_scaled(FloatValue scale_width, FloatValue scale_height) const {
    return BBox(
        center(),
        WHDims{
            .width = width() * scale_width, .height = height() * scale_height});
  }
  BBox inflate(
      FloatValue dleft,
      FloatValue dtop,
      FloatValue dright,
      FloatValue dbottom) const {
    return BBox(left + dleft, top + dtop, right + dright, bottom + dbottom);
  }
  void validate() const {
#ifndef NDEBUG
    assert(right >= left);
    assert(bottom >= top);
#endif
  }
  // The four bbox values
  FloatValue left{0.0};
  FloatValue top{0.0};
  FloatValue right{0.0};
  FloatValue bottom{0.0};
};

} // namespace play_tracker
} // namespace hm
