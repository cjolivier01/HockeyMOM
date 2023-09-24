#pragma once

// TODO: remove pybind dependency int his file/class

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cstddef>

namespace py = pybind11;

namespace hm {
/**
 * @brief Class to carry python array for RGB data into the blend function
 *        (in-place)
 */
class MatrixRGB {
  static inline constexpr std::size_t kChannels = 3;

public:
  MatrixRGB() {}
  MatrixRGB(const MatrixRGB&) = delete;
  MatrixRGB(MatrixRGB&& other) = default;

  MatrixRGB(py::array_t<uint8_t> &input_image, std::size_t xpos,
                   std::size_t ypos) {
    // Check if the input is a 3D array with dtype uint8 (RGB image)
    if (input_image.ndim() != kChannels || input_image.shape(2) != kChannels ||
        !input_image.dtype().is(py::dtype::of<uint8_t>())) {
      throw std::runtime_error("Input must be a 3D uint8 RGB image array");
    }

    // Access the data and information from the NumPy array
    py::buffer_info buf_info = input_image.request();

    // Get the pointer to the data
    m_data = static_cast<uint8_t *>(buf_info.ptr);

    // Get the dimensions
    m_rows = buf_info.shape[0];
    m_cols = buf_info.shape[1];
    m_own_data = false;
    m_array = input_image;
    m_xpos = xpos;
    m_ypos = ypos;
  }
  MatrixRGB(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
    m_data = new std::uint8_t[rows * cols * kChannels];
    m_own_data = true;
  }
  MatrixRGB(size_t rows, size_t cols, std::uint8_t *consume_data)
      : m_rows(rows), m_cols(cols) {
    m_data = consume_data;
    m_own_data = true;
  }
  ~MatrixRGB() {
    if (m_data && m_own_data) {
      delete m_data;
    }
  }
  std::vector<std::size_t> xy_pos() const {
    return {m_xpos, m_ypos};
  }
  constexpr std::uint8_t *data() { return m_data; }
  constexpr size_t rows() const { return m_rows; }
  constexpr  size_t cols() const { return m_cols; }
  constexpr  size_t channels() const { return kChannels; }

  py::array_t<std::uint8_t> to_py_array() {
    if (!m_data) {
      return std::move(m_array);
    }
    py::array_t<std::uint8_t> result(
        {rows(), cols(), channels()},
        {channels() * sizeof(std::uint8_t) *
             cols(), /* Strides (in bytes) for each index */
         channels() * sizeof(std::uint8_t), sizeof(std::uint8_t)},
        m_data);
    m_data = nullptr;

    return result;
  }

private:
  py::array_t<uint8_t> m_array;
  bool m_own_data{false};
  size_t m_rows{0}, m_cols{0};
  std::uint8_t *m_data;
  std::size_t m_xpos{0};
  std::size_t m_ypos{0};
};


}