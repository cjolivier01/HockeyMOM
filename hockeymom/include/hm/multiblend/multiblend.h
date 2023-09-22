#pragma once

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <string>
#include <vector>

namespace py = pybind11;

namespace enblend {
int enblend_main(std::string output_image,
                 std::vector<std::string> input_files);

class MatrixRGB {
  static inline constexpr std::size_t kChannels = 3;

public:
  inline MatrixRGB(py::array_t<uint8_t> &input_image) {
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
  }
  inline MatrixRGB(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
    m_data = new std::uint8_t[rows * cols * kChannels];
    m_own_data = true;
  }
  inline ~MatrixRGB() { delete m_data; }

  inline std::uint8_t *data() { return m_data; }
  inline size_t rows() const { return m_rows; }
  inline size_t cols() const { return m_cols; }
  inline size_t channels() const { return kChannels; }

  inline py::array_t<std::uint8_t> to_py_array() {
    if (m_array) {
      return std::move(m_array);
    }
    py::array_t<std::uint8_t> result(
        {m_rows, m_cols},
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
  size_t m_rows, m_cols;
  std::uint8_t *m_data;
};

MatrixRGB enblend(MatrixRGB &image1, MatrixRGB &image2);

} // namespace enblend
