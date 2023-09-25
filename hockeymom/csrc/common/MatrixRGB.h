#pragma once

#include <vigra/impex.hxx>
#include <vigra/multi_array.hxx>

// TODO: remove pybind dependency int his file/class

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cassert>
#include <cstddef>
#include <vector>

namespace py = pybind11;

namespace hm {
/**
 * @brief Class to carry python array for RGB data into the blend function
 *        (in-place)
 */
struct MatrixRGB {
 public:

  static inline constexpr std::size_t kChannels = 3;

  MatrixRGB() {}
  MatrixRGB(const MatrixRGB&) = delete;
  MatrixRGB(MatrixRGB&& other) = default;

  MatrixRGB(
      py::array_t<uint8_t>& input_image,
      std::size_t xpos,
      std::size_t ypos) {
    // Check if the input is a 3D array with dtype uint8 (RGB image)
    if (input_image.ndim() != kChannels || input_image.shape(2) != kChannels ||
        !input_image.dtype().is(py::dtype::of<uint8_t>())) {
      throw std::runtime_error("Input must be a 3D uint8 RGB image array");
    }

    // Access the data and information from the NumPy array
    py::buffer_info buf_info = input_image.request();

    // Get the pointer to the data
    m_data = static_cast<uint8_t*>(buf_info.ptr);

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
  MatrixRGB(size_t rows, size_t cols, std::uint8_t* consume_data)
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
  void set_xy_pos(std::size_t xpos, std::size_t ypos) {
    m_xpos = xpos;
    m_ypos = ypos;
  }
  constexpr std::uint8_t* data() {
    return m_data;
  }
  constexpr size_t rows() const {
    return m_rows;
  }
  constexpr size_t cols() const {
    return m_cols;
  }
  constexpr size_t channels() const {
    return kChannels;
  }

  py::array_t<std::uint8_t> to_py_array() {
    if (!m_data) {
      return std::move(m_array);
    }
    py::array_t<std::uint8_t> result(
        {rows(), cols(), channels()},
        {channels() * sizeof(std::uint8_t) *
             cols(), /* Strides (in bytes) for each index */
         channels() * sizeof(std::uint8_t),
         sizeof(std::uint8_t)},
        m_data);
    m_data = nullptr;

    return result;
  }

  std::unique_ptr<vigra::BRGBImage> to_vigra_brgb_image() {
    const vigra::RGBValue<unsigned char>* rgb_data =
        reinterpret_cast<vigra::RGBValue<unsigned char>*>(data());
    return std::make_unique<vigra::BRGBImage>(cols(), rows(), rgb_data);
  }

 private:
  bool m_own_data{false};
  size_t m_rows{0}, m_cols{0};
  std::uint8_t* m_data;
  std::size_t m_xpos{0};
  std::size_t m_ypos{0};
  py::array_t<uint8_t> m_array;
};

struct MatrixRGBEncoder : public vigra::Encoder {
  virtual ~MatrixRGBEncoder() = default;
  virtual void init(const std::string&) {}

  // initialize with file access mode. For codecs that do not support this
  // feature, the standard init is called.
  virtual void init(const std::string& fileName, const std::string&) {
    init(fileName);
  }

  virtual void close() {}
  virtual void abort() {
    assert(false);
  }

  virtual std::string getFileType() const {
    return "TIFF";
  }
  virtual unsigned int getOffset() const {
    return MatrixRGB::kChannels;
    //return 1;
  }

  virtual void setWidth(unsigned int width) {
    width_ = width;
  }
  virtual void setHeight(unsigned int height) {
    height_ = height;
  }
  virtual void setNumBands(unsigned int num_bands) {
    num_bands_ = num_bands;
  }
  virtual void setCompressionType(const std::string& type, int = -1) {
  }
  virtual void setPixelType(const std::string& pixel_type) {
    assert(pixel_type == "UINT8");
  }
  virtual void finalizeSettings() {
    std::size_t total_image_size = sizeof(std::uint8_t) * width_ * height_ * MatrixRGB::kChannels;
    data_ = std::make_unique<std::uint8_t[]>(total_image_size);
    bzero(data_.get(), total_image_size);
    dummy_alpha_ = std::make_unique<std::uint8_t[]>(sizeof(std::uint8_t) * width_ * MatrixRGB::kChannels);
    scanlines_.resize(MatrixRGB::kChannels);
    for (std::size_t i = 0; i < MatrixRGB::kChannels; ++i) {
      //scanlines_.at(i) = data_.get() + (sizeof(std::uint8_t) * width_ * height_ * i);
      scanlines_.at(i) = data_.get() + (sizeof(std::uint8_t) * i);
    }
  }

  virtual void setPosition(const vigra::Diff2D& pos) {
    position_ = pos;
  }
  virtual void setCanvasSize(const vigra::Size2D& size) {
    canvas_size_ = size;
  }
  virtual void setXResolution(float xres) {
    x_res_ = xres;
  }
  virtual void setYResolution(float yres) {
    y_res_ = yres;
  }

  typedef vigra::ArrayVector<unsigned char> ICCProfile;

  virtual void setICCProfile(const ICCProfile& data) {
    icc_profile_ = data;
  }

  virtual void* currentScanlineOfBand(unsigned int band) {
    if (band == MatrixRGB::kChannels) {
      return dummy_alpha_.get();
    }
    return scanlines_[band];
  }
  virtual void nextScanline() {
    for (auto& ptr : scanlines_) {
      ptr += sizeof(std::uint8_t) * width_ * MatrixRGB::kChannels;
    }
  }

  std::unique_ptr<MatrixRGB> consume() {
    auto matrix = std::make_unique<MatrixRGB>(height_, width_, data_.release());
    matrix->set_xy_pos(position_.x, position_.y);
    return matrix;
  }

 private:
  std::size_t width_{0}, height_{0};
  float x_res_{0}, y_res_{0};
  int num_bands_{0};
  ICCProfile icc_profile_;
  vigra::Diff2D position_;
  vigra::Size2D canvas_size_;
  std::unique_ptr<std::uint8_t[]> data_;
  std::unique_ptr<std::uint8_t[]> dummy_alpha_;
  std::unique_ptr<MatrixRGB> matrix_rgb_{nullptr};
  std::vector<std::uint8_t*> scanlines_;
};

} // namespace hm
