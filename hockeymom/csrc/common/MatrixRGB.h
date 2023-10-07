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
#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace hm {
/**
 * @brief Class to carry python array for RGB data into the blend function
 *        (in-place)
 */
// template <std::size_t CHANNELS>
class __attribute__((visibility("default"))) MatrixImage {
 public:
  static constexpr std::size_t kPixelSampleSize = sizeof(std::uint8_t);

  MatrixImage() {}
  MatrixImage(const MatrixImage&) = delete;
  MatrixImage(MatrixImage&& other) = default;

  MatrixImage(
      py::array_t<uint8_t>& input_image,
      std::size_t xpos,
      std::size_t ypos,
      bool copy_data = false) {
    // Check if the input is a 3D array with dtype uint8 (RGB image)
    if (input_image.ndim() != 3 ||
        !input_image.dtype().is(py::dtype::of<uint8_t>())) {
      throw std::runtime_error("Input must be a 3D uint8 RGB image array");
    }

    // Access the data and information from the NumPy array
    auto py_buffer_info = input_image.request();

    // Get the dimensions
    m_rows = py_buffer_info.shape[0];
    m_cols = py_buffer_info.shape[1];
    m_channels = py_buffer_info.shape[2];
    if (copy_data) {
      std::size_t image_bytes =
          sizeof(std::uint8_t) * m_rows * m_cols * m_channels;
      m_data = new std::uint8_t[image_bytes];
      memcpy(m_data, py_buffer_info.ptr, image_bytes);
      m_own_data = true;
    } else {
      m_data = static_cast<uint8_t*>(py_buffer_info.ptr);
      input_image.release();
      m_own_data = true;
    }
    m_xpos = xpos;
    m_ypos = ypos;
  }
  MatrixImage(size_t rows, size_t cols, size_t channels)
      : m_rows(rows), m_cols(cols), m_channels(channels) {
    m_data = new std::uint8_t[rows * cols * m_channels * kPixelSampleSize];
    m_own_data = true;
  }
  MatrixImage(
      size_t rows,
      size_t cols,
      size_t channels,
      std::uint8_t* consume_data)
      : m_rows(rows), m_cols(cols), m_channels(channels) {
    m_data = consume_data;
    m_own_data = true;
  }
  virtual ~MatrixImage() {
    if (m_data && m_own_data) {
      delete[] m_data;
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
  constexpr size_t channels() {
    return m_channels;
  }

  py::array_t<std::uint8_t> to_py_array() {
    auto capsule = py::capsule(m_data, [](void* data) {
      assert(data);
      delete[] reinterpret_cast<std::uint8_t*>(data);
    });
    assert(m_data && m_own_data);
    py::array_t<std::uint8_t> result(
        {rows(),
         cols(),
         channels() * kPixelSampleSize} /* total buffer size in bytes */,
        {channels() * kPixelSampleSize *
             cols() /* Strides (in bytes) for each index */,
         channels() * kPixelSampleSize,
         kPixelSampleSize},
        m_data,
        std::move(capsule));
    m_own_data = false;
    m_data = nullptr;
    return result;
  }

  std::unique_ptr<vigra::BRGBImage> to_vigra_image() {
    if (channels() == 3) {
      const vigra::RGBValue<unsigned char>* rgb_data =
          reinterpret_cast<vigra::RGBValue<unsigned char>*>(data());
      return std::make_unique<vigra::BRGBImage>(cols(), rows(), rgb_data);
    } else if (channels() == 4) {
      assert(false);
      const vigra::RGBValue<unsigned char>* rgba_data =
          reinterpret_cast<vigra::RGBValue<unsigned char>*>(data());
      return std::make_unique<vigra::BRGBImage>(cols(), rows(), rgba_data);
    } else {
      assert(false);
    }
  }

 private:
  bool m_own_data{false};
  size_t m_rows{0}, m_cols{0}, m_channels{0};
  std::uint8_t* m_data;
  std::size_t m_xpos{0};
  std::size_t m_ypos{0};
};

using MatrixRGB = MatrixImage;

template <std::size_t CHANNELS>
struct MatrixEncoder : public vigra::Encoder {
 public:
  virtual ~MatrixEncoder() = default;
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
    return "TIFF"; // not really TIFF but tell it we are so we get some more
                   // interesting info
  }
  virtual unsigned int getOffset() const {
    return CHANNELS;
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
  virtual void setCompressionType(const std::string& type, int = -1) {}
  virtual void setPixelType(const std::string& pixel_type) {
    assert(pixel_type == "UINT8");
  }
  constexpr std::size_t width() const {
    return width_;
  }
  constexpr std::size_t height() const {
    return height_;
  }
  virtual void finalizeSettings() {
    std::size_t num_channels = CHANNELS;
    std::size_t total_image_size =
        MatrixRGB::kPixelSampleSize * width() * height() * num_channels;
    assert(total_image_size);
    data_ = std::make_unique<std::uint8_t[]>(total_image_size);
    scanlines_.resize(CHANNELS);
    for (std::size_t i = 0; i < CHANNELS; ++i) {
      scanlines_.at(i) = data_.get() + (MatrixRGB::kPixelSampleSize * i);
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
    return scanlines_[band];
  }

  virtual void nextScanline() {
    for (auto& ptr : scanlines_) {
      ptr += sizeof(std::uint8_t) * width_ * CHANNELS;
    }
  }
  std::unique_ptr<MatrixRGB> consume() {
    auto matrix =
        std::make_unique<MatrixRGB>(height_, width_, CHANNELS, data_.release());
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
  std::unique_ptr<MatrixRGB> matrix_rgb_{nullptr};

 protected:
  std::vector<std::uint8_t*> scanlines_;
};

using MatrixEncoderRGB = MatrixEncoder<3>;
using MatrixEncoderRGBA = MatrixEncoder<4>;

} // namespace hm
