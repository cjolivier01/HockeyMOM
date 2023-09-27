#include "hm/ImagePostProcess.h"

#include "hockeymom/csrc/dataloader/StitchingDataLoader.h"
#include "hockeymom/csrc/mblend/mblend.h"
#include "hockeymom/csrc/stitcher/HmNona.h"

#include <iostream>

// namespace with types that lacks pybind counterpart
namespace forgotten {

struct Unbound {};

enum Enum { ONE = 1, TWO = 2 };

} // namespace forgotten

PYBIND11_MAKE_OPAQUE(std::map<std::string, std::complex<double>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::pair<std::string, double>>);

namespace py = pybind11;

class Matrix2DFloat {
 public:
  Matrix2DFloat(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
    m_data = new float[rows * cols];
  }
  float* data() {
    return m_data;
  }
  size_t rows() const {
    return m_rows;
  }
  size_t cols() const {
    return m_cols;
  }

 private:
  size_t m_rows, m_cols;
  float* m_data;
};

PYBIND11_MODULE(_hockeymom, m) {
  std::cout << "Initializing hockymom module" << std::endl;

  py::class_<Matrix2DFloat>(m, "Matrix2DFloat", py::buffer_protocol())
      .def_buffer([](Matrix2DFloat& m) -> py::buffer_info {
        return py::buffer_info(
            m.data(), /* Pointer to buffer */
            sizeof(float), /* Size of one scalar */
            py::format_descriptor<float>::format(), /* Python struct-style
                                                       format descriptor */
            2, /* Number of dimensions */
            {m.rows(), m.cols()}, /* Buffer dimensions */
            {sizeof(float) * m.cols(), /* Strides (in bytes) for each index */
             sizeof(float)});
      });

  py::class_<hm::MatrixRGB, std::shared_ptr<hm::MatrixRGB>>(
      m, "MatrixRGB", py::buffer_protocol())
      .def_buffer([](hm::MatrixRGB& m) -> py::buffer_info {
        return py::buffer_info(
            m.data(), /* Pointer to buffer */
            sizeof(std::uint8_t), /* Size of one scalar */
            py::format_descriptor<std::uint8_t>::
                format(), /* Python struct-style format descriptor */
            3, /* Number of dimensions */
            {m.rows(), m.cols(), m.channels()}, /* Buffer dimensions */
            {m.channels() * sizeof(std::uint8_t) * m.cols(),
             m.channels() * sizeof(std::uint8_t),
             sizeof(std::uint8_t)});
      });

  py::class_<hm::StitchingDataLoader, std::shared_ptr<hm::StitchingDataLoader>>(
      m, "StitchingDataLoader")
      .def(py::init<std::size_t, std::string, std::size_t, std::size_t>());

  m.def(
      "_add_to_stitching_data_loader",
      [](std::shared_ptr<hm::StitchingDataLoader> data_loader,
         std::size_t frame_id,
         py::array_t<uint8_t>& image1,
         py::array_t<uint8_t>& image2) {
        py::gil_scoped_release release_gil();
        // We expect a three-channel RGB image here
        assert(image1.ndim() == 3);
        assert(image2.ndim() == 3);
        auto m1 = std::make_shared<hm::MatrixRGB>(image1, 0, 0);
        auto m2 = std::make_shared<hm::MatrixRGB>(image2, 0, 0);
        data_loader->add_frame(frame_id, {std::move(m1), std::move(m2)});
        return frame_id;
      });

  m.def(
      "_get_stitched_frame_from_data_loader",
      [](std::shared_ptr<hm::StitchingDataLoader> data_loader,
         std::size_t frame_id) -> py::array_t<std::uint8_t> {
        py::gil_scoped_release release_gil();
        auto stitched_image = data_loader->get_stitched_frame(frame_id);
        return stitched_image->to_py_array();
      });

  m.def("_hello_world", []() {
    py::gil_scoped_release release_gil();
    std::cout << "Hello, world!" << std::endl;
  });

  m.def(
      "_enblend",
      [](std::string output_image,
         std::vector<std::string> input_files) -> int {
        py::gil_scoped_release release_gil();
        return hm::enblend::enblend_main(
            std::move(output_image), std::move(input_files));
      });

  m.def(
      "_emblend_images",
      [](py::array_t<uint8_t>& image1,
         std::vector<std::size_t> xy_pos_1,
         py::array_t<uint8_t>& image2,
         std::vector<std::size_t> xy_pos_2) {
        py::gil_scoped_release release_gil();
        hm::MatrixRGB m1(image1, xy_pos_1.at(0), xy_pos_1.at(1));
        hm::MatrixRGB m2(image2, xy_pos_2.at(0), xy_pos_2.at(1));
        // Just blend (no remap)
        std::unique_ptr<hm::MatrixRGB> result = hm::enblend::enblend(m1, m2);
        return result->to_py_array();
      });

  py::class_<hm::HmNona, std::shared_ptr<hm::HmNona>>(m, "HmNona")
      .def(py::init<std::string>())
      .def("load_project", &hm::HmNona::load_project);

  m.def(
      "_nona_process_images",
      [](std::shared_ptr<hm::HmNona> nona,
         py::array_t<uint8_t>& image1,
         py::array_t<uint8_t>& image2) -> std::vector<py::array_t<uint8_t>> {
        py::gil_scoped_release release_gil();

        // We expect a three-channel RGB image here
        assert(image1.ndim() == 3);
        assert(image2.ndim() == 3);
        auto m1 = std::make_shared<hm::MatrixRGB>(image1, 0, 0);
        auto m2 = std::make_shared<hm::MatrixRGB>(image2, 0, 0);
        // Just remap (no blend)
        std::vector<std::unique_ptr<hm::MatrixRGB>> result_matrices =
            nona->remap_images(std::move(m1), std::move(m2));
        std::vector<py::array_t<uint8_t>> results;
        results.reserve(result_matrices.size());
        for (auto& m : result_matrices) {
          if (m) {
            results.emplace_back(m->to_py_array());
          }
        }
        return results;
      });

  m.def(
      "_stitch_images",
      [](std::shared_ptr<hm::HmNona> nona,
         py::array_t<uint8_t>& image1,
         py::array_t<uint8_t>& image2) -> py::array_t<uint8_t> {
        py::gil_scoped_release release_gil();
        auto m1 = std::make_shared<hm::MatrixRGB>(image1, 0, 0);
        auto m2 = std::make_shared<hm::MatrixRGB>(image2, 0, 0);
        // First remap...
        std::vector<std::unique_ptr<hm::MatrixRGB>> result_matrices =
            nona->remap_images(std::move(m1), std::move(m2));
        // Then blend...
        std::unique_ptr<hm::MatrixRGB> result = hm::enblend::enblend(
            *result_matrices.at(0), *result_matrices.at(1));
        return result->to_py_array();
      });
}

static std::string get_python_string(PyObject* obj) {
  std::string str;
  PyObject* temp_bytes = PyUnicode_AsEncodedString(obj, "UTF-8", "strict");
  if (temp_bytes != NULL) {
    const char* s = PyBytes_AS_STRING(temp_bytes);
    str = s;
    Py_DECREF(temp_bytes);
  }
  return str;
}

extern "C" int __py_bt() {
  int count = 0;
  PyThreadState* tstate = PyThreadState_GET();
  if (NULL != tstate && NULL != tstate->frame) {
    PyFrameObject* frame = tstate->frame;

    printf("Python stack trace:\n");
    while (NULL != frame) {
      // int line = frame->f_lineno;
      /*
       frame->f_lineno will not always return the correct line number
       you need to call PyCode_Addr2Line().
      */
      int line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
      PyObject* temp_bytes = PyUnicode_AsEncodedString(
          frame->f_code->co_filename, "UTF-8", "strict");
      if (temp_bytes != NULL) {
        auto filename = get_python_string(frame->f_code->co_filename);
        auto funcname = get_python_string(frame->f_code->co_name);
        printf("    %s(%d): %s\n", filename.c_str(), line, funcname.c_str());
        Py_DECREF(temp_bytes);
      }
      frame = frame->f_back;
      ++count;
    }
  }
  return count;
}
