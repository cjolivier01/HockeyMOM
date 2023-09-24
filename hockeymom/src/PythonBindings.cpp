#include "hm/ImagePostProcess.h"
#include "hm/Inheritance.h"
#include "hm/NestedClasses.h"
#include "hm/sublibA/ConsoleColors.h"
#include "hm/sublibA/add.h"

#include "hockeymom/csrc/mblend/mblend.h"
#include "hockeymom/csrc/stitcher/HmStitcher.h"

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

  py::class_<hm::MatrixRGB>(m, "MatrixRGB", py::buffer_protocol())
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

  auto pyFoo = py::class_<hm::Foo>(m, "Foo");
  pyFoo.def(py::init<>()).def("f", &hm::Foo::f);

  py::class_<hm::Foo::Child>(pyFoo, "FooChild")
      .def(py::init<>())
      .def("g", &hm::Foo::Child::g);

  auto sublibA = m.def_submodule("sublibA");
  sublibA.def("add", hm::sublibA::add);

  py::enum_<hm::sublibA::ConsoleForegroundColor>(
      sublibA, "ConsoleForegroundColor")
      .value("Green", hm::sublibA::ConsoleForegroundColor::Green)
      .value("Yellow", hm::sublibA::ConsoleForegroundColor::Yellow)
      .value("Blue", hm::sublibA::ConsoleForegroundColor::Blue)
      .value("Magenta", hm::sublibA::ConsoleForegroundColor::Magenta)
      .export_values();

  py::enum_<hm::sublibA::ConsoleBackgroundColor>(
      sublibA, "ConsoleBackgroundColor")
      .value("Green", hm::sublibA::Green)
      .value("Yellow", hm::sublibA::Yellow)
      .value("Blue", hm::sublibA::Blue)
      .value("Magenta", hm::sublibA::Magenta)
      .export_values();

  sublibA.def(
      "accept_defaulted_enum",
      [](const hm::sublibA::ConsoleForegroundColor& color) {},
      py::arg("color") = hm::sublibA::ConsoleForegroundColor::Blue);

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
        std::unique_ptr<hm::MatrixRGB> result = hm::enblend::enblend(m1, m2);
        return result->to_py_array();
      });

  py::class_<hm::HmNona, std::shared_ptr<hm::HmNona>>(m, "HmNona")
      .def(py::init<std::string>())
      .def("count", &hm::HmNona::count)
      .def("load_project", &hm::HmNona::load_project);

  m.def(
      "_nona_process_images",
      [](std::shared_ptr<hm::HmNona> nona,
         py::array_t<uint8_t>& image1,
         py::array_t<uint8_t>& image2) {
        py::gil_scoped_release release_gil();
        auto m1 = std::make_shared<hm::MatrixRGB>(image1, 0, 0);
        auto m2 = std::make_shared<hm::MatrixRGB>(image2, 0, 0);
        std::
            pair<std::unique_ptr<hm::MatrixRGB>, std::unique_ptr<hm::MatrixRGB>>
                result_pair =
                    nona->process_images(std::move(m1), std::move(m2));
        // TODO: return something meaningful
        return true;
      });

  auto pyOuter = py::class_<hm::Outer>(m, "Outer");
  auto pyInner = py::class_<hm::Outer::Inner>(pyOuter, "Inner");

  py::enum_<hm::Outer::Inner::NestedEnum>(pyInner, "NestedEnum")
      .value("ONE", hm::Outer::Inner::NestedEnum::ONE)
      .value("TWO", hm::Outer::Inner::NestedEnum::TWO);

  py::class_<hm::Base> pyBase(m, "Base");

  pyBase.def_readwrite("name", &hm::Base::name);

  py::class_<hm::Base::Inner>(pyBase, "Inner");

  // py::class_<hm::Derived, hm::Base> (m, "Derived")
  //   .def_readwrite("count", &hm::Derived::count);

  pyInner.def_readwrite("value", &hm::Outer::Inner::value);

  pyOuter.def_readwrite("inner", &hm::Outer::inner)
      .def_property_readonly_static("linalg", [](py::object) {
        return py::module::import("numpy.linalg");
      });

  py::register_exception<hm::CppException>(m, "CppException");

  m.attr("foovar") = hm::Foo();

  py::list foolist;
  foolist.append(hm::Foo());
  foolist.append(hm::Foo());

  m.attr("foolist") = foolist;
  m.attr("none") = py::none();
  {
    py::list li;
    li.append(py::none{});
    li.append(2);
    li.append(py::dict{});
    m.attr("list_with_none") = li;
  }

  auto numeric = m.def_submodule("numeric");
  numeric.def("get_ndarray_int", [] { return py::array_t<int>{}; });
  numeric.def("get_ndarray_float64", [] { return py::array_t<double>{}; });
  numeric.def("accept_ndarray_int", [](py::array_t<int>) {});
  numeric.def("accept_ndarray_float64", [](py::array_t<double>) {});

  auto eigen = m.def_submodule("eigen");
  eigen.def("get_matrix_int", [] { return Eigen::Matrix3i{}; });
  eigen.def("get_vector_float64", [] { return Eigen::Vector3d{}; });
  eigen.def("accept_matrix_int", [](Eigen::Matrix3i) {});
  eigen.def("accept_vector_float64", [](Eigen::Vector3d) {});

  auto opaque_types = m.def_submodule("opaque_types");

  py::bind_vector<std::vector<std::pair<std::string, double>>>(
      opaque_types, "VectorPairStringDouble");
  py::bind_map<std::map<std::string, std::complex<double>>>(
      opaque_types, "MapStringComplex");

  opaque_types.def("get_complex_map", [] {
    return std::map<std::string, std::complex<double>>{};
  });
  opaque_types.def("get_vector_of_pairs", [] {
    return std::vector<std::pair<std::string, double>>{};
  });

  auto copy_types = m.def_submodule("copy_types");
  copy_types.def(
      "get_complex_map", [] { return std::map<int, std::complex<double>>{}; });
  copy_types.def("get_vector_of_pairs", [] {
    return std::vector<std::pair<int, double>>{};
  });

  // This submodule will have C++ signatures in python docstrings to emulate
  // poorly written pybind11-bindings
  auto invalid_signatures = m.def_submodule("invalid_signatures");
  invalid_signatures.def(
      "get_unbound_type", [] { return forgotten::Unbound{}; });
  invalid_signatures.def(
      "accept_unbound_type",
      [](std::pair<forgotten::Unbound, int>) { return 0; });
  invalid_signatures.def(
      "accept_unbound_enum", [](forgotten::Enum) { return 0; });

  py::class_<forgotten::Unbound>(invalid_signatures, "Unbound");
  py::class_<forgotten::Enum>(invalid_signatures, "Enum");
  invalid_signatures.def(
      "accept_unbound_type_defaulted",
      [](forgotten::Unbound) { return 0; },
      py::arg("x") = forgotten::Unbound{});
  invalid_signatures.def(
      "accept_unbound_enum_defaulted",
      [](forgotten::Enum) { return 0; },
      py::arg("x") = forgotten::Enum::ONE);

  auto issues = m.def_submodule("issues");
  issues.def(
      "issue_51", [](int*, int*) {}, R"docstring(

    Use-case:
        issue_51(os.get_handle_inheritable, os.set_handle_inheritable))docstring");
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
