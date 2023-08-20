#include "hm/Foo.h"
#include "hm/sublibA/add.h"
#include "hm/sublibA/ConsoleColors.h"
#include "hm/NestedClasses.h"
#include "hm/Inheritance.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <iostream>

// namespace with types that lacks pybind counterpart
namespace forgotten {

struct Unbound {};

enum Enum{
    ONE=1,
    TWO=2
};

}


PYBIND11_MAKE_OPAQUE(std::map<std::string, std::complex<double>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::pair<std::string, double>>);

namespace py = pybind11;

PYBIND11_MODULE(_hockeymom, m)
{
  std::cout << "Initializing hockymom module" << std::endl;
  auto pyFoo = py::class_<hm::Foo>(m,"Foo");
  pyFoo
    .def(py::init<>())
    .def("f",&hm::Foo::f);

  py::class_<hm::Foo::Child> (pyFoo, "FooChild")
    .def(py::init<>())
    .def("g",&hm::Foo::Child::g);

  auto sublibA = m.def_submodule("sublibA");
  sublibA.def("add", hm::sublibA::add);

  py::enum_<hm::sublibA::ConsoleForegroundColor> (sublibA, "ConsoleForegroundColor")
    .value("Green", hm::sublibA::ConsoleForegroundColor::Green)
    .value("Yellow", hm::sublibA::ConsoleForegroundColor::Yellow)
    .value("Blue", hm::sublibA::ConsoleForegroundColor::Blue)
    .value("Magenta", hm::sublibA::ConsoleForegroundColor::Magenta)
    .export_values();

  py::enum_<hm::sublibA::ConsoleBackgroundColor> (sublibA, "ConsoleBackgroundColor")
    .value("Green", hm::sublibA::Green)
    .value("Yellow", hm::sublibA::Yellow)
    .value("Blue", hm::sublibA::Blue)
    .value("Magenta", hm::sublibA::Magenta)
    .export_values();

  sublibA.def("accept_defaulted_enum",
      [](const hm::sublibA::ConsoleForegroundColor& color){},
      py::arg("color") = hm::sublibA::ConsoleForegroundColor::Blue
  );

  m.def("_hello_worls", []() {
    std::cout << "Hello, world!" << std::endl;
  });

  auto pyOuter = py::class_<hm::Outer> (m, "Outer");
  auto pyInner = py::class_<hm::Outer::Inner> (pyOuter, "Inner");

  py::enum_<hm::Outer::Inner::NestedEnum> (pyInner, "NestedEnum")
    .value("ONE", hm::Outer::Inner::NestedEnum::ONE)
    .value("TWO", hm::Outer::Inner::NestedEnum::TWO)
    ;

  py::class_<hm::Base> pyBase(m, "Base");

  pyBase
    .def_readwrite("name", &hm::Base::name);

  py::class_<hm::Base::Inner>(pyBase, "Inner");

  // py::class_<hm::Derived, hm::Base> (m, "Derived")
  //   .def_readwrite("count", &hm::Derived::count);

  pyInner
    .def_readwrite("value", &hm::Outer::Inner::value );

  pyOuter
    .def_readwrite("inner", &hm::Outer::inner)
    .def_property_readonly_static("linalg", [](py::object){ return py::module::import("numpy.linalg"); });

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
  numeric.def("get_ndarray_int", []{ return py::array_t<int>{}; });
  numeric.def("get_ndarray_float64", []{ return py::array_t<double>{}; });
  numeric.def("accept_ndarray_int", [](py::array_t<int>){});
  numeric.def("accept_ndarray_float64", [](py::array_t<double>){});


  auto eigen = m.def_submodule("eigen");
  eigen.def("get_matrix_int", []{ return Eigen::Matrix3i{}; });
  eigen.def("get_vector_float64", []{ return Eigen::Vector3d{}; });
  eigen.def("accept_matrix_int", [](Eigen::Matrix3i){});
  eigen.def("accept_vector_float64", [](Eigen::Vector3d){});

  auto opaque_types = m.def_submodule("opaque_types");

  py::bind_vector<std::vector<std::pair<std::string, double>>>(opaque_types, "VectorPairStringDouble");
  py::bind_map<std::map<std::string, std::complex<double>>>(opaque_types, "MapStringComplex");

  opaque_types.def("get_complex_map", []{return std::map<std::string, std::complex<double>>{}; });
  opaque_types.def("get_vector_of_pairs", []{return std::vector<std::pair<std::string, double>>{}; });

  auto copy_types = m.def_submodule("copy_types");
  copy_types.def("get_complex_map", []{return std::map<int, std::complex<double>>{}; });
  copy_types.def("get_vector_of_pairs", []{return std::vector<std::pair<int, double>>{}; });

  // This submodule will have C++ signatures in python docstrings to emulate poorly written pybind11-bindings
  auto invalid_signatures = m.def_submodule("invalid_signatures");
  invalid_signatures.def("get_unbound_type", []{return forgotten::Unbound{}; });
  invalid_signatures.def("accept_unbound_type", [](std::pair<forgotten::Unbound, int>){ return 0;});
  invalid_signatures.def("accept_unbound_enum", [](forgotten::Enum){ return 0;});

  py::class_<forgotten::Unbound>(invalid_signatures, "Unbound");
  py::class_<forgotten::Enum>(invalid_signatures, "Enum");
  invalid_signatures.def("accept_unbound_type_defaulted", [](forgotten::Unbound){ return 0;}, py::arg("x")=forgotten::Unbound{});
  invalid_signatures.def("accept_unbound_enum_defaulted", [](forgotten::Enum){ return 0;}, py::arg("x")=forgotten::Enum::ONE);

  auto issues = m.def_submodule("issues");
  issues.def("issue_51", [](int*, int*){}, R"docstring(

    Use-case:
        issue_51(os.get_handle_inheritable, os.set_handle_inheritable))docstring");

}

static std::string get_python_string(PyObject *obj) {
  std::string str;
  PyObject *temp_bytes = PyUnicode_AsEncodedString(obj, "UTF-8", "strict");
  if (temp_bytes != NULL) {
    const char *s = PyBytes_AS_STRING(temp_bytes);
    str = s;
    Py_DECREF(temp_bytes);
  }
  return str;
}

extern "C" int __py_bt() {
  int count = 0;
  PyThreadState *tstate = PyThreadState_GET();
  if (NULL != tstate && NULL != tstate->frame) {
    PyFrameObject *frame = tstate->frame;

    printf("Python stack trace:\n");
    while (NULL != frame) {
      // int line = frame->f_lineno;
      /*
       frame->f_lineno will not always return the correct line number
       you need to call PyCode_Addr2Line().
      */
      int line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
      PyObject *temp_bytes = PyUnicode_AsEncodedString(
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
