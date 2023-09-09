#pragma once
#include <stdexcept>

//#include <ATen/ATen.h>

namespace hm {

class CppException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Foo {
  void f();

  struct Child {
    void g();
  };
};

class ImagePostProcessor {
  public:
    ImagePostProcessor();
  //at::Tensor
};



} // namespace hm
