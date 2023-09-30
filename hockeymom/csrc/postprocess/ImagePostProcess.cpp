#include "hockeymom/csrc/postprocess/ImagePostProcess.h"

#include <iostream>

void hm::Foo::f() { std::cout << "invoked hm::Foo::f()" << std::endl; }

void hm::Foo::Child::g() {
  std::cout << "invoked hm::Foo::Child::g()" << std::endl;
}
