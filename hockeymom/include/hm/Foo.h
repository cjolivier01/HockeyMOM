#pragma once
#include <stdexcept>

namespace hm{


class CppException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Foo {
    void f();

    struct Child {
        void g();
    };
};

}