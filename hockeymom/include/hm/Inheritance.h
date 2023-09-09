#pragma once
#include <string>

namespace hm{

struct Base {
  struct Inner{};
  std::string name;
};

struct Derived : Base {
  int count;
};

}