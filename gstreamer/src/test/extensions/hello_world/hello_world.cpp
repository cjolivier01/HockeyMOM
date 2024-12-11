
#include "HelloWorld.hpp"

#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(
    0x05f887afddee3d15,
    0xb467168585f324f4,
    "hello_world",
    "colivier:A Dummy Example",
    "",
    "1.0.2",
    "NVIDIA");
GXF_EXT_FACTORY_ADD(
    0xf849597b07313143,
    0xb442234de9857f53,
    sample::test::HelloWorld,
    nvidia::gxf::Codelet,
    "colivier: Dummy example source codelet.");
GXF_EXT_FACTORY_END()
