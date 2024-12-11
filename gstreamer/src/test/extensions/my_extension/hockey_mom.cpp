
#include "HockeyMom.hpp"

#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(
    0xeab0452c34f84520,
    0x9cc8b2ed347240a2,
    "hockey_mom",
    "Hockey Mom extension (testing)",
    "Christopher Olivier",
    "0.0.1",
    "NVIDIA");
GXF_EXT_FACTORY_ADD(
    0xeab0452c34f84520,
    0x9cc8b2ed347240a2,
    sample::test::HockeyMom,
    nvidia::gxf::Codelet,
    "colivier: Dummy example source codelet.");
GXF_EXT_FACTORY_END()
