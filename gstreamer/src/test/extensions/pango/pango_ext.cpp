#include "gxf/std/extension_factory_helper.hpp"
#include "timeoverlay.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0x5de23bd3a3ce38bb, 0xa38b71087b47f43b, "pango",
                          "Pango-based text rendering and overlay", "AUTHOR", "0.0.1",
                          "LICENSE");

GXF_EXT_FACTORY_ADD_GstTimeOverlay();

GXF_EXT_FACTORY_END()
