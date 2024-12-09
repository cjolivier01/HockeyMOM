#include "gxf/std/extension_factory_helper.hpp"
#include "filesink.hpp"
#include "filesrc.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0x31dcbaca115a376b, 0x9986e2883874ed93, "coreelements",
                          "GStreamer core elements", "AUTHOR", "0.0.1",
                          "LICENSE");

GXF_EXT_FACTORY_ADD_GstFileSink();

GXF_EXT_FACTORY_ADD_GstFileSrc();

GXF_EXT_FACTORY_END()
