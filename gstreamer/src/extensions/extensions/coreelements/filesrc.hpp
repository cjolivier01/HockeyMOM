#pragma once

#include <string>

#include "nvdsbase/nvds_io.hpp"

namespace nvidia {
namespace deepstream {

class GstFileSrc : public INvDsElement {
 public:
  gxf_result_t initialize() override {
    GXF_LOG_INFO("initialize: %s %s\n", GST_ELEMENT_NAME, name());
    return GXF_SUCCESS;
  }

  gxf_result_t create_element() override {
    std::string ename = entity().name();
    ename += "/";
    ename += name();
    GXF_LOG_INFO("create_element: %s %s\n", GST_ELEMENT_NAME, name());
    element_ = gst_element_factory_make(GST_ELEMENT_NAME, ename.c_str());

    if (!element_) {
      GXF_LOG_ERROR("Could not create GStreamer element '%s'",
                    GST_ELEMENT_NAME);
      return GXF_FAILURE;
    }

    auto p_src_pad = src_pad.try_get();
    if (p_src_pad) {
      p_src_pad.value()->set_element(this);
      p_src_pad.value()->set_template_name("src");
    }
    return GXF_SUCCESS;
  }

  gxf_result_t bin_add(GstElement *pipeline) override {
    GXF_LOG_INFO("bin_add: %s %s\n", GST_ELEMENT_NAME, name());
    if (!gst_bin_add(GST_BIN(pipeline), element_)) {
      return GXF_FAILURE;
    }

    auto p_blocksize = blocksize.try_get();
    if (p_blocksize && p_blocksize.value() != 4096UL) {
      guint propvalue = (guint)p_blocksize.value();
      g_object_set(element_, "blocksize", propvalue, NULL);
    }

    auto p_num_buffers = num_buffers.try_get();
    if (p_num_buffers && p_num_buffers.value() != -1L) {
      gint propvalue = (gint)p_num_buffers.value();
      g_object_set(element_, "num-buffers", propvalue, NULL);
    }

    auto p_typefind = typefind.try_get();
    if (p_typefind && p_typefind.value() != false) {
      gboolean propvalue = (gboolean)p_typefind.value();
      g_object_set(element_, "typefind", propvalue, NULL);
    }

    auto p_do_timestamp = do_timestamp.try_get();
    if (p_do_timestamp && p_do_timestamp.value() != false) {
      gboolean propvalue = (gboolean)p_do_timestamp.value();
      g_object_set(element_, "do-timestamp", propvalue, NULL);
    }

    auto p_location = location.try_get();
    if (p_location && p_location.value() != std::string{""}) {
      gchararray propvalue = (gchararray)p_location.value().c_str();
      g_object_set(element_, "location", propvalue, NULL);
    }

    return GXF_SUCCESS;
  }

  GstElement *get_element_ptr() override { return element_; }

  gxf_result_t registerInterface(nvidia::gxf::Registrar *registrar) override {
    nvidia::gxf::Expected<void> result;
    result &=
        registrar->parameter(blocksize, "blocksize", "Block size",
                             "Size in bytes to read per buffer (-1 = default)",
                             4096UL, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        num_buffers, "num-buffers", "num-buffers",
        "Number of buffers to output before sending EOS (-1 = unlimited)", -1L,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        typefind, "typefind", "Typefind",
        "Run typefind before negotiating (deprecated, non-functional)", false,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(do_timestamp, "do-timestamp", "Do timestamp",
                                   "Apply current stream time to buffers",
                                   false, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        location, "location", "File Location", "Location of the file to read",
        std::string{""}, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        src_pad, "out", "out",
        "Handle to a nvidia::deepstream::NvDsStaticOutput component. Supported "
        "formats - ANY",
        gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
    return nvidia::gxf::ToResultCode(result);
  }

  nvidia::gxf::Parameter<uint64_t> blocksize;
  nvidia::gxf::Parameter<int64_t> num_buffers;
  nvidia::gxf::Parameter<bool> typefind;
  nvidia::gxf::Parameter<bool> do_timestamp;
  nvidia::gxf::Parameter<std::string> location;
  nvidia::gxf::Parameter<nvidia::gxf::Handle<NvDsStaticOutput>> src_pad;

 protected:
  GstElement *element_;
  const char *GST_ELEMENT_NAME = "filesrc";
};

#define GXF_EXT_FACTORY_ADD_GstFileSrc()                                       \
  do {                                                                         \
    GXF_EXT_FACTORY_ADD_VERBOSE(                                               \
        0xa2408d9d9d6a3597UL, 0x85f9ae038e22866cUL,                            \
        nvidia::deepstream::GstFileSrc, nvidia::deepstream::INvDsElement,      \
        "GstFileSrc placeholder display-name", "GstFileSrc placeholder brief", \
        "Read from arbitrary point in a file");                                \
  } while (0)

}  // namespace deepstream
}  // namespace nvidia
