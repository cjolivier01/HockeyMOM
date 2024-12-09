#pragma once

#include <string>

#include "nvdsbase/nvds_io.hpp"

namespace nvidia {
namespace deepstream {

class GstFileSink : public INvDsElement {
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

    auto p_sink_pad = sink_pad.try_get();
    if (p_sink_pad) {
      p_sink_pad.value()->set_element(this);
      p_sink_pad.value()->set_template_name("sink");
    }
    return GXF_SUCCESS;
  }

  gxf_result_t bin_add(GstElement *pipeline) override {
    GXF_LOG_INFO("bin_add: %s %s\n", GST_ELEMENT_NAME, name());
    if (!gst_bin_add(GST_BIN(pipeline), element_)) {
      return GXF_FAILURE;
    }

    auto p_sync = sync.try_get();
    if (p_sync && p_sync.value() != false) {
      gboolean propvalue = (gboolean)p_sync.value();
      g_object_set(element_, "sync", propvalue, NULL);
    }

    auto p_max_lateness = max_lateness.try_get();
    if (p_max_lateness && p_max_lateness.value() != -1L) {
      gint64 propvalue = (gint64)p_max_lateness.value();
      g_object_set(element_, "max-lateness", propvalue, NULL);
    }

    auto p_qos = qos.try_get();
    if (p_qos && p_qos.value() != false) {
      gboolean propvalue = (gboolean)p_qos.value();
      g_object_set(element_, "qos", propvalue, NULL);
    }

    auto p_async = async.try_get();
    if (p_async && p_async.value() != true) {
      gboolean propvalue = (gboolean)p_async.value();
      g_object_set(element_, "async", propvalue, NULL);
    }

    auto p_ts_offset = ts_offset.try_get();
    if (p_ts_offset && p_ts_offset.value() != 0L) {
      gint64 propvalue = (gint64)p_ts_offset.value();
      g_object_set(element_, "ts-offset", propvalue, NULL);
    }

    auto p_enable_last_sample = enable_last_sample.try_get();
    if (p_enable_last_sample && p_enable_last_sample.value() != true) {
      gboolean propvalue = (gboolean)p_enable_last_sample.value();
      g_object_set(element_, "enable-last-sample", propvalue, NULL);
    }

    auto p_blocksize = blocksize.try_get();
    if (p_blocksize && p_blocksize.value() != 4096UL) {
      guint propvalue = (guint)p_blocksize.value();
      g_object_set(element_, "blocksize", propvalue, NULL);
    }

    auto p_render_delay = render_delay.try_get();
    if (p_render_delay && p_render_delay.value() != 0UL) {
      guint64 propvalue = (guint64)p_render_delay.value();
      g_object_set(element_, "render-delay", propvalue, NULL);
    }

    auto p_throttle_time = throttle_time.try_get();
    if (p_throttle_time && p_throttle_time.value() != 0UL) {
      guint64 propvalue = (guint64)p_throttle_time.value();
      g_object_set(element_, "throttle-time", propvalue, NULL);
    }

    auto p_max_bitrate = max_bitrate.try_get();
    if (p_max_bitrate && p_max_bitrate.value() != 0UL) {
      guint64 propvalue = (guint64)p_max_bitrate.value();
      g_object_set(element_, "max-bitrate", propvalue, NULL);
    }

    auto p_processing_deadline = processing_deadline.try_get();
    if (p_processing_deadline && p_processing_deadline.value() != 20000000UL) {
      guint64 propvalue = (guint64)p_processing_deadline.value();
      g_object_set(element_, "processing-deadline", propvalue, NULL);
    }

    auto p_location = location.try_get();
    if (p_location && p_location.value() != std::string{""}) {
      gchararray propvalue = (gchararray)p_location.value().c_str();
      g_object_set(element_, "location", propvalue, NULL);
    }

    auto p_buffer_mode = buffer_mode.try_get();
    if (p_buffer_mode && p_buffer_mode.value() != -1L) {
      gint64 propvalue = (gint64)p_buffer_mode.value();
      g_object_set(element_, "buffer-mode", propvalue, NULL);
    }

    auto p_buffer_size = buffer_size.try_get();
    if (p_buffer_size && p_buffer_size.value() != 65536UL) {
      guint propvalue = (guint)p_buffer_size.value();
      g_object_set(element_, "buffer-size", propvalue, NULL);
    }

    auto p_append = append.try_get();
    if (p_append && p_append.value() != false) {
      gboolean propvalue = (gboolean)p_append.value();
      g_object_set(element_, "append", propvalue, NULL);
    }

    auto p_o_sync = o_sync.try_get();
    if (p_o_sync && p_o_sync.value() != false) {
      gboolean propvalue = (gboolean)p_o_sync.value();
      g_object_set(element_, "o-sync", propvalue, NULL);
    }

    auto p_max_transient_error_timeout = max_transient_error_timeout.try_get();
    if (p_max_transient_error_timeout &&
        p_max_transient_error_timeout.value() != 0L) {
      gint propvalue = (gint)p_max_transient_error_timeout.value();
      g_object_set(element_, "max-transient-error-timeout", propvalue, NULL);
    }

    return GXF_SUCCESS;
  }

  GstElement *get_element_ptr() override { return element_; }

  gxf_result_t registerInterface(nvidia::gxf::Registrar *registrar) override {
    nvidia::gxf::Expected<void> result;
    result &= registrar->parameter(sync, "sync", "Sync", "Sync on the clock",
                                   false, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &=
        registrar->parameter(max_lateness, "max-lateness", "Max Lateness",
                             "Maximum number of nanoseconds that a buffer can "
                             "be late before it is dropped (-1 unlimited)",
                             -1L, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        qos, "qos", "Qos", "Generate Quality-of-Service events upstream", false,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(async, "async", "Async",
                                   "Go asynchronously to PAUSED", true,
                                   GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(ts_offset, "ts-offset", "TS Offset",
                                   "Timestamp offset in nanoseconds", 0L,
                                   GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        enable_last_sample, "enable-last-sample", "Enable Last Buffer",
        "Enable the last-sample property", true, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &=
        registrar->parameter(blocksize, "blocksize", "Block size",
                             "Size in bytes to pull per buffer (0 = default)",
                             4096UL, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        render_delay, "render-delay", "Render Delay",
        "Additional render delay of the sink in nanoseconds", 0UL,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        throttle_time, "throttle-time", "Throttle time",
        "The time to keep between rendered buffers (0 = disabled)", 0UL,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        max_bitrate, "max-bitrate", "Max Bitrate",
        "The maximum bits per second to render (0 = disabled)", 0UL,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        processing_deadline, "processing-deadline", "Processing deadline",
        "Maximum processing time for a buffer in nanoseconds", 20000000UL,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        location, "location", "File Location", "Location of the file to write",
        std::string{""}, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &=
        registrar->parameter(buffer_mode, "buffer-mode", "Buffering mode",
                             "The buffering mode to use\nValid values:\n -1: "
                             "default\n  0: full\n  1: line\n  2: unbuffered",
                             -1L, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        buffer_size, "buffer-size", "Buffering size",
        "Size of buffer in number of bytes for line or full buffer-mode",
        65536UL, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(append, "append", "Append",
                                   "Append to an already existing file", false,
                                   GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        o_sync, "o-sync", "Synchronous IO",
        "Open the file with O_SYNC for enabling synchronous IO", false,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        max_transient_error_timeout, "max-transient-error-timeout",
        "Max Transient Error Timeout",
        "Retry up to this many ms on transient errors (currently EACCES)", 0L,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        sink_pad, "in", "in",
        "Handle to a nvidia::deepstream::NvDsStaticInput component. Supported "
        "formats - ANY",
        gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
    return nvidia::gxf::ToResultCode(result);
  }

  nvidia::gxf::Parameter<bool> sync;
  nvidia::gxf::Parameter<int64_t> max_lateness;
  nvidia::gxf::Parameter<bool> qos;
  nvidia::gxf::Parameter<bool> async;
  nvidia::gxf::Parameter<int64_t> ts_offset;
  nvidia::gxf::Parameter<bool> enable_last_sample;
  nvidia::gxf::Parameter<uint64_t> blocksize;
  nvidia::gxf::Parameter<uint64_t> render_delay;
  nvidia::gxf::Parameter<uint64_t> throttle_time;
  nvidia::gxf::Parameter<uint64_t> max_bitrate;
  nvidia::gxf::Parameter<uint64_t> processing_deadline;
  nvidia::gxf::Parameter<std::string> location;
  nvidia::gxf::Parameter<int64_t> buffer_mode;
  nvidia::gxf::Parameter<uint64_t> buffer_size;
  nvidia::gxf::Parameter<bool> append;
  nvidia::gxf::Parameter<bool> o_sync;
  nvidia::gxf::Parameter<int64_t> max_transient_error_timeout;
  nvidia::gxf::Parameter<nvidia::gxf::Handle<NvDsStaticInput>> sink_pad;

 protected:
  GstElement *element_;
  const char *GST_ELEMENT_NAME = "filesink";
};

#define GXF_EXT_FACTORY_ADD_GstFileSink()                                  \
  do {                                                                     \
    GXF_EXT_FACTORY_ADD_VERBOSE(                                           \
        0x8f7558890fbc3070UL, 0x818ad701664a70b7UL,                        \
        nvidia::deepstream::GstFileSink, nvidia::deepstream::INvDsElement, \
        "GstFileSink placeholder display-name",                            \
        "GstFileSink placeholder brief", "Write stream to a file");        \
  } while (0)

}  // namespace deepstream
}  // namespace nvidia
