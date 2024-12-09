#pragma once

#include <string>

#include "nvdsbase/nvds_io.hpp"

namespace nvidia {
namespace deepstream {

class GstTimeOverlay : public INvDsElement {
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

    auto p_video_sink_pad = video_sink_pad.try_get();
    if (p_video_sink_pad) {
      p_video_sink_pad.value()->set_element(this);
      p_video_sink_pad.value()->set_template_name("video_sink");
    }
    return GXF_SUCCESS;
  }

  gxf_result_t bin_add(GstElement *pipeline) override {
    GXF_LOG_INFO("bin_add: %s %s\n", GST_ELEMENT_NAME, name());
    if (!gst_bin_add(GST_BIN(pipeline), element_)) {
      return GXF_FAILURE;
    }

    auto p_text = text.try_get();
    if (p_text && p_text.value() != std::string{""}) {
      gchararray propvalue = (gchararray)p_text.value().c_str();
      g_object_set(element_, "text", propvalue, NULL);
    }

    auto p_shaded_background = shaded_background.try_get();
    if (p_shaded_background && p_shaded_background.value() != false) {
      gboolean propvalue = (gboolean)p_shaded_background.value();
      g_object_set(element_, "shaded-background", propvalue, NULL);
    }

    auto p_shading_value = shading_value.try_get();
    if (p_shading_value && p_shading_value.value() != 80UL) {
      guint propvalue = (guint)p_shading_value.value();
      g_object_set(element_, "shading-value", propvalue, NULL);
    }

    auto p_halignment = halignment.try_get();
    if (p_halignment && p_halignment.value() != 0L) {
      gint64 propvalue = (gint64)p_halignment.value();
      g_object_set(element_, "halignment", propvalue, NULL);
    }

    auto p_valignment = valignment.try_get();
    if (p_valignment && p_valignment.value() != 2L) {
      gint64 propvalue = (gint64)p_valignment.value();
      g_object_set(element_, "valignment", propvalue, NULL);
    }

    auto p_xpad = xpad.try_get();
    if (p_xpad && p_xpad.value() != 25L) {
      gint propvalue = (gint)p_xpad.value();
      g_object_set(element_, "xpad", propvalue, NULL);
    }

    auto p_ypad = ypad.try_get();
    if (p_ypad && p_ypad.value() != 25L) {
      gint propvalue = (gint)p_ypad.value();
      g_object_set(element_, "ypad", propvalue, NULL);
    }

    auto p_deltax = deltax.try_get();
    if (p_deltax && p_deltax.value() != 0L) {
      gint propvalue = (gint)p_deltax.value();
      g_object_set(element_, "deltax", propvalue, NULL);
    }

    auto p_deltay = deltay.try_get();
    if (p_deltay && p_deltay.value() != 0L) {
      gint propvalue = (gint)p_deltay.value();
      g_object_set(element_, "deltay", propvalue, NULL);
    }

    auto p_xpos = xpos.try_get();
    if (p_xpos && p_xpos.value() != 0.5) {
      gdouble propvalue = (gdouble)p_xpos.value();
      g_object_set(element_, "xpos", propvalue, NULL);
    }

    auto p_ypos = ypos.try_get();
    if (p_ypos && p_ypos.value() != 0.5) {
      gdouble propvalue = (gdouble)p_ypos.value();
      g_object_set(element_, "ypos", propvalue, NULL);
    }

    auto p_x_absolute = x_absolute.try_get();
    if (p_x_absolute && p_x_absolute.value() != 0.5) {
      gdouble propvalue = (gdouble)p_x_absolute.value();
      g_object_set(element_, "x-absolute", propvalue, NULL);
    }

    auto p_y_absolute = y_absolute.try_get();
    if (p_y_absolute && p_y_absolute.value() != 0.5) {
      gdouble propvalue = (gdouble)p_y_absolute.value();
      g_object_set(element_, "y-absolute", propvalue, NULL);
    }

    auto p_wrap_mode = wrap_mode.try_get();
    if (p_wrap_mode && p_wrap_mode.value() != 2L) {
      gint64 propvalue = (gint64)p_wrap_mode.value();
      g_object_set(element_, "wrap-mode", propvalue, NULL);
    }

    auto p_font_desc = font_desc.try_get();
    if (p_font_desc && p_font_desc.value() != std::string{""}) {
      gchararray propvalue = (gchararray)p_font_desc.value().c_str();
      g_object_set(element_, "font-desc", propvalue, NULL);
    }

    auto p_silent = silent.try_get();
    if (p_silent && p_silent.value() != false) {
      gboolean propvalue = (gboolean)p_silent.value();
      g_object_set(element_, "silent", propvalue, NULL);
    }

    auto p_line_alignment = line_alignment.try_get();
    if (p_line_alignment && p_line_alignment.value() != 1L) {
      gint64 propvalue = (gint64)p_line_alignment.value();
      g_object_set(element_, "line-alignment", propvalue, NULL);
    }

    auto p_wait_text = wait_text.try_get();
    if (p_wait_text && p_wait_text.value() != true) {
      gboolean propvalue = (gboolean)p_wait_text.value();
      g_object_set(element_, "wait-text", propvalue, NULL);
    }

    auto p_auto_resize = auto_resize.try_get();
    if (p_auto_resize && p_auto_resize.value() != true) {
      gboolean propvalue = (gboolean)p_auto_resize.value();
      g_object_set(element_, "auto-resize", propvalue, NULL);
    }

    auto p_vertical_render = vertical_render.try_get();
    if (p_vertical_render && p_vertical_render.value() != false) {
      gboolean propvalue = (gboolean)p_vertical_render.value();
      g_object_set(element_, "vertical-render", propvalue, NULL);
    }

    auto p_scale_mode = scale_mode.try_get();
    if (p_scale_mode && p_scale_mode.value() != 0L) {
      gint64 propvalue = (gint64)p_scale_mode.value();
      g_object_set(element_, "scale-mode", propvalue, NULL);
    }

    auto p_scale_pixel_aspect_ratio = scale_pixel_aspect_ratio.try_get();
    if (p_scale_pixel_aspect_ratio &&
        p_scale_pixel_aspect_ratio.value() != std::string{""}) {
      GValue v = G_VALUE_INIT;
      g_value_init(&v, GST_TYPE_FRACTION);
      gst_value_deserialize(&v, p_scale_pixel_aspect_ratio.value().c_str());
      g_object_set_property(G_OBJECT(element_), "scale-pixel-aspect-ratio", &v);
      g_value_unset(&v);
    }

    auto p_color = color.try_get();
    if (p_color && p_color.value() != 4294967295UL) {
      guint propvalue = (guint)p_color.value();
      g_object_set(element_, "color", propvalue, NULL);
    }

    auto p_draw_shadow = draw_shadow.try_get();
    if (p_draw_shadow && p_draw_shadow.value() != true) {
      gboolean propvalue = (gboolean)p_draw_shadow.value();
      g_object_set(element_, "draw-shadow", propvalue, NULL);
    }

    auto p_draw_outline = draw_outline.try_get();
    if (p_draw_outline && p_draw_outline.value() != true) {
      gboolean propvalue = (gboolean)p_draw_outline.value();
      g_object_set(element_, "draw-outline", propvalue, NULL);
    }

    auto p_outline_color = outline_color.try_get();
    if (p_outline_color && p_outline_color.value() != 4278190080UL) {
      guint propvalue = (guint)p_outline_color.value();
      g_object_set(element_, "outline-color", propvalue, NULL);
    }

    auto p_time_mode = time_mode.try_get();
    if (p_time_mode && p_time_mode.value() != 0L) {
      gint64 propvalue = (gint64)p_time_mode.value();
      g_object_set(element_, "time-mode", propvalue, NULL);
    }

    auto p_show_times_as_dates = show_times_as_dates.try_get();
    if (p_show_times_as_dates && p_show_times_as_dates.value() != false) {
      gboolean propvalue = (gboolean)p_show_times_as_dates.value();
      g_object_set(element_, "show-times-as-dates", propvalue, NULL);
    }

    /* auto p_datetime_epoch = datetime_epoch.try_get();
    if (p_datetime_epoch && p_datetime_epoch.value() != <GLib.DateTime object at
    0x7ca4ac6ba440 (GDateTime at 0x563379aaacd0)>) { GDateTime propvalue =
    (GDateTime) p_datetime_epoch.value(); g_object_set (element_,
    "datetime-epoch", propvalue, NULL);
    } */

    auto p_datetime_format = datetime_format.try_get();
    if (p_datetime_format &&
        p_datetime_format.value() != std::string{"%F %T"}) {
      gchararray propvalue = (gchararray)p_datetime_format.value().c_str();
      g_object_set(element_, "datetime-format", propvalue, NULL);
    }

    return GXF_SUCCESS;
  }

  GstElement *get_element_ptr() override { return element_; }

  gxf_result_t registerInterface(nvidia::gxf::Registrar *registrar) override {
    nvidia::gxf::Expected<void> result;
    result &=
        registrar->parameter(text, "text", "text", "Text to be display.",
                             std::string{""}, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        shaded_background, "shaded-background", "shaded background",
        "Whether to shade the background under the text area", false,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        shading_value, "shading-value", "background shading value",
        "Shading value to apply if shaded-background is true", 80UL,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        halignment, "halignment", "horizontal alignment",
        "Horizontal alignment of the text\nValid values:\n  0: left\n  1: "
        "center\n  2: right\n  4: Absolute position clamped to canvas\n  5: "
        "Absolute position",
        0L, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        valignment, "valignment", "vertical alignment",
        "Vertical alignment of the text\nValid values:\n  0: baseline\n  1: "
        "bottom\n  2: top\n  3: Absolute position clamped to canvas\n  4: "
        "center\n  5: Absolute position",
        2L, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        xpad, "xpad", "horizontal paddding",
        "Horizontal paddding when using left/right alignment", 25L,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &=
        registrar->parameter(ypad, "ypad", "vertical padding",
                             "Vertical padding when using top/bottom alignment",
                             25L, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        deltax, "deltax", "X position modifier",
        "Shift X position to the left or to the right. Unit is pixels.", 0L,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &=
        registrar->parameter(deltay, "deltay", "Y position modifier",
                             "Shift Y position up or down. Unit is pixels.", 0L,
                             GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        xpos, "xpos", "horizontal position",
        "Horizontal position when using clamped position alignment", 0.5,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        ypos, "ypos", "vertical position",
        "Vertical position when using clamped position alignment", 0.5,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        x_absolute, "x-absolute", "horizontal position",
        "Horizontal position when using absolute alignment", 0.5,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &=
        registrar->parameter(y_absolute, "y-absolute", "vertical position",
                             "Vertical position when using absolute alignment",
                             0.5, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        wrap_mode, "wrap-mode", "wrap mode",
        "Whether to wrap the text and if so how.\nValid values:\n -1: none\n  "
        "0: word\n  1: char\n  2: wordchar",
        2L, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        font_desc, "font-desc", "font description",
        "Pango font description of font to be used for rendering. See "
        "documentation of pango_font_description_from_string for syntax.",
        std::string{""}, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(silent, "silent", "silent",
                                   "Whether to render the text string", false,
                                   GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        line_alignment, "line-alignment", "line alignment",
        "Alignment of text lines relative to each other.\nValid values:\n  0: "
        "left\n  1: center\n  2: right",
        1L, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(wait_text, "wait-text", "Wait Text",
                                   "Whether to wait for subtitles", true,
                                   GXF_PARAMETER_FLAGS_OPTIONAL);
    result &=
        registrar->parameter(auto_resize, "auto-resize", "auto resize",
                             "Automatically adjust font size to screen-size.",
                             true, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(vertical_render, "vertical-render",
                                   "vertical render", "Vertical Render.", false,
                                   GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        scale_mode, "scale-mode", "scale mode",
        "Scale text to compensate for and avoid distortion by subsequent video "
        "scaling.\nValid values:\n  0: none\n  1: par\n  2: display\n  3: user",
        0L, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        scale_pixel_aspect_ratio, "scale-pixel-aspect-ratio",
        "scale pixel aspect ratio",
        "Pixel aspect ratio of video scale to compensate for in user "
        "scale-mode. Format: <numerator>/<denominator>",
        std::string{""}, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(color, "color", "Color",
                                   "Color to use for text (big-endian ARGB).",
                                   4294967295UL, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(draw_shadow, "draw-shadow", "draw-shadow",
                                   "Whether to draw shadow", true,
                                   GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(draw_outline, "draw-outline", "draw-outline",
                                   "Whether to draw outline", true,
                                   GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        outline_color, "outline-color", "Text Outline Color",
        "Color to use for outline the text (big-endian ARGB).", 4278190080UL,
        GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        time_mode, "time-mode", "Time Mode",
        "What time to show\nValid values:\n  0: buffer-time\n  1: "
        "stream-time\n  2: running-time\n  3: time-code\n  4: "
        "elapsed-running-time",
        0L, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        show_times_as_dates, "show-times-as-dates", "Show times as dates",
        "Whether to display times, counted from datetime-epoch, as dates",
        false, GXF_PARAMETER_FLAGS_OPTIONAL);
    // result &= registrar->parameter(datetime_epoch, "datetime-epoch",
    // "Datetime Epoch", "When showing times as dates, the initial date from
    // which time is counted, if not specified prime epoch is used
    // (1900-01-01)", <GLib.DateTime object at 0x7ca4ac6ba440 (GDateTime at
    // 0x563379aaacd0)>, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        datetime_format, "datetime-format", "Datetime Format",
        "When showing times as dates, the format to render date and time in",
        std::string{"%F %T"}, GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        src_pad, "video-out", "video-out",
        "Handle to a nvidia::deepstream::NvDsStaticOutput component. Supported "
        "formats - video(ABGR64_LE, BGRA64_LE, AYUV64, ARGB64_LE, ARGB64, "
        "RGBA64_LE, ABGR64_BE, BGRA64_BE, ARGB64_BE, RGBA64_BE, GBRA_12LE, "
        "GBRA_12BE, Y412_LE, Y412_BE, A444_10LE, GBRA_10LE, A444_10BE, "
        "GBRA_10BE, A422_10LE, A422_10BE, A420_10LE, A420_10BE, RGB10A2_LE, "
        "BGR10A2_LE, Y410, GBRA, ABGR, VUYA, BGRA, AYUV, ARGB, RGBA, A420, "
        "AV12, Y444_16LE, Y444_16BE, v216, P016_LE, P016_BE, Y444_12LE, "
        "GBR_12LE, Y444_12BE, GBR_12BE, I422_12LE, I422_12BE, Y212_LE, "
        "Y212_BE, I420_12LE, I420_12BE, P012_LE, P012_BE, Y444_10LE, GBR_10LE, "
        "Y444_10BE, GBR_10BE, r210, I422_10LE, I422_10BE, NV16_10LE32, Y210, "
        "v210, UYVP, I420_10LE, I420_10BE, P010_10LE, NV12_10LE32, "
        "NV12_10LE40, P010_10BE, Y444, RGBP, GBR, BGRP, NV24, xBGR, BGRx, "
        "xRGB, RGBx, BGR, IYU2, v308, RGB, Y42B, NV61, NV16, VYUY, UYVY, YVYU, "
        "YUY2, I420, YV12, NV21, NV12, NV12_64Z32, NV12_4L4, NV12_32L32, Y41B, "
        "IYU1, YVU9, YUV9, RGB16, BGR16, RGB15, BGR15, RGB8P, GRAY16_LE, "
        "GRAY16_BE, GRAY10_LE32, GRAY8)",
        gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
    result &= registrar->parameter(
        video_sink_pad, "video-in", "video-in",
        "Handle to a nvidia::deepstream::NvDsStaticInput component. Supported "
        "formats - video(ABGR64_LE, BGRA64_LE, AYUV64, ARGB64_LE, ARGB64, "
        "RGBA64_LE, ABGR64_BE, BGRA64_BE, ARGB64_BE, RGBA64_BE, GBRA_12LE, "
        "GBRA_12BE, Y412_LE, Y412_BE, A444_10LE, GBRA_10LE, A444_10BE, "
        "GBRA_10BE, A422_10LE, A422_10BE, A420_10LE, A420_10BE, RGB10A2_LE, "
        "BGR10A2_LE, Y410, GBRA, ABGR, VUYA, BGRA, AYUV, ARGB, RGBA, A420, "
        "AV12, Y444_16LE, Y444_16BE, v216, P016_LE, P016_BE, Y444_12LE, "
        "GBR_12LE, Y444_12BE, GBR_12BE, I422_12LE, I422_12BE, Y212_LE, "
        "Y212_BE, I420_12LE, I420_12BE, P012_LE, P012_BE, Y444_10LE, GBR_10LE, "
        "Y444_10BE, GBR_10BE, r210, I422_10LE, I422_10BE, NV16_10LE32, Y210, "
        "v210, UYVP, I420_10LE, I420_10BE, P010_10LE, NV12_10LE32, "
        "NV12_10LE40, P010_10BE, Y444, RGBP, GBR, BGRP, NV24, xBGR, BGRx, "
        "xRGB, RGBx, BGR, IYU2, v308, RGB, Y42B, NV61, NV16, VYUY, UYVY, YVYU, "
        "YUY2, I420, YV12, NV21, NV12, NV12_64Z32, NV12_4L4, NV12_32L32, Y41B, "
        "IYU1, YVU9, YUV9, RGB16, BGR16, RGB15, BGR15, RGB8P, GRAY16_LE, "
        "GRAY16_BE, GRAY10_LE32, GRAY8)",
        gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
    return nvidia::gxf::ToResultCode(result);
  }

  nvidia::gxf::Parameter<std::string> text;
  nvidia::gxf::Parameter<bool> shaded_background;
  nvidia::gxf::Parameter<uint64_t> shading_value;
  nvidia::gxf::Parameter<int64_t> halignment;
  nvidia::gxf::Parameter<int64_t> valignment;
  nvidia::gxf::Parameter<int64_t> xpad;
  nvidia::gxf::Parameter<int64_t> ypad;
  nvidia::gxf::Parameter<int64_t> deltax;
  nvidia::gxf::Parameter<int64_t> deltay;
  nvidia::gxf::Parameter<double> xpos;
  nvidia::gxf::Parameter<double> ypos;
  nvidia::gxf::Parameter<double> x_absolute;
  nvidia::gxf::Parameter<double> y_absolute;
  nvidia::gxf::Parameter<int64_t> wrap_mode;
  nvidia::gxf::Parameter<std::string> font_desc;
  nvidia::gxf::Parameter<bool> silent;
  nvidia::gxf::Parameter<int64_t> line_alignment;
  nvidia::gxf::Parameter<bool> wait_text;
  nvidia::gxf::Parameter<bool> auto_resize;
  nvidia::gxf::Parameter<bool> vertical_render;
  nvidia::gxf::Parameter<int64_t> scale_mode;
  nvidia::gxf::Parameter<std::string> scale_pixel_aspect_ratio;
  nvidia::gxf::Parameter<uint64_t> color;
  nvidia::gxf::Parameter<bool> draw_shadow;
  nvidia::gxf::Parameter<bool> draw_outline;
  nvidia::gxf::Parameter<uint64_t> outline_color;
  nvidia::gxf::Parameter<int64_t> time_mode;
  nvidia::gxf::Parameter<bool> show_times_as_dates;
  // nvidia::gxf::Parameter<Unknown> datetime_epoch;
  nvidia::gxf::Parameter<std::string> datetime_format;
  nvidia::gxf::Parameter<nvidia::gxf::Handle<NvDsStaticOutput>> src_pad;
  nvidia::gxf::Parameter<nvidia::gxf::Handle<NvDsStaticInput>> video_sink_pad;

 protected:
  GstElement *element_;
  const char *GST_ELEMENT_NAME = "timeoverlay";
};

#define GXF_EXT_FACTORY_ADD_GstTimeOverlay()                                  \
  do {                                                                        \
    GXF_EXT_FACTORY_ADD_VERBOSE(                                              \
        0x813de5f5a1cf3759UL, 0x87b3bbc265997b6bUL,                           \
        nvidia::deepstream::GstTimeOverlay, nvidia::deepstream::INvDsElement, \
        "GstTimeOverlay placeholder display-name",                            \
        "GstTimeOverlay placeholder brief",                                   \
        "Overlays buffer time stamps on a video stream");                     \
  } while (0)

}  // namespace deepstream
}  // namespace nvidia
