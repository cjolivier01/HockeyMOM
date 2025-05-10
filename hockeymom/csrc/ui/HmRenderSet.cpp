#include "hockeymom/csrc/ui/HmRenderSet.h"

namespace hm {
namespace display {

namespace {
imageFormat get_image_format(int channels) {
  switch (channels) {
    case 4:
      return imageFormat::IMAGE_RGBA8;
    case 3:
      // return imageFormat::IMAGE_RGB8;
      return imageFormat::IMAGE_BGR8;
    default:
      assert(false);
      return imageFormat::IMAGE_UNKNOWN;
  }
}

std::mutex grs_mu;
std::shared_ptr<HmRenderSet> global_render_set;

} // namespace

std::weak_ptr<HmRenderSet> get_or_create_global_render_set() {
  std::unique_lock lk(grs_mu);
  if (!global_render_set) {
    global_render_set = std::make_shared<HmRenderSet>();
  }
  return global_render_set;
}

void destroy_global_render_set() {
  std::unique_lock lk(grs_mu);
  global_render_set.reset();
}

std::unique_ptr<glDisplay> HmRenderSet::create_video_output(
    const std::string& name,
    const DisplaySurface& surface) {
  videoOptions vo;
  vo.width = (int)surface.pitch_width();
  vo.height = (int)surface.height;
  auto video_output = std::unique_ptr<glDisplay>(glDisplay::Create(vo));
  video_output->SetTitle(name.c_str());
  return video_output;
}

videoOutput* HmRenderSet::get_video_output(
    const std::string& name,
    const DisplaySurface& surface) {
  std::unique_lock lk(mu_);
  auto found = video_outputs_.find(name);
  if (found == video_outputs_.end()) {
    found =
        video_outputs_.emplace(name, create_video_output(name, surface)).first;
  }
  return found->second.get();
}

bool HmRenderSet::render(
    const std::string& name,
    const DisplaySurface& surface,
    cudaStream_t stream) {
  return get_video_output(name, surface)
      ->Render(
          surface.d_ptr,
          surface.pitch_width(),
          surface.height,
          get_image_format(surface.channels),
          stream);
}

} // namespace display
} // namespace hm
