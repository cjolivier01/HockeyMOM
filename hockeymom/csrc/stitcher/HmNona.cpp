#include "hockeymom/csrc/stitcher/HmNona.h"
#include "concurrentqueue/blockingconcurrentqueue.h"

#include <atomic>

namespace hm {
using namespace HuginBase;
using namespace HuginBase::Nona;

/**
 *  _    _           _   _
 * | |  | |         | \ | |
 * | |__| |_ __ ___ |  \| | ___  _ __   __ _
 * |  __  | '_ ` _ \| . ` |/ _ \| '_ \ / _` |
 * | |  | | | | | | | |\  | (_) | | | | (_| |
 * |_|  |_|_| |_| |_|_| \_|\___/|_| |_|\__,_|
 *
 *
 */
HmNona::HmNona(std::string project_file)
    : project_file_(std::move(project_file)) {
  {
    std::unique_lock<std::mutex> lk(gpu_thread_pool_mu_);
    ++nona_count_;
  }
  TIFFSetWarningHandler(0);
  if (!load_project(project_file_)) {
    throw std::runtime_error(
        std::string("Failed to laod project file: ") + project_file_);
  }
}

std::size_t HmNona::nona_count_{0};

HmNona::~HmNona() {
  std::unique_lock<std::mutex> lk(gpu_thread_pool_mu_);
  if (!--nona_count_) {
    gpu_thread_pool_.reset();
  }
}

bool HmNona::load_project(const std::string& project_file) {
  if (!pano_.ReadPTOFile(
          project_file, hugin_utils::getPathPrefix(project_file))) {
    return false;
  };
  opts_ = pano_.getOptions();
  opts_.tiffCompression = "NONE";
  opts_.outputPixelType = "UINT8";
  opts_.outputEMoRParams = pano_.getSrcImage(0).getEMoRParams();

  {
    std::unique_lock<std::mutex> lk(gpu_thread_pool_mu_);
    if (!gpu_thread_pool_) {
      gpu_thread_pool_ = std::make_unique<Eigen::ThreadPool>(1);
    }
  }
  ManualResetGate gate;
  gpu_thread_pool_->Schedule([&]() {
    set_thread_name("gpu_nona");
    opts_.remapUsingGPU = check_cuda_opengl();
    gate.signal();
  });
  gate.wait();
  pano_.setOptions(opts_);
  return true;
}

std::vector<std::unique_ptr<hm::MatrixRGB>> HmNona::remap_images(
    std::shared_ptr<hm::MatrixRGB> image1,
    std::shared_ptr<hm::MatrixRGB> image2) {
  ++image_pair_pass_count_;
  // Set up panorama options for the two images beforehand
  if (image_pair_pass_count_ == 1) {
    file_remapper_.setAdvancedOptions(adv_options_);
  }
  if (!pdisp_) {
    pdisp_ = std::make_unique<AppBase::DummyProgressDisplay>();
  }
  if (!stitcher_) {
    stitcher_ =
        std::make_unique<HmMultiImageRemapper<ImageType, vigra::BImage>>(
            pano_, pdisp_.get());
  }
  stitcher_->set_input_images(image1, image2);
  UIntSet img_indexes{0, 1};
  auto output_images = stitcher_->stitch(
      opts_,
      img_indexes,
      std::string("hm_nona-") + std::to_string(image_pair_pass_count_) + "-",
      file_remapper_,
      adv_options_,
      *gpu_thread_pool_);
  return output_images;
}

} // namespace hm
