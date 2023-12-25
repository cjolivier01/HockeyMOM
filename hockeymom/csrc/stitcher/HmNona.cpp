#include "hockeymom/csrc/stitcher/HmNona.h"
//#include "concurrentqueue/blockingconcurrentqueue.h"
#include "algorithms/basic/CalculateOptimalScale.h"
#include "panodata/Panorama.h"

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

std::vector<std::tuple<std::tuple<float, float>, std::tuple<float, float>>>
HmNona::get_control_points() const {
  std::vector<std::tuple<std::tuple<float, float>, std::tuple<float, float>>>
      results;
  std::cout << "HmNona::get_control_points()" << std::endl;
  const auto& control_points = pano_.getCtrlPoints();
  for (const HuginBase::ControlPoint& cp : control_points) {
    std::map<std::size_t, std::tuple<float, float>> thispt;
    if (cp.mode != HuginBase::ControlPoint::X_Y) {
      std::cerr << "Ignoring non X_Y control point" << std::endl;
      continue;
    }
    // assuming only two images 0 and 1
    assert(cp.image1Nr == 0 || cp.image1Nr == 1);
    assert(cp.image2Nr == 0 || cp.image2Nr == 1);
    thispt[cp.image1Nr] = std::tuple<float, float>(cp.x1, cp.y1);
    thispt[cp.image2Nr] = std::tuple<float, float>(cp.x2, cp.y2);
    results.emplace_back(std::make_tuple(thispt[0], thispt[1]));
  }
  return results;
}

std::vector<std::unique_ptr<hm::MatrixRGB>> HmNona::remap_images(
    std::shared_ptr<hm::MatrixRGB> image1,
    std::shared_ptr<hm::MatrixRGB> image2) {
  {
    std::scoped_lock lk(nona_init_mu_);
    // Set up panorama options for the two images beforehand
    if (++image_pair_pass_count_ == 1) {
      auto new_opt = pano_.getOptions();
      if (new_opt.fovCalcSupported(new_opt.getProjection())) {
        // calc optimal size of pano, only if projection is supported
        // otherwise use current width as start point
        long opt_width = hugin_utils::roundi(
            HuginBase::CalculateOptimalScale::calcOptimalScale(pano_) *
            new_opt.getWidth());
        // double sizeFactor = HUGIN_ASS_PANO_DOWNSIZE_FACTOR;
        double sizeFactor = 1.0;
        new_opt.setWidth(hugin_utils::floori(sizeFactor * opt_width), true);
      };

      // std::size_t ww = image1->cols() + image2->cols();
      pano_.setOptions(new_opt);
      opts_ = new_opt;
      file_remapper_.setAdvancedOptions(adv_options_);
    }
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
