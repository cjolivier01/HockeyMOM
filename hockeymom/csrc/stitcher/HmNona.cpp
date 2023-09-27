#include "hockeymom/csrc/stitcher/HmNona.h"
#include "hockeymom/csrc/stitcher/HmStitcher.h"
#include "concurrentqueue/blockingconcurrentqueue.h"

// #include "hugin/src/hugin_base/nona/Stitcher.h"

// #include <hugin_config.h>
// #include <fstream>
// #include <sstream>

#include <atomic>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>

// #include <algorithm>
// #include <cctype>
// #include <string>

// #include <vigra/error.hxx>

// #include <getopt.h>

// #include <algorithms/nona/NonaFileStitcher.h>
// #include <hugin_basic.h>
// #include <hugin_utils/platform.h>
// #include <vigra_ext/ImageTransformsGPU.h>
// #include "hugin_base/algorithms/basic/LayerStacks.h"
// #include "hugin_utils/stl_utils.h"
// #include "nona/StitcherOptions.h"

// #include <tbb/tbb.h>
// #include <tbb/parallel_pipeline.h>
// #include <tbb/global_control.h>

namespace hm {
using namespace HuginBase;
using namespace HuginBase::Nona;

namespace {

// struct StitcherWorker : public Worker{
//   std::unique_ptr<HmMultiImageRemapper<ImageType, vigra::BImage>> stitcher_;
//   ~StitcherWorker() {
//     if (thread_.joinable()) {6
//       StitcherWorker.join();
//     }
//   }
// };

bool check_cuda_opengl() {
  static bool attempted = false;
  static bool attempted_result = false;
  if (attempted) {
    return attempted_result;
  }
  attempted = true;

  int argc = 2;
  char *argv[2] = {"python", "-g"};
  attempted_result = hugin_utils::initGPU(&argc, &argv[0]);
  return attempted_result;

  // // Request an OpenGL 3.3 core profile context
  // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // //assert(false);
  // // Initialize GLFW and create an OpenGL context
  // if (!glfwInit()) {
  //   fprintf(stderr, "Failed to initialize GLFW\n");
  //   return false;
  // }

  // // Create an OpenGL context
  // GLFWwindow* window = glfwCreateWindow(100, 100, "OpenGL Window", NULL, NULL);
  // if (!window) {
  //   fprintf(stderr, "Failed to create GLFW window\n");
  //   glfwTerminate();
  //   return false;
  // }
  // glfwHideWindow(window);

  // // // Make the OpenGL context current
  // glfwMakeContextCurrent(window);

  // // Initialize GLEW (or equivalent)
  // GLenum err = glewInit();
  // if (err != GLEW_OK) {
  //   fprintf(stderr, "GLEW initialization error: %s\n", glewGetErrorString(err));
  //   glfwTerminate();
  //   return false;
  // }

  // // Query and print the OpenGL vendor string
  // const char* vendor = (const char*)glGetString(GL_VENDOR);
  // if (vendor) {
  //   printf("OpenGL Vendor: %s\n", vendor);
  //   attempted_result = true;
  //   return true;
  // } else {
  //   GLenum err = glGetError();
  //   fprintf(stderr, "Unable to retrieve OpenGL vendor string: %d\n", err);
  //   return false;
  // }
}

class InputStage {
 public:
  explicit InputStage(std::vector<int>& input) : input_(input), index_(0) {}

  bool operator()(int& output) {
    if (index_ < input_.size()) {
      output = input_[index_++];
      return true;
    }
    return false; // End of stream
  }

 private:
  std::vector<int>& input_;
  std::atomic<size_t> index_;
};

class ProcessingStage {
 public:
  int operator()(int input) {
    // Perform some processing on the input
    return input * input; // Square the input
  }
};

class OutputStage {
 public:
  void operator()(int result) {
    // Print the processed result
    std::cout << "Processed Result: " << result << std::endl;
  }
};
} // namespace

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
  TIFFSetWarningHandler(0);
  if (!load_project(project_file_)) {
    throw std::runtime_error(
        std::string("Failed to laod project file: ") + project_file_);
  }
}

bool HmNona::load_project(const std::string& project_file) {
  std::cout << "Project file: " << project_file_ << std::endl;

  if (!pano_.ReadPTOFile(
          project_file, hugin_utils::getPathPrefix(project_file))) {
    return false;
  };
  opts_ = pano_.getOptions();
  opts_.tiffCompression = "NONE";
  opts_.outputPixelType = "UINT8";
  opts_.outputEMoRParams = pano_.getSrcImage(0).getEMoRParams();
  opts_.remapUsingGPU = check_cuda_opengl();
  pano_.setOptions(opts_);
  return true;
}

void HmNona::run_pipeline() {
  // Create an input data vector
  std::vector<int> inputData{1, 2, 3, 4, 5};

  // Define the pipeline stages
  InputStage inputStage(inputData);
  ProcessingStage processingStage;
  OutputStage outputStage;

  // // Create the pipeline
  // tbb::parallel_pipeline(
  //     /* Number of tokens */ 30,
  //     tbb::make_filter<void, int>(tbb::filter_mode::serial_in_order, std::move(inputStage)) &
  //         tbb::make_filter<int, int>(tbb::filter_mode::parallel, std::move(processingStage)) &
  //         tbb::make_filter<int, void>(
  //             tbb::filter_mode::serial_in_order, std::move(outputStage)));
}

std::vector<std::unique_ptr<hm::MatrixRGB>> HmNona::process_images(
    std::shared_ptr<hm::MatrixRGB> image1,
    std::shared_ptr<hm::MatrixRGB> image2) {
  ++image_pair_pass_count_;
  // Set up panorama options for the two images beforehand
  static auto pdisp = std::make_unique<AppBase::DummyProgressDisplay>();
  if (image_pair_pass_count_ == 1) {
    file_remapper_.setAdvancedOptions(adv_options_);
  }
  // TODO: make class member
  static auto stitcher =
      std::make_unique<HmMultiImageRemapper<ImageType, vigra::BImage>>(
          pano_, pdisp.get());
  stitcher->set_input_images(image1, image2);
  UIntSet img_indexes{0, 1};
  auto output_images = stitcher->stitch(
      opts_,
      img_indexes,
      std::string("hm_nona-") + std::to_string(image_pair_pass_count_) + "-",
      file_remapper_,
      adv_options_);
  return output_images;
}

HmNona::~HmNona() {}

} // namespace hm
