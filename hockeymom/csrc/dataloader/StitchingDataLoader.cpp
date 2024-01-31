#include "hockeymom/csrc/dataloader/StitchingDataLoader.h"
#include "hockeymom/csrc/stitcher/HmNona.h"

#include <unistd.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace hm {

//#define FAKE_REMAP // ~4 fps
//#define FAKE_BLEND // ~10 fps

namespace {
constexpr std::size_t kPrintInterval = 50;

std::shared_ptr<MatrixRGB> tensor_to_matrix_rgb_image(
    at::Tensor tensor,
    const std::vector<int>& xy_pos) {
  // Get the dimensions of the tensor
  auto dims = tensor.sizes();
  std::vector<std::ptrdiff_t> shape(dims.begin(), dims.end());

  // Ensure the tensor is of type uint8_t
  if (tensor.dtype() != at::kByte) {
    tensor = tensor.to(at::kByte);
  }

  auto strides_array_ref = tensor.strides();
  std::vector<std::size_t> strides{
      strides_array_ref.begin(), strides_array_ref.end()};
  std::transform(
      strides.begin(),
      strides.end(),
      strides.begin(),
      [tensor](std::ptrdiff_t stride) {
        return stride * tensor.element_size();
      });
  // Create a NumPy array with the same data
  std::size_t rows = shape[0];
  std::size_t cols = shape[1];
  std::size_t channels = shape[2];
  assert(channels == 3 || channels == 4);

  // #ifdef FAKE_BLEND
  // bool copy_data = true;
  // #else
  bool copy_data = false;
  //#endif

  // py::array_t<uint8_t> array(shape, strides, tensor.data_ptr<uint8_t>());
  // auto rgb = std::make_shared<MatrixRGB>(array, /*copy_data=*/false);
  auto rgb = std::make_shared<MatrixRGB>(
      tensor, xy_pos.at(0), xy_pos.at(1), /*copy_data=*/copy_data);
  // auto rgb = std::make_shared<MatrixRGB>(rows, cols, channels, strides);
  // memcpy(rgb->data(), tensor.data_ptr<uint8_t>(), rows * cols * channels);
  // rgb->set_xy_pos(xy_pos.at(0), xy_pos.at(1));
  return rgb;
}

} // namespace

StitchingDataLoader::StitchingDataLoader(
    std::size_t start_frame_id,
    std::string project_file,
    std::string seam_file,
    std::string xor_mask_file,
    bool save_seam_and_xor_mask,
    std::size_t max_queue_size,
    std::size_t remap_thread_count,
    std::size_t blend_thread_count)
    : project_file_(std::move(project_file)),
      seam_file_(std::move(seam_file)),
      xor_mask_file_(std::move(xor_mask_file)),
      save_seam_and_xor_mask_(save_seam_and_xor_mask),
      max_queue_size_(max_queue_size),
      next_frame_id_(start_frame_id),
      remap_thread_count_(remap_thread_count),
      blend_thread_count_(blend_thread_count),
      input_queue_(std::make_shared<JobRunnerT::InputQueue>(
          /*(int(remap_thread_count * 1.5))*/)),
      remap_runner_(
          input_queue_,
          [this](
              std::size_t worker_index,
              StitchingDataLoader::FRAME_DATA_TYPE&& frame) {
            return this->remap_worker(worker_index, std::move(frame));
          }),
      blend_runner_(
          remap_runner_.outputs(),
          [this](
              std::size_t worker_index,
              StitchingDataLoader::FRAME_DATA_TYPE&& frame) {
            return this->blend_worker(worker_index, std::move(frame));
          }),
      thread_pool_(std::make_unique<Eigen::ThreadPool>(4)),
      remap_thread_pool_(std::make_unique<HmThreadPool>(thread_pool_.get())),
      remap_inner_("remap_inner"),
      remap_outer_("remap_outer"),
      blend_inner_("blend_inner"),
      blend_outer_("blend_outer") {
  initialize();
}

StitchingDataLoader::~StitchingDataLoader() {
  shutdown();
}

void StitchingDataLoader::configure_remapper(
    std::vector<ops::RemapperConfig> remapper_configs) {
  remapper_configs_ = std::move(remapper_configs);
  remappers_.clear();
}

std::shared_ptr<ops::ImageRemapper> StitchingDataLoader::get_remapper(
    std::size_t image_index) {
  std::scoped_lock lk(nonas_create_mu_);
  if (!remapper_configs_.empty()) {
    std::shared_ptr<ops::ImageRemapper> remapper;
    if (remappers_.empty()) {
      for (const auto& config : remapper_configs_) {
        remapper = std::make_shared<ops::ImageRemapper>(
            config.src_width,
            config.src_height,
            config.col_map,
            config.row_map,
            config.add_alpha_channel,
            config.interpolation);
        remapper->init(config.batch_size);
        remapper->to(config.device);
        remappers_.emplace_back(remapper);
      }
    }
    return remappers_.at(image_index);
  }
  return nullptr;
}

void StitchingDataLoader::shutdown() {
  remap_runner_.stop();
  blend_runner_.stop();
  nonas_.clear();
}

void StitchingDataLoader::initialize() {
  assert(nonas_.empty());
  nonas_.resize(remap_thread_count_);
  remap_runner_.start(remap_thread_count_);
  blend_runner_.start(blend_thread_count_);
#ifndef FAKE_BLEND
  std::vector<std::string> args;
  if (!seam_file_.empty()) {
    std::string seam_file_arg(
        save_seam_and_xor_mask_ ? "--save-seams" : "--load-seams");
    // args.emplace_back(std::move(seam_file_arg));
    // seam_file_arg += seam_file_;
    // args.emplace_back(seam_file_);
  }
  if (!xor_mask_file_.empty() && save_seam_and_xor_mask_) {
    // args.emplace_back(std::string("--save-xor=") + xor_mask_file_);
  }
  enblender_ = std::make_shared<enblend::EnBlender>(args);
#endif
}

void StitchingDataLoader::add_frame(
    std::size_t frame_id,
    std::vector<std::shared_ptr<MatrixRGB>>&& images) {
  auto frame_info = std::make_shared<FrameData>();
  frame_info->frame_id = frame_id;
  frame_info->input_images = std::move(images);
  remap_runner_.inputs()->enqueue(frame_id, std::move(frame_info));
}

std::shared_ptr<MatrixRGB> StitchingDataLoader::get_stitched_frame(
    std::size_t frame_id) {
  auto final_frame = blend_runner_.outputs()->dequeue_key(frame_id);
  if (!final_frame) {
    // Closing down
    return nullptr;
  }
  return std::move(final_frame->blended_image);
}

void StitchingDataLoader::add_torch_frame(
    std::size_t frame_id,
    at::Tensor image_1,
    at::Tensor image_2) {
  auto frame_info = std::make_shared<FrameData>();
  frame_info->frame_id = frame_id;
  frame_info->torch_input_images = {
      FrameData::TorchImage{
          .tensor = image_1,
          .xy_pos =
              {remapper_configs_.at(0).x_pos, remapper_configs_.at(0).y_pos},
      },
      FrameData::TorchImage{
          .tensor = image_2,
          .xy_pos =
              {remapper_configs_.at(1).x_pos, remapper_configs_.at(1).y_pos},
      },
  };
  remap_runner_.inputs()->enqueue(frame_id, std::move(frame_info));
}

void StitchingDataLoader::add_remapped_frame(
    std::size_t frame_id,
    std::vector<std::shared_ptr<MatrixRGB>>&& images) {
  auto frame_info = std::make_shared<FrameData>();
  frame_info->frame_id = frame_id;
  frame_info->remapped_images = std::move(images);
  blend_runner_.inputs()->enqueue(frame_id, std::move(frame_info));
}

const std::shared_ptr<HmNona>& StitchingDataLoader::get_nona_worker(
    std::size_t worker_index) {
  std::scoped_lock lk(nonas_create_mu_);
  if (!nonas_.at(worker_index)) {
    set_thread_name("remapper", worker_index);
    assert(worker_index < nonas_.size());
    nonas_[worker_index] = std::make_unique<HmNona>(project_file_);
  }
  return nonas_.at(worker_index);
}

StitchingDataLoader::FRAME_DATA_TYPE StitchingDataLoader::remap_worker(
    std::size_t worker_index,
    StitchingDataLoader::FRAME_DATA_TYPE&& frame) {
  try {
    if (frame->input_images.empty() && frame->torch_input_images.empty()) {
      // Shutting down
      frame->remapped_images.clear();
      frame->torch_remapped_images.clear();
      return frame;
    }
#ifdef FAKE_REMAP
    frame->remapped_images.clear();
    frame->remapped_images.emplace_back(std::move(frame->input_images.at(0)));
    frame->remapped_images.emplace_back(std::move(frame->input_images.at(1)));
    frame->remapped_images.at(0)->set_xy_pos(0, 42);
    frame->remapped_images.at(1)->set_xy_pos(12, 255);
#else
    // Timer stuff
    //const bool is_first_timed = remap_inner_.count() == 0;
    //remap_inner_.tic();

    // The actual work
    if (!frame->torch_input_images.empty()) {
      HmThreadPool local_pool(thread_pool_.get());
      frame->torch_remapped_images.resize(frame->torch_input_images.size());
      frame->remapped_images.resize(frame->torch_input_images.size());
      for (std::size_t i = 0; i < frame->torch_input_images.size(); ++i) {
        auto img = frame->torch_input_images[i];
        auto remapper = get_remapper(i);
        assert(remapper);
        local_pool.Schedule([this, frame, index = i, img, r = remapper]() {
          at::Tensor remapped = remappers_.at(index)->forward(
              img.tensor.to(remapper_configs_.at(index).device));

          // Prepare layout for blending
          assert(remapped.size(0) == 1); // batch dimensions
          remapped = remapped.squeeze(0).permute({1, 2, 0});
          remapped = remapped.contiguous().cpu();

          frame->torch_remapped_images.at(index) = FrameData::TorchImage{
              .tensor = remapped,
              .xy_pos = img.xy_pos,
          };
          frame->remapped_images.at(index) = tensor_to_matrix_rgb_image(
              frame->torch_remapped_images.at(index).tensor,
              frame->torch_remapped_images.at(index).xy_pos);
        });
        local_pool.join_all();
      }
    } else {
      auto nona = get_nona_worker(worker_index);
      auto remapped = nona->remap_images(
          std::move(frame->input_images.at(0)),
          std::move(frame->input_images.at(1)));
      frame->remapped_images.clear();
      for (auto& r : remapped) {
        frame->remapped_images.emplace_back(std::move(r));
      }
    }

    // More timer stuff
    // double outer_remap_fps = 0.0;
    //remap_inner_.toc();
    // if (!is_first_timed) {
    //   remap_outer_.toc();
    //   outer_remap_fps = remap_outer_.fps();
    // }
    // remap_outer_.tic();
    //if ((remap_inner_.count() % kPrintInterval) == 0) {
      //std::cout << remap_inner_ << std::endl;
      //remap_inner_.reset();
    //}
    // if (outer_remap_fps != 0.0 &&
    //     (remap_outer_.count() % kPrintInterval) == 0) {
    //   std::cout << remap_outer_ << std::endl;
    //   remap_outer_.reset();
    // }

#endif
  } catch (std::exception e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    throw;
  }
  return frame;
}

StitchingDataLoader::FRAME_DATA_TYPE StitchingDataLoader::blend_worker(
    std::size_t worker_index,
    StitchingDataLoader::FRAME_DATA_TYPE&& frame) {
  try {
    if (frame->remapped_images.empty()) {
      // Shutting down, make sure that blended_image is null
      frame->blended_image.reset();
      return frame;
    }
    auto blender = enblender_;
#ifdef FAKE_BLEND
    // frame->remapped_images.at(0) = tensor_to_matrix_rgb_image(
    //     frame->torch_remapped_images.at(0).tensor,
    //     frame->torch_remapped_images.at(0).xy_pos);

    frame->blended_image = frame->remapped_images[0];
#else
    // Timer stuff
    blend_inner_.tic();

    frame->blended_image = blender->blend_images(frame->remapped_images);

    // More timer stuff
    blend_inner_.toc();
    if ((blend_inner_.count() % kPrintInterval) == 0) {
      std::cout << blend_inner_ << std::endl;
      //blend_inner_.reset();
    }
#endif
  } catch (...) {
    std::cerr << "Caught exception" << std::endl;
    assert(false);
  }
  return frame;
}

} // namespace hm
