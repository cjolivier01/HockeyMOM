#include "hockeymom/csrc/dataloader/StitchingDataLoader.h"
#include "hockeymom/csrc/mblend/mblend.h"
#include "hockeymom/csrc/postprocess/ImagePostProcess.h"
#include "hockeymom/csrc/pytorch/image_blend.h"
#include "hockeymom/csrc/pytorch/image_remap.h"
#include "hockeymom/csrc/stitcher/HmNona.h"
#include "hockeymom/csrc/video/video_writer.h"

#include <iostream>

#include <torch/extension.h>
#include <torch/torch.h>

PYBIND11_MAKE_OPAQUE(std::map<std::string, std::complex<double>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::pair<std::string, double>>);

namespace py = pybind11;

PYBIND11_MODULE(_hockeymom, m) {
  hm::init_stack_trace();

  py::class_<hm::HMPostprocessConfig, std::shared_ptr<hm::HMPostprocessConfig>>(
      m, "HMPostprocessConfig")
      .def(py::init<>())
      .def("to_string", &hm::HMPostprocessConfig::to_string)
      .def_readwrite("show_image", &hm::HMPostprocessConfig::show_image)
      .def_readwrite(
          "plot_individual_player_tracking",
          &hm::HMPostprocessConfig::plot_individual_player_tracking)
      .def_readwrite(
          "plot_cluster_tracking",
          &hm::HMPostprocessConfig::plot_cluster_tracking)
      .def_readwrite(
          "plot_camera_tracking",
          &hm::HMPostprocessConfig::plot_camera_tracking)
      .def_readwrite("plot_speed", &hm::HMPostprocessConfig::plot_speed)
      .def_readwrite(
          "max_in_aspec_ratio", &hm::HMPostprocessConfig::max_in_aspec_ratio)
      .def_readwrite(
          "no_max_in_aspec_ratio_at_edges",
          &hm::HMPostprocessConfig::no_max_in_aspec_ratio_at_edges)
      .def_readwrite(
          "apply_fixed_edge_scaling",
          &hm::HMPostprocessConfig::apply_fixed_edge_scaling)
      .def_readwrite(
          "fixed_edge_scaling_factor",
          &hm::HMPostprocessConfig::fixed_edge_scaling_factor)
      .def_readwrite(
          "fixed_edge_rotation", &hm::HMPostprocessConfig::fixed_edge_rotation)
      .def_readwrite(
          "plot_sticky_camera", &hm::HMPostprocessConfig::plot_sticky_camera)
      .def_readwrite(
          "crop_output_image", &hm::HMPostprocessConfig::crop_output_image)
      .def_readwrite("use_cuda", &hm::HMPostprocessConfig::use_cuda)
      .def_readwrite("use_watermark", &hm::HMPostprocessConfig::use_watermark);

  py::class_<hm::ImagePostProcessor, std::shared_ptr<hm::ImagePostProcessor>>(
      m, "ImagePostProcessor")
      .def(py::init<std::shared_ptr<hm::HMPostprocessConfig>, std::string>());

  py::class_<hm::ops::RemapperConfig, std::shared_ptr<hm::ops::RemapperConfig>>(
      m, "RemapperConfig")
      .def(py::init<>())
      .def_readwrite("src_width", &hm::ops::RemapperConfig::src_width)
      .def_readwrite("src_height", &hm::ops::RemapperConfig::src_height)
      .def_readwrite("x_pos", &hm::ops::RemapperConfig::x_pos)
      .def_readwrite("y_pos", &hm::ops::RemapperConfig::y_pos)
      .def_readwrite("col_map", &hm::ops::RemapperConfig::col_map)
      .def_readwrite("row_map", &hm::ops::RemapperConfig::row_map)
      .def_readwrite(
          "add_alpha_channel", &hm::ops::RemapperConfig::add_alpha_channel)
      .def_readwrite("interpolation", &hm::ops::RemapperConfig::interpolation)
      .def_readwrite("batch_size", &hm::ops::RemapperConfig::batch_size)
      .def_readwrite("device", &hm::ops::RemapperConfig::device);

  py::class_<hm::ops::BlenderConfig, std::shared_ptr<hm::ops::BlenderConfig>>(
      m, "BlenderConfig")
      .def(py::init<>())
      .def_readwrite("mode", &hm::ops::BlenderConfig::mode)
      .def_readwrite("levels", &hm::ops::BlenderConfig::levels)
      .def_readwrite("seam", &hm::ops::BlenderConfig::seam)
      .def_readwrite("xor_map", &hm::ops::BlenderConfig::xor_map)
      .def_readwrite("interpolation", &hm::ops::BlenderConfig::interpolation)
      .def_readwrite("device", &hm::ops::BlenderConfig::device);

  py::class_<hm::StitchingDataLoader, std::shared_ptr<hm::StitchingDataLoader>>(
      m, "StitchingDataLoader")
      .def(py::init<
           std::size_t,
           std::string,
           std::string,
           std::string,
           bool,
           std::size_t,
           std::size_t,
           std::size_t>())
      .def("fps", &hm::StitchingDataLoader::fps)
      .def(
          "configure_remapper",
          &hm::StitchingDataLoader::configure_remapper,
          py::arg("remapper_config"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "configure_blender",
          &hm::StitchingDataLoader::configure_blender,
          py::arg("blender_config"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "add_frame",
          [](std::shared_ptr<hm::StitchingDataLoader> data_loader,
             std::size_t frame_id,
             std::optional<py::array_t<uint8_t>> image1,
             std::optional<py::array_t<uint8_t>> image2) {
            // We expect a three-channel RGB image here
            if (!image1 && !image2) {
              // Exiting
              py::gil_scoped_release release_gil;
              data_loader->add_frame(frame_id, {});
              return frame_id;
            }
            assert(image1->ndim() == 3);
            assert(image2->ndim() == 3);
            auto m1 = std::make_shared<hm::MatrixRGB>(*image1, 0, 0);
            auto m2 = std::make_shared<hm::MatrixRGB>(*image2, 0, 0);
            {
              py::gil_scoped_release release_gil;
              data_loader->add_frame(frame_id, {std::move(m1), std::move(m2)});
            }
            return frame_id;
          })
      .def(
          "add_torch_frame",
          &hm::StitchingDataLoader::add_torch_frame,
          py::arg("frame_id"),
          py::arg("image_1"),
          py::arg("image_2"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "add_remapped_frame",
          [](std::shared_ptr<hm::StitchingDataLoader> data_loader,
             std::size_t frame_id,
             std::optional<py::array_t<uint8_t>> image1,
             const std::vector<std::size_t>& xy_pos_1,
             std::optional<py::array_t<uint8_t>> image2,
             const std::vector<std::size_t>& xy_pos_2) {
            // We expect a three-channel RGB image here
            if (!image1 && !image2) {
              // Exiting
              py::gil_scoped_release release_gil;
              data_loader->add_frame(frame_id, {});
              return frame_id;
            }
            assert(image1->ndim() == 3);
            assert(image2->ndim() == 3);
            auto m1 = std::make_shared<hm::MatrixRGB>(
                *image1, xy_pos_1.at(0), xy_pos_1.at(1));
            auto m2 = std::make_shared<hm::MatrixRGB>(
                *image2, xy_pos_2.at(0), xy_pos_2.at(1));
            {
              py::gil_scoped_release release_gil;
              data_loader->add_remapped_frame(
                  frame_id, {std::move(m1), std::move(m2)});
            }
            return frame_id;
          })
      .def(
          "get_stitched_frame",
          [](std::shared_ptr<hm::StitchingDataLoader> data_loader,
             std::size_t frame_id) -> std::optional<at::Tensor> {
            std::shared_ptr<hm::MatrixRGB> stitched_image;
            {
              py::gil_scoped_release release_gil;
              stitched_image = data_loader->get_stitched_frame(frame_id);
              if (!stitched_image) {
                return std::nullopt;
              }
              return stitched_image->to_tensor();
            }
          })
      .def(
          "get_stitched_pytorch_frame",
          [](std::shared_ptr<hm::StitchingDataLoader> data_loader,
             std::size_t frame_id) -> std::optional<at::Tensor> {
            py::gil_scoped_release release_gil;
            at::Tensor stitched_image =
                data_loader->get_stitched_pytorch_frame(frame_id);
            if (!stitched_image.defined()) {
              return std::nullopt;
            }
            return stitched_image;
          },
          py::arg("frame_id"));

  py::class_<
      hm::av::FFmpegVideoWriter,
      std::shared_ptr<hm::av::FFmpegVideoWriter>>(m, "FFmpegVideoWriter")
      .def(py::init<>())
      .def(
          "open",
          &hm::av::FFmpegVideoWriter::open,
          py::arg("filename"),
          py::arg("codec_name"),
          py::arg("fps"),
          py::arg("frame_size"),
          py::arg("isColor"))
      .def("isOpenned", &hm::av::FFmpegVideoWriter::isOpened)
      .def("release", &hm::av::FFmpegVideoWriter::release)
      .def("write", &hm::av::FFmpegVideoWriter::write_v);
  ;

  using SortedPyArrayUin8Queue =
      hm::SortedQueue<std::size_t, std::unique_ptr<py::array_t<std::uint8_t>>>;
  py::class_<SortedPyArrayUin8Queue, std::shared_ptr<SortedPyArrayUin8Queue>>(
      m, "SortedPyArrayUin8Queue")
      .def(py::init<>())
      .def(
          "enqueue",
          [](const std::shared_ptr<SortedPyArrayUin8Queue>& sq,
             std::size_t key,
             py::array_t<std::uint8_t> array) -> void {
            sq->enqueue(
                key,
                std::make_unique<py::array_t<std::uint8_t>>(std::move(array)));
          })
      .def(
          "dequeue_key",
          [](const std::shared_ptr<SortedPyArrayUin8Queue>& sq,
             std::size_t key) -> py::array_t<std::uint8_t> {
            std::unique_ptr<py::array_t<std::uint8_t>> result;
            {
              py::gil_scoped_release release;
              result = sq->dequeue_key(key);
            }
            return std::move(*result);
          })
      .def(
          "dequeue_smallest_key",
          [](const std::shared_ptr<SortedPyArrayUin8Queue>& sq) {
            std::size_t key = ~0;
            std::unique_ptr<py::array_t<std::uint8_t>> result;
            {
              py::gil_scoped_release release;
              result = sq->dequeue_smallest_key(&key);
            }
            return std::make_tuple(key, std::move(*result));
          });

  using SortedRGBImageQueue =
      hm::SortedQueue<std::size_t, std::unique_ptr<hm::MatrixRGB>>;

  py::class_<SortedRGBImageQueue, std::shared_ptr<SortedRGBImageQueue>>(
      m, "SortedRGBImageQueue")
      .def(py::init<>())
      .def(
          "identity",
          [](const std::shared_ptr<SortedRGBImageQueue>& sq,
             py::array_t<std::uint8_t>& array,
             bool copy_data) -> py::array_t<std::uint8_t> {
            auto matrix =
                std::make_unique<hm::MatrixRGB>(array, 0, 0, copy_data);
            {
              // Unlock the GIL in order to let python muck with the input array
              // if it wants to
              py::gil_scoped_release release;
            }
            return matrix->to_py_array();
          })
      .def(
          "enqueue",
          [](const std::shared_ptr<SortedRGBImageQueue>& sq,
             std::size_t key,
             py::array_t<std::uint8_t> array,
             bool copy_data) -> void {
            auto matrix =
                std::make_unique<hm::MatrixRGB>(array, 0, 0, copy_data);
            {
              py::gil_scoped_release release;
              sq->enqueue(key, std::move(matrix));
            }
          })
      .def(
          "dequeue_key",
          [](const std::shared_ptr<SortedRGBImageQueue>& sq,
             std::size_t key) -> py::array_t<std::uint8_t> {
            std::unique_ptr<hm::MatrixRGB> matrix;
            {
              py::gil_scoped_release release;
              matrix = sq->dequeue_key(key);
            }
            return matrix->to_py_array();
          })
      .def(
          "dequeue_smallest_key",
          [](const std::shared_ptr<SortedRGBImageQueue>& sq) {
            std::size_t key = ~0;
            std::unique_ptr<hm::MatrixRGB> matrix;
            {
              py::gil_scoped_release release;
              matrix = sq->dequeue_smallest_key(&key);
            }
            return std::make_tuple(key, matrix->to_py_array());
          });

  using SortedTensorQueue = hm::SortedQueue<std::size_t, at::Tensor>;
  py::class_<SortedTensorQueue, std::shared_ptr<SortedTensorQueue>>(
      m, "SortedTensorQueue")
      .def(py::init<>())
      .def(
          "enqueue",
          [](const std::shared_ptr<SortedTensorQueue>& sq,
             std::size_t key,
             at::Tensor tensor) -> void {
            sq->enqueue(key, std::move(tensor));
          })
      .def(
          "dequeue_key",
          [](const std::shared_ptr<SortedTensorQueue>& sq,
             std::size_t key) -> at::Tensor {
            py::gil_scoped_release release;
            return sq->dequeue_key(key);
          })
      .def(
          "dequeue_smallest_key",
          [](const std::shared_ptr<SortedTensorQueue>& sq)
              -> std::tuple<std::size_t, at::Tensor> {
            std::size_t key = ~0;
            std::unique_ptr<py::array_t<std::uint8_t>> result;
            py::gil_scoped_release release;
            auto tensor = sq->dequeue_smallest_key(&key);
            return std::make_tuple(key, tensor);
          });

  py::class_<hm::enblend::EnBlender, std::shared_ptr<hm::enblend::EnBlender>>(
      m, "EnBlender")
      .def(
          py::init<std::vector<std::string>>(),
          py::arg("args") = std::vector<std::string>{})
      .def(
          "blend_images",
          [](std::shared_ptr<hm::enblend::EnBlender> blender,
             py::array_t<std::uint8_t>& image1,
             const std::vector<std::size_t>& xy_pos_1,
             py::array_t<std::uint8_t>& image2,
             const std::vector<std::size_t>& xy_pos_2)
              -> py::array_t<std::uint8_t> {
            assert(image1.ndim() == 3);
            assert(image2.ndim() == 3);
            auto m1 = std::make_shared<hm::MatrixRGB>(
                image1, xy_pos_1.at(0), xy_pos_1.at(1), /*copy_data=*/true);
            auto m2 = std::make_shared<hm::MatrixRGB>(
                image2, xy_pos_2.at(0), xy_pos_2.at(1), /*copy_data=*/true);
            std::unique_ptr<hm::MatrixRGB> blended_image;
            {
              py::gil_scoped_release release_gil;
              blended_image = blender->blend_images(
                  std::vector<std::shared_ptr<hm::MatrixRGB>>{m1, m2});
            }
            py::array_t<std::uint8_t> result;
            result = blended_image->to_py_array();
            return result;
          },
          // They don't really have to be left/right here, just the proper order
          // image-1 and image-2, but just for the sake of consistency, lets
          // call it left and right
          py::arg("left_image"),
          py::arg("left_xy_pos"),
          py::arg("right_image"),
          py::arg("right_xy_pos"));

  m.def(
      "_add_to_stitching_data_loader",
      [](std::shared_ptr<hm::StitchingDataLoader> data_loader,
         std::size_t frame_id,
         std::optional<py::array_t<uint8_t>> image1,
         std::optional<py::array_t<uint8_t>> image2) {
        // We expect a three-channel RGB image here
        if (!image1 && !image2) {
          py::gil_scoped_release release_gil;
          data_loader->add_frame(frame_id, {});
          return frame_id;
        }
        assert(image1->ndim() == 3);
        assert(image2->ndim() == 3);
        auto m1 = std::make_shared<hm::MatrixRGB>(*image1, 0, 0);
        auto m2 = std::make_shared<hm::MatrixRGB>(*image2, 0, 0);
        {
          py::gil_scoped_release release_gil;
          data_loader->add_frame(frame_id, {std::move(m1), std::move(m2)});
        }
        return frame_id;
      });

  m.def("_hello_world", []() {
    py::gil_scoped_release release_gil;
    std::cout << "Hello, world!" << std::endl;
  });

  m.def(
      "_enblend",
      [](std::string output_image,
         std::vector<std::string> input_files) -> int {
        py::gil_scoped_release release_gil;
        return hm::enblend::enblend_main(
            std::move(output_image), std::move(input_files));
      });

  py::class_<hm::HmNona, std::shared_ptr<hm::HmNona>>(m, "HmNona")
      .def(py::init<std::string>())
      .def("load_project", &hm::HmNona::load_project)
      .def("get_control_points", [](const std::shared_ptr<hm::HmNona>& nona) {
        auto results = nona->get_control_points();
        return results;
      });

  m.def(
      "_nona_process_images",
      [](std::shared_ptr<hm::HmNona> nona,
         py::array_t<uint8_t>& image1,
         py::array_t<uint8_t>& image2) -> std::vector<py::array_t<uint8_t>> {
        // We expect a three-channel RGB image here
        assert(image1.ndim() == 3);
        assert(image2.ndim() == 3);
        auto m1 = std::make_shared<hm::MatrixRGB>(image1, 0, 0);
        auto m2 = std::make_shared<hm::MatrixRGB>(image2, 0, 0);
        // Just remap (no blend)
        std::vector<py::array_t<uint8_t>> results;
        std::vector<std::unique_ptr<hm::MatrixRGB>> result_matrices;
        {
          py::gil_scoped_release release_gil;
          result_matrices = nona->remap_images(std::move(m1), std::move(m2));
        }
        results.reserve(result_matrices.size());
        for (auto& m : result_matrices) {
          if (m) {
            results.emplace_back(m->to_py_array());
          }
        }
        return results;
      });

  /**
   *   ____                         _
   *  / __ \                       | |
   * | |  | |_ __   ___  _ __  __ _| |_  ___  _ __  ___
   * | |  | | '_ \ / _ \| '__|/ _` | __|/ _ \| '__|/ __|
   * | |__| | |_) |  __/| |  | (_| | |_| (_) | |   \__ \
   *  \____/| .__/ \___||_|   \__,_|\__|\___/|_|   |___/
   *        | |
   *        |_|
   */
  py::class_<hm::ops::ImageRemapper, std::shared_ptr<hm::ops::ImageRemapper>>(
      m, "ImageRemapper")
      .def(
          py::init<
              std::size_t,
              std::size_t,
              at::Tensor,
              at::Tensor,
              bool,
              std::optional<std::string>>(),
          py::arg("src_width"),
          py::arg("src_height"),
          py::arg("col_map"),
          py::arg("row_map"),
          py::arg("add_alpha_channel"),
          py::arg("interpolation"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "init",
          &hm::ops::ImageRemapper::init,
          py::arg("batch_size"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "to",
          &hm::ops::ImageRemapper::to,
          py::arg("device"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "is_initialized",
          &hm::ops::ImageRemapper::is_initialized,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "forward",
          &hm::ops::ImageRemapper::forward,
          py::arg("source_tensor"),
          py::call_guard<py::gil_scoped_release>());

  py::enum_<hm::ops::ImageBlender::Mode>(m, "ImageBlenderMode")
      .value("HardSeam", hm::ops::ImageBlender::Mode::HardSeam)
      .value("Laplacian", hm::ops::ImageBlender::Mode::Laplacian)
      .export_values();

  py::class_<hm::ops::ImageBlender, std::shared_ptr<hm::ops::ImageBlender>>(
      m, "ImageBlender")
      .def(
          py::init<
              hm::ops::ImageBlender::Mode,
              std::size_t,
              at::Tensor,
              at::Tensor,
              std::optional<std::string>>(),
          py::arg("mode"),
          py::arg("levels"),
          py::arg("seam"),
          py::arg("xor_map"),
          py::arg("interpolation"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "init",
          &hm::ops::ImageBlender::init,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "to",
          &hm::ops::ImageBlender::to,
          py::arg("device"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "is_initialized",
          &hm::ops::ImageBlender::is_initialized,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "make_full",
          &hm::ops::ImageBlender::forward,
          py::arg("image_1"),
          py::arg("xy_pos_1"),
          py::arg("image_2"),
          py::arg("xy_pos_2"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "forward",
          &hm::ops::ImageBlender::forward,
          py::arg("image_1"),
          py::arg("xy_pos_1"),
          py::arg("image_2"),
          py::arg("xy_pos_2"),
          py::call_guard<py::gil_scoped_release>());
}
