#pragma once

#include <ATen/ATen.h>

#include "absl/status/status.h"

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>

#include <memory>
#include <string>


namespace hm {

namespace av {

using Size = std::array<int, 2>;

class AVFrameDeleter {
public:
    void operator()(AVFrame* frame) const {
        if (frame) {
            av_frame_free(&frame);
        }
    }
};

using unique_frame_ptr = std::unique_ptr<AVFrame, AVFrameDeleter>;
using shared_frame_ptr = std::shared_ptr<AVFrame>;

class FFmpegVideoWriter {
 public:
  // Constructor
  FFmpegVideoWriter();

  // Destructor
  ~FFmpegVideoWriter();

  // Open the video file for writing
  absl::Status open(
      const std::string& filename,
      const std::string& codec_name,
      double fps,
      Size frame_size,
      bool isColor = true);

  // Check if the video writer is open
  bool isOpened() const;

  // Write a frame
  absl::Status write(at::Tensor& tensor);

  void write_v(at::Tensor& tensor) {
    auto status = write(tensor);
    if (!status.ok()) {
      std::cerr << status << std::endl;
    }
  }

  void release() {
    freeResources();
  }

 private:
  std::string codec_name_;
  Size video_size_;
  // FFmpeg context and other necessary structures
  AVFormatContext* format_context_{nullptr};
  AVCodecContext* codec_context_{nullptr};
  AVStream* stream_{nullptr};
  // AVFrame* frame_{nullptr};
  AVPacket* pkt_{nullptr};
  SwsContext* sws_context_{nullptr};

  // Internal helper methods
  void close();
  int sendFrameToEncoder(AVFrame* frame);
  int writePacketToFile(AVPacket* pkt);
  bool initializeSwsContext(Size frame_size, bool isColor);
  void freeResources();
};

} // namespace av
} // namespace hm
