#pragma once

#include <ATen/ATen.h>

#include "absl/status/status.h"

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>

#include <string>

namespace hm {

struct Size {
  int width{0};
  int height{0};
};

struct Frame {
  void *data{nullptr};
  std::size_t size{0};
};
class FFmpegVideoWriter {
public:
    // Constructor
    FFmpegVideoWriter();

    // Destructor
    ~FFmpegVideoWriter();

    // Open the video file for writing
    absl::Status open(const std::string &filename, const std::string& codec_name, double fps, Size frame_size, bool isColor=true);

    // Check if the video writer is open
    bool isOpened() const;

    // Write a frame
    absl::Status write(at::Tensor& tensor);

private:
    std::string codec_name_;
    Size video_size_;
    // FFmpeg context and other necessary structures
    AVFormatContext* format_context_{nullptr};
    AVCodecContext* codec_context_{nullptr};
    AVStream* stream_{nullptr};
    //AVFrame* frame_{nullptr};
    AVPacket* pkt_{nullptr};
    SwsContext* sws_context_{nullptr};

    // Internal helper methods
    void flush();
    int sendFrameToEncoder(AVFrame* frame);
    int writePacketToFile(AVPacket* pkt);
    bool initializeSwsContext(Size frame_size, bool isColor);
    void freeResources();
};


}
