#include "hockeymom/csrc/video/video_writer.h"

extern "C"
{
#include <libavutil/frame.h>
}

namespace hm {
namespace av {
namespace {
AVRational make_rational(float value, bool reversed = false) {
  // Ten decimal places
  constexpr std::size_t kPlacesMover = 100000000;
  AVRational avr;
  avr.num = static_cast<int>(value * kPlacesMover);
  avr.den = kPlacesMover;
  if (reversed) {
    std::swap(avr.num, avr.den);
  }
  return avr;
}

unique_frame_ptr tensorToAVFrame(at::Tensor& tensor) {
  // Ensure tensor is in CPU memory and is contiguous
  tensor = tensor.to(at::kCPU).contiguous();

  // Create a new AVFrame
  unique_frame_ptr frame(av_frame_alloc());
  if (!frame) {
    // Handle error
    return nullptr;
  }

  // Set frame properties
  assert(tensor.size(0) == 3);
  frame->width = tensor.size(1); // Assuming tensor shape is [C, H, W]
  frame->height = tensor.size(2);
  frame->format = AV_PIX_FMT_RGB24; // Or another compatible format``

  // Allocate buffer for frame data
  if (av_frame_get_buffer(frame.get(), 0) < 0) {
    // Handle error
    return nullptr;
  }

  // Copy tensor data to AVFrame
  // This assumes the tensor is in the format compatible with AVFrame settings
  std::memcpy(
      frame->data[0], tensor.data_ptr(), tensor.numel() * sizeof(uint8_t));

  return frame;
}

struct PacketScope {
  PacketScope(AVPacket* p) : p_(p) {}
  ~PacketScope() {
    if (p_) {
      av_packet_unref(p_);
    }
  }

 private:
  AVPacket* p_;
};

} // namespace

FFmpegVideoWriter::FFmpegVideoWriter() {
  // Register all codecs and formats
}

FFmpegVideoWriter::~FFmpegVideoWriter() {
  freeResources();
}

absl::Status FFmpegVideoWriter::open(
    const std::string& filename,
    const std::string& codec_name,
    double fps,
    Size frame_size,
    bool isColor) {
  video_size_ = frame_size;

  // Register all formats and codecs
  avformat_network_init();

  // Find the HEVC NVENC encoder
  const AVCodec* codec = avcodec_find_encoder_by_name(codec_name.c_str());
  if (!codec) {
    return absl::Status(
        absl::StatusCode::kNotFound,
        std::string("Codec not found: ") + codec_name);
  }

  codec_context_ = avcodec_alloc_context3(codec);
  if (!codec_context_) {
    return absl::Status(
        absl::StatusCode::kInternal,
        std::string("Could not allocate video codec context for codec ") +
            codec_name);
  }

  // Set codec parameters
  codec_context_->bit_rate = 400000;
  codec_context_->width = frame_size[0];
  codec_context_->height = frame_size[1];
  codec_context_->time_base = make_rational(fps, /*reversed=*/true);
  codec_context_->framerate = make_rational(fps);
  codec_context_->gop_size = 10;
  codec_context_->max_b_frames = 1;
  codec_context_->pix_fmt = AV_PIX_FMT_CUDA;

  // Open the codec
  if (avcodec_open2(codec_context_, codec, nullptr) < 0) {
    return absl::Status(
        absl::StatusCode::kInternal,
        "Could not open codec with the given properties");
  }

  // Create and initialize a format context
  avformat_alloc_output_context2(
      &format_context_, nullptr, nullptr, filename.c_str());
  if (!format_context_) {
    return absl::Status(
        absl::StatusCode::kInternal,
        std::string("Could not create output context for file: ") + filename);
  }

  stream_ = avformat_new_stream(format_context_, nullptr);
  if (!stream_) {
    return absl::Status(
        absl::StatusCode::kInternal,
        std::string("Could not create video stream"));
  }

  stream_->time_base = codec_context_->time_base;
  stream_->codecpar->codec_id = codec->id;
  stream_->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
  stream_->codecpar->width = codec_context_->width;
  stream_->codecpar->height = codec_context_->height;
  stream_->codecpar->format = codec_context_->pix_fmt;

  // Write file header
  if (avformat_write_header(format_context_, nullptr) < 0) {
    return absl::Status(
        absl::StatusCode::kInternal,
        std::string("Error occurred when opening output file: ") + filename);
  }

  pkt_ = av_packet_alloc();

  return absl::OkStatus();
}

bool FFmpegVideoWriter::isOpened() const {
  return format_context_ != nullptr && codec_context_ != nullptr &&
      stream_ != nullptr;
}

absl::Status FFmpegVideoWriter::write(at::Tensor& tensor) {
  if (!isOpened()) {
    return absl::Status(
        absl::StatusCode::kUnavailable, "Video is not open for writing");
  }

  unique_frame_ptr frame = tensorToAVFrame(tensor);
  if (!frame) {
    return absl::Status(
        absl::StatusCode::kInternal,
        "Unable to convert tensor to a video frame");
  }
  if (frame->width != codec_context_->width ||
      frame->height != codec_context_->height) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        "Frame size does not match video size");
  }

  // Send the frame for encoding
  if (sendFrameToEncoder(frame.get()) < 0) {
    // Handle error
    return absl::Status(
        absl::StatusCode ::kInternal, "Error sending frame to encoder");
  }

  // Write the encoded packet to the file
  if (writePacketToFile(pkt_) < 0) {
    return absl::Status(
        absl::StatusCode ::kInternal, "Frame size does not match video size");
  }
  av_packet_unref(pkt_);
  return absl::OkStatus();
}

int FFmpegVideoWriter::sendFrameToEncoder(AVFrame* frame) {
  int ret = avcodec_send_frame(codec_context_, frame);
  if (ret < 0) {
    fprintf(stderr, "Error sending a frame for encoding\n");
    return ret;
  }
  return 0;
}

int FFmpegVideoWriter::writePacketToFile(AVPacket* pkt) {
  int ret = avcodec_receive_packet(codec_context_, pkt);
  if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
    return ret;
  } else if (ret < 0) {
    fprintf(stderr, "Error during encoding\n");
    return ret;
  }

  // Write the encoded packet
  av_packet_rescale_ts(
      pkt, codec_context_->time_base, format_context_->streams[0]->time_base);
  pkt->stream_index = 0;

  ret = av_interleaved_write_frame(format_context_, pkt);
  if (ret < 0) {
    fprintf(stderr, "Error while writing output packet\n");
    return ret;
  }

  return 0;
}

bool FFmpegVideoWriter::initializeSwsContext(Size frame_size, bool isColor) {
  // Initialize SwsContext and other settings
  // ...
  return true;
}

void FFmpegVideoWriter::close() {
  if (isOpened()) {
    sendFrameToEncoder(nullptr);
    writePacketToFile(pkt_);
  }
  if (format_context_) {
    // Write file trailer
    av_write_trailer(format_context_);
    // Clean up
    avformat_free_context(format_context_);
  }
  if (codec_context_) {
    avcodec_free_context(&codec_context_);
  }
  if (pkt_) {
    av_packet_free(&pkt_);
  }
  codec_context_ = nullptr;
  format_context_ = nullptr;
  stream_ = nullptr;
  pkt_ = nullptr;
}

void FFmpegVideoWriter::freeResources() {
  close();
}

} // namespace av
} // namespace hm
