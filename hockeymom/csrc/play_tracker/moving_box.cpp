#include <ATen/ATen.h>
#include <torch/torch.h>
//#include <opencv2/opencv.hpp>
#include <memory>
#include <iostream>
#include <vector>
#include <unordered_map>

//using namespace cv;
using namespace torch;

class BasicBox : public torch::nn::Module {
public:
    BasicBox(torch::Tensor bbox, c10::optional<torch::Device> device = c10::nullopt)
        : device_(device ? *device : bbox.device()), bbox_(bbox.clone()) {
        zero_float_tensor_ = torch::tensor(0.0, torch::dtype(torch::kFloat).device(device_));
        zero_int_tensor_ = torch::tensor(0, torch::dtype(torch::kInt64).device(device_));
        one_float_tensor_ = torch::tensor(1, torch::dtype(torch::kInt64).device(device_));
    }

    void set_bbox(torch::Tensor bbox) {
        bbox_ = bbox.clone();
    }

    torch::Tensor bounding_box() const {
        return bbox_.clone();
    }

    torch::Tensor one() const {
        return one_float_tensor_.clone();
    }

    torch::Tensor zero() const {
        return zero_float_tensor_.clone();
    }

    torch::Tensor zero_int() const {
        return zero_int_tensor_.clone();
    }

protected:
    torch::Device device_;
    torch::Tensor bbox_;
    torch::Tensor zero_float_tensor_, zero_int_tensor_, one_float_tensor_;
};

using DrawOptions = std::unordered_map<int, bool>;

class ResizingBox : public BasicBox {
public:
    ResizingBox(
        torch::Tensor bbox,
        double max_speed_w,
        double max_speed_h,
        double max_accel_w,
        double max_accel_h,
        double min_width,
        double min_height,
        double max_width,
        double max_height,
        bool stop_on_dir_change,
        bool sticky_sizing = false,
        c10::optional<torch::Device> device = c10::nullopt)
        : BasicBox(bbox, device),
          max_speed_w_(max_speed_w),
          max_speed_h_(max_speed_h),
          max_accel_w_(max_accel_w),
          max_accel_h_(max_accel_h),
          min_width_(min_width),
          min_height_(min_height),
          max_width_(max_width),
          max_height_(max_height),
          stop_on_dir_change_(stop_on_dir_change),
          sticky_sizing_(sticky_sizing) {
    }

    virtual void draw(at::Tensor& img, bool draw_thresholds = true) {
        // cv::Rect bbox_rect(
        //     static_cast<int>(bbox_[0].item<double>()),
        //     static_cast<int>(bbox_[1].item<double>()),
        //     static_cast<int>(bbox_[2].item<double>() - bbox_[0].item<double>()),
        //     static_cast<int>(bbox_[3].item<double>() - bbox_[1].item<double>())
        // );
        // cv::rectangle(img, bbox_rect, cv::Scalar(0, 255, 0), 2);
    }

    void set_destination(torch::Tensor dest_box) {
        set_destination_size(dest_box[2] - dest_box[0], dest_box[3] - dest_box[1]);
    }

    void set_destination_size(torch::Tensor dest_width, torch::Tensor dest_height) {
        torch::Tensor current_w = bbox_[2] - bbox_[0];
        torch::Tensor current_h = bbox_[3] - bbox_[1];

        torch::Tensor dw = dest_width - current_w;
        torch::Tensor dh = dest_height - current_h;

        adjust_size(dw, dh);
    }

    void adjust_size(torch::optional<torch::Tensor> accel_w = c10::nullopt, torch::optional<torch::Tensor> accel_h = c10::nullopt, bool use_constraints = true) {
        if (accel_w.has_value()) {
            current_speed_w_ += *accel_w;
        }

        if (accel_h.has_value()) {
            current_speed_h_ += *accel_h;
        }

        if (use_constraints) {
            clamp_resizing();
        }
    }

    void clamp_resizing() {
        current_speed_w_ = torch::clamp(current_speed_w_, -max_speed_w_, max_speed_w_);
        current_speed_h_ = torch::clamp(current_speed_h_, -max_speed_h_, max_speed_h_);
    }

private:
    double max_speed_w_, max_speed_h_;
    double max_accel_w_, max_accel_h_;
    double min_width_, min_height_;
    double max_width_, max_height_;
    bool stop_on_dir_change_;
    bool sticky_sizing_;
    torch::Tensor current_speed_w_ = torch::tensor(0.0);
    torch::Tensor current_speed_h_ = torch::tensor(0.0);
};

class MovingBox : public ResizingBox {
public:
    MovingBox(
        std::string label,
        torch::Tensor bbox,
        double max_speed_x,
        double max_speed_y,
        double max_accel_x,
        double max_accel_y,
        double max_width,
        double max_height,
        bool stop_on_dir_change,
        int min_width = 10,
        int min_height = 10,
        c10::optional<torch::Device> device = c10::nullopt)
        : ResizingBox(bbox, max_speed_x, max_speed_y, max_accel_x, max_accel_y, min_width, min_height, max_width, max_height, stop_on_dir_change, false, device),
          label_(label) {
    }

    void draw(at::Tensor& img, bool draw_thresholds = false) override {
        ResizingBox::draw(img, draw_thresholds);
        //putText(img, label_, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
    }

    void set_destination(torch::Tensor dest_box) {
        set_destination_size(dest_box[2] - dest_box[0], dest_box[3] - dest_box[1]);
    }

    torch::Tensor get_zoom_level() {
        torch::Tensor current_w = bbox_[2] - bbox_[0];
        torch::Tensor current_h = bbox_[3] - bbox_[1];
        return torch::sqrt(current_w * current_w + current_h * current_h);
    }

    void adjust_speed(torch::optional<torch::Tensor> accel_x = c10::nullopt, torch::optional<torch::Tensor> accel_y = c10::nullopt, torch::optional<double> scale_constraints = c10::nullopt, torch::optional<torch::Tensor> nonstop_delay = c10::nullopt) {
        if (accel_x.has_value()) {
            current_speed_x_ += *accel_x;
        }

        if (accel_y.has_value()) {
            current_speed_y_ += *accel_y;
        }

        if (scale_constraints.has_value()) {
            clamp_speed(*scale_constraints);
        }
    }

    void scale_speed(torch::optional<torch::Tensor> ratio_x = c10::nullopt, torch::optional<torch::Tensor> ratio_y = c10::nullopt, bool clamp_to_max = false) {
        if (ratio_x.has_value()) {
            current_speed_x_ *= *ratio_x;
        }

        if (ratio_y.has_value()) {
            current_speed_y_ *= *ratio_y;
        }

        if (clamp_to_max) {
            clamp_speed(1.0);
        }
    }

    torch::Tensor next_position() {
        torch::Tensor dx = current_speed_x_;
        torch::Tensor dy = current_speed_y_;

        // bbox_ += torch::tensor(std::vector<at::Tensor>{dx, dy, dx, dy}, torch::dtype(torch::kFloat).device(device_));

        clamp_to_arena();
        return bbox_;
    }

private:
    void clamp_speed(double scale) {
        current_speed_x_ = torch::clamp(current_speed_x_, -max_speed_x_ * scale, max_speed_x_ * scale);
        current_speed_y_ = torch::clamp(current_speed_y_, -max_speed_y_ * scale, max_speed_y_ * scale);
    }

    void clamp_to_arena() {
        bbox_[0] = torch::clamp(bbox_[0], 0.0, max_width_);
        bbox_[1] = torch::clamp(bbox_[1], 0.0, max_height_);
        bbox_[2] = torch::clamp(bbox_[2], 0.0, max_width_);
        bbox_[3] = torch::clamp(bbox_[3], 0.0, max_height_);
    }

    std::string label_;
    torch::Tensor current_speed_x_ = torch::tensor(0.0);
    torch::Tensor current_speed_y_ = torch::tensor(0.0);
};

#if 0
int main() {
    // Example usage
    torch::Tensor bbox = torch::tensor({10, 10, 100, 100}, torch::kFloat);
    auto device = torch::kCPU;
    MovingBox moving_box("Example Box", bbox, 5.0, 5.0, 1.0, 1.0, 200.0, 200.0, true);

    at::Tensor img = at::Tensor::zeros(500, 500, CV_8UC3);
    moving_box.draw(img);

    cv::imshow("Moving Box", img);
    cv::waitKey(0);

    return 0;
}
#endif
