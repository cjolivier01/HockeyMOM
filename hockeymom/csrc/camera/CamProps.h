#pragma once

#include <cstdint>
#include <list>
#include <unordered_map>
#include <unordered_set>

//#include <opencv2/opencv.hpp>

namespace hm {
namespace camera {

struct Point {
  float x{0};
  float y{0};
};

struct Box {
  float left{0};
  float top{0};
  float right{0};
  float bottom{0};
};

struct Tlwh : public Box {
};

//using TlwhHistory = std::list<Tlwh>;

struct TlwhHistory {
  std::size_t id;
  std::size_t video_
};

struct Venue {

};

struct Camera {
  std::string name;
  float elevation_m{3.0};
  float tilt_degrees{45.0};
  float roll_degrees{0.0};
  float focal_length{6.2};

};

struct CameraProperties {
  Box   video_frame;
  Box   clamp_box;
};

struct CameraBehavior {
  std::size_t max_history{26};
};

struct PlayerState {

};

struct CameraState {
  std::unordered_map<std::size_t, PlayerState> online_id_to_player_map;

  std::unordered_set<std::size_t> online_ids;
  std::unordered_map<std::size_t, TlwhHistory> online_tlwhs_history;
};

} // namespace camera
} // namespace hm
