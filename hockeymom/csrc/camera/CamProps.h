#pragma once

#include <cstdint>
#include <list>
#include <unordered_map>
#include <unordered_set>

namespace hm {
namespace camera {
struct Box {
  int left{0};
  int top{0};
  int right{0};
  int bottom{0};
};

struct Tlwh : public Box {
};

using TlwhHistory = std::list<Tlwh>;

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
