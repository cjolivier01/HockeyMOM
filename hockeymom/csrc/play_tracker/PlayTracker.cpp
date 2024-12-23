#include "hockeymom/csrc/play_tracker/PlayTracker.h"
#include "hockeymom/csrc//kmeans/kmeans.h"
#include "hockeymom/csrc/play_tracker/LivingBoxImpl.h"

namespace hm {
namespace play_tracker {

namespace {
std::vector<size_t> get_largest_cluster_item_indexes(
    const std::vector<float> points,
    size_t num_clusters,
    int dim) {
  std::size_t item_count = points.size() / dim;
  std::vector<int> assignments;
  assignments.reserve(item_count);
  hm::kmeans::compute_kmeans(
      points,
      num_clusters,
      /*dim=*/2,
      /*numIterations=*/6,
      assignments,
      hm::kmeans::KMEANS_TYPE::KM_OMP);
  std::vector<std::vector<size_t>> cluster_to_indexes(num_clusters);
  std::for_each(
      cluster_to_indexes.begin(),
      cluster_to_indexes.end(),
      [item_count](auto& v) { v.reserve(item_count); });
  std::vector<size_t> frequency(num_clusters, 0);
  for (size_t item_index = 0; item_index < item_count; ++item_index) {
    int cluster_nr = assignments.at(item_index);
    cluster_to_indexes[cluster_nr].push_back(item_index);
    ++frequency[cluster_nr];
  }
  return cluster_to_indexes.at(std::distance(
      frequency.begin(), std::max_element(frequency.begin(), frequency.end())));
}
} // namespace

PlayTracker::PlayTracker(const PlayTrackerConfig& config) : config_(config) {
  create_boxes();
}

void PlayTracker::create_boxes() {
  for (std::size_t i = 0; i < config_.living_boxes.size(); ++i) {
    living_boxes_.emplace_back(std::make_unique<LivingBox>(
        std::to_string(i + 1), BBox(), config_.living_boxes[i]));
  }
}

BBox PlayTracker::get_cluster_box(
    const std::vector<BBox>& tracking_boxes) const {
  BBox result_box;
  size_t counter = 0;
  std::vector<float> points;
  points.reserve(tracking_boxes.size() << 1);

  for (const auto& box : tracking_boxes) {
    const Point c = box.center();
    points.push_back(c.x);
    points.push_back(c.y);
  }
  std::vector<int> assignments;
  size_t cluster_count = cluster_sizes_.size();
#pragma omp parallel for num_threads(cluster_count)
  for (size_t cluster_id = 0; cluster_id < cluster_count; ++cluster_id) {
    assignments.clear();
    hm::kmeans::compute_kmeans(
        points,
        cluster_sizes_[cluster_id],
        /*dim=*/2,
        /*numIterations=*/6,
        assignments,
        hm::kmeans::KMEANS_TYPE::KM_OMP);
  }
  // std::unordered_map<size_t,
  return result_box;
}

PlayTrackerResults PlayTracker::forward(
    std::vector<size_t>& tracking_ids,
    std::vector<BBox>& tracking_boxes) {
  assert(tracking_ids.size() == living_boxes_.size());

  PlayTrackerResults results;
  for (std::size_t i = 0; i < living_boxes_.size(); ++i) {
    auto& living_box = living_boxes_[i];

    // auto& player_track = living_box->player_track();
    // if (player_track.age(tick_count_) > config_.max_lost_track_age) {
    //   player_track.reset();
    // }
    // if (tracking_ids[i] != 0) {
    //   player_track.add_position(tick_count_, tracking_boxes[i].center());
    // }
  }
  ++tick_count_;
  return results;
}

} // namespace play_tracker
} // namespace hm
