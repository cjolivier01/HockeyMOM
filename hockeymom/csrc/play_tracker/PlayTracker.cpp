#include "hockeymom/csrc/play_tracker/PlayTracker.h"
#include "hockeymom/csrc//kmeans/kmeans.h"
#include "hockeymom/csrc/play_tracker/LivingBoxImpl.h"

#include <cassert>
#include <stdexcept>
#include <unordered_set>

namespace hm {
namespace play_tracker {

namespace {

constexpr size_t kMinTracksBeforePruning = 3;
constexpr size_t kBadIdOrIndex = std::numeric_limits<size_t>::max();

std::tuple</*index_removed=*/size_t, std::vector<size_t>, std::vector<BBox>>
remove_largest(std::vector<size_t> ids, std::vector<BBox> bboxes) {
  double largest_area = 0.0;
  size_t largest_index = kBadIdOrIndex;
  for (size_t i = 0, n = ids.size(); i < n; ++i) {
    double this_area = bboxes[i].area();
    if (this_area > largest_area) {
      largest_area = this_area;
      largest_index = i;
    }
  }
  // Making a new vector will be more expensive than just removing a vector item
  // because to create a new vector, you are guaranteed to traverse the entire
  // vector except one item, plus construction and allocation overhead, etc.
  // So, simply erase the element.
  if (largest_index != kBadIdOrIndex) {
    ids.erase(ids.begin() + largest_index);
    bboxes.erase(bboxes.begin() + largest_index);
  }
  return std::make_tuple(largest_index, std::move(ids), std::move(bboxes));
}

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

BBox get_union_bounding_box(const std::vector<BBox>& boxes) {
  if (boxes.empty()) {
    return BBox();
  }

  float minTop = std::numeric_limits<float>::max();
  float minLeft = std::numeric_limits<float>::max();
  float maxBottom = std::numeric_limits<float>::min();
  float maxRight = std::numeric_limits<float>::min();

  for (const auto& box : boxes) {
    minTop = std::min(minTop, box.top);
    minLeft = std::min(minLeft, box.left);
    maxBottom = std::max(maxBottom, box.bottom);
    maxRight = std::max(maxRight, box.right);
  }

  return BBox{minTop, minLeft, maxBottom, maxRight};
}

} // namespace

PlayTracker::PlayTracker(
    const BBox& initial_box,
    const PlayTrackerConfig& config)
    : config_(config), play_detector_(PlayDetectorConfig(), this) {
  create_boxes(initial_box);
}

void PlayTracker::create_boxes(const BBox& initial_box) {
  for (std::size_t i = 0; i < config_.living_boxes.size(); ++i) {
    living_boxes_.emplace_back(std::make_unique<LivingBox>(
        std::to_string(i + 1), initial_box, config_.living_boxes[i]));
  }
}

PlayTracker::ClusterBoxes PlayTracker::get_cluster_boxes(
    const std::vector<BBox>& tracking_boxes) const {
  ClusterBoxes cluster_boxes_result;

  BBox result_box;
  size_t counter = 0;
  std::vector<float> points;
  points.reserve(tracking_boxes.size() << 1);

  for (const auto& box : tracking_boxes) {
    const Point c = box.center();
    points.push_back(c.x);
    points.push_back(c.y);
  }

  // Restrict toi cluster sizes that have some meaning based on how many
  // tracking objects we have (or are valid, for that matter)
  std::vector<size_t> cluster_sizes;
  cluster_sizes.reserve(cluster_sizes_.size());
  for (size_t cluster_size : cluster_sizes_) {
    // Equal or less clustering is meaningless
    if (cluster_size > tracking_boxes.size()) {
      cluster_sizes.emplace_back(cluster_size);
    }
  }

  size_t cluster_count = cluster_sizes.size();
  std::vector<std::vector<size_t>> cluster_item_indexes(cluster_count);
  std::vector<BBox> cluster_bboxes(cluster_count);

#pragma omp parallel for num_threads(cluster_count)
  for (size_t cluster_id = 0; cluster_id < cluster_count; ++cluster_id) {
    auto& this_cluster_item_indexes = cluster_item_indexes.at(cluster_id);
    this_cluster_item_indexes = get_largest_cluster_item_indexes(
        points,
        cluster_sizes.at(cluster_id),
        /*dim=*/2);
    std::vector<BBox> bboxes;
    bboxes.reserve(this_cluster_item_indexes.size());
    for (size_t idx : this_cluster_item_indexes) {
      bboxes.emplace_back(tracking_boxes.at(idx));
    }
    cluster_bboxes.at(cluster_id) = get_union_bounding_box(bboxes);
  }

  for (size_t i = 0; i < cluster_sizes.size(); ++i) {
    cluster_boxes_result.cluster_boxes[cluster_sizes[i]] = cluster_bboxes.at(i);
  }

  cluster_boxes_result.final_cluster_box =
      get_union_bounding_box(cluster_bboxes);

  return cluster_boxes_result;
}

void PlayTracker::set_bboxes(const std::vector<BBox>& bboxes) {
  if (bboxes.size() != 1 && bboxes.size() != living_boxes_.size()) {
    throw std::runtime_error(
        "Number of bounding boxes should be one or the same as the number of living boxes");
  }
  for (size_t i = 0, n = living_boxes_.size(); i < n; ++i) {
    living_boxes_[i]->set_bbox(bboxes.size() == 1 ? bboxes[0] : bboxes.at(i));
  }
}

void PlayTracker::set_bboxes_scaled(BBox bbox, float scale_step) {
  const BBox arena_box = get_play_box();
  for (size_t i = 0, n = living_boxes_.size(); i < n; ++i) {
    bbox = clamp_box(bbox, arena_box);
    living_boxes_[i]->set_bbox(bbox);
    BBox new_bbox = bbox.make_scaled(scale_step, scale_step);
    // We don't allow it to go an empty box (w==0 or h==0)
    if (!new_bbox.empty()) {
      bbox = new_bbox;
    }
  }
}

std::shared_ptr<ILivingBox> PlayTracker::get_live_box(size_t index) const {
  return living_boxes_.at(index);
}

BBox PlayTracker::get_play_box() const {
  auto arena_box = living_boxes_.at(living_boxes_.size() - 1)->get_arena_box();
  assert(arena_box.has_value());
  return *arena_box;
}

PlayTrackerResults PlayTracker::forward(
    std::vector<size_t>& tracking_ids,
    std::vector<BBox>& tracking_boxes) {
  assert(tracking_ids.size() == tracking_boxes.size());

  PlayTrackerResults results;

  std::set<size_t> ignore_tracking_ids;
  std::tuple</*index_removed=*/size_t, std::vector<size_t>, std::vector<BBox>>
      prune_results;
  std::vector<BBox>* p_cluster_bboxes = &tracking_boxes;
  if (config_.ignore_largest_bbox &&
      tracking_ids.size() > kMinTracksBeforePruning) {
    prune_results = remove_largest(tracking_ids, tracking_boxes);
    const size_t ignore_index = std::get<0>(prune_results);
    if (ignore_index != kBadIdOrIndex) {
      p_cluster_bboxes = &std::get<2>(prune_results);
      assert(p_cluster_bboxes->size() == tracking_boxes.size() - 1);
      ignore_tracking_ids.emplace(tracking_ids.at(ignore_index));
      results.largest_tracking_bbox_id = tracking_ids.at(ignore_index);
      results.largest_tracking_bbox = tracking_boxes.at(ignore_index);
    }
  }

  //
  // Compute the next box
  //
  const BBox arena_box = get_play_box();
  {
    ClusterBoxes cluster_boxes_result = get_cluster_boxes(*p_cluster_bboxes);
    results.cluster_boxes = std::move(cluster_boxes_result.cluster_boxes);
    results.final_cluster_box = cluster_boxes_result.final_cluster_box;
  }

  BBox start_bbox = results.final_cluster_box;
  // Special cases
  if (start_bbox.empty()) {
    // probably not enough tracks
    if (!tracking_boxes.empty()) {
      // Just union all tracking boxes
      start_bbox = get_union_bounding_box(tracking_boxes);
    }
  }
  clamp_box(start_bbox, arena_box);
  if (start_bbox.empty()) {
    start_bbox = arena_box;
  }

  if (!tick_count_ && config_.no_wide_start) {
    // We start at our first detected box
    set_bboxes_scaled(start_bbox, 1.2);
  }
  assert(!results.final_cluster_box.empty()); // TODO: need arena size
  BBox current_box = results.final_cluster_box;

  results.play_detection = play_detector_.forward(
      /*frame_id=*/tick_count_,
      /*current_target_bbox=*/current_box,
      /*current_roi_center=*/get_live_box(0)->bounding_box().center(),
      tracking_ids,
      tracking_boxes,
      ignore_tracking_ids);

  if (results.play_detection->breakaway_target_bbox.has_value()) {
    current_box = *results.play_detection->breakaway_target_bbox;
  }

  for (std::size_t i = 0; i < living_boxes_.size(); ++i) {
    auto& living_box = living_boxes_[i];
    current_box = living_box->forward(current_box);
    results.tracking_boxes.emplace_back(current_box);
  }
  ++tick_count_;
  return results;
}

} // namespace play_tracker
} // namespace hm
