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
  std::unordered_set<size_t> all_indexes;
  all_indexes.reserve(tracking_boxes.size());
  size_t cluster_count = cluster_sizes_.size();

#pragma omp parallel for num_threads(cluster_count)
  for (size_t cluster_id = 0; cluster_id < cluster_count; ++cluster_id) {
    std::vector<size_t> item_indexes = get_largest_cluster_item_indexes(
        points,
        cluster_sizes_.at(cluster_id),
        /*dim=*/2);
#pragma omp critical // Ensure single-threaded execution for these iterations
    { all_indexes.insert(item_indexes.begin(), item_indexes.end()); }
  }

  std::vector<BBox> inlarge_bboxes;
  inlarge_bboxes.reserve(tracking_boxes.size());
  for (size_t idx : all_indexes) {
    inlarge_bboxes.emplace_back(tracking_boxes.at(idx));
  }
  result_box = get_union_bounding_box(inlarge_bboxes);

  return result_box;
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
  for (size_t i = 0, n = living_boxes_.size(); i < n; ++i) {
    living_boxes_[i]->set_bbox(bbox);
    BBox new_bbox = bbox.make_scaled(scale_step, scale_step);
    // We don't allow it to go an empty box (w==0 or h==0)
    if (!new_bbox.empty()) {
      bbox = new_bbox;
    }
  }
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
    }
  }

  play_detector_.forward(
      tick_count_, tracking_ids, tracking_boxes, ignore_tracking_ids);

  //
  // Compute the next box
  //
  BBox current_box = get_cluster_box(*p_cluster_bboxes);
  results.cluster_boxes.emplace_back(current_box);

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
