#pragma once

#include "hockeymom/csrc/play_tracker/BoxUtils.h"
#include "hockeymom/csrc/play_tracker/LivingBox.h"
#include "hockeymom/csrc/play_tracker/ResizingBox.h"
#include "hockeymom/csrc/play_tracker/TranslatingBox.h"

namespace hm {
namespace play_tracker {

struct LivingState {
  bool was_size_constrained{false};
};

class LivingBox : public ILivingBox,
                  public BoundingBox,
                  public ResizingBox,
                  public TranslatingBox 
                  {
 public:
  LivingBox(std::string label, BBox bbox, const AllLivingBoxConfig& config);

  // -ILivingBox
  void set_destination(
      const std::variant<BBox, std::shared_ptr<IBasicLivingBox>>& dest)
      override;
  void set_dest(std::shared_ptr<IBasicLivingBox>) override {}
  void set_dest_ex(const std::variant<BBox, std::shared_ptr<IBasicLivingBox>>&
                       dest) override {}
  // ILivingBox-

  WHDims get_size_scale() const;

  // -IBasicLivingBox
  void set_bbox(const BBox& bbox) override;

  BBox bounding_box() const override;

  void set_destination(const BBox& dest_box) override;
  // IBasicLivingBox-

  BBox forward(const std::variant<BBox, std::shared_ptr<IBasicLivingBox>>& dest)
      override;

  const std::string& name() const override { return label_; }

  const LivingBoxConfig& config() const { return config_; }
  const LivingState& state() const { return state_; }

 protected:
  BBox next_position();

  void clamp_to_arena();

 private:
  const std::string label_;
  const LivingBoxConfig config_;

  // Flag to show we were size-constrained on the last update
  // (debugging/visualization only)
  LivingState state_;
  std::size_t forward_counter_{0};
};

} // namespace play_tracker
} // namespace hm
