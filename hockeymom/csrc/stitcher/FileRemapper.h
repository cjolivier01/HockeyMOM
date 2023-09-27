#pragma once

#include "hockeymom/csrc/stitcher/HmRemappedPanoImage.h"

#include "nona/StitcherOptions.h"
#include "panodata/Panorama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hm {
/** functor to create a remapped image */
template <typename ImageType, typename AlphaType>
class HmSingleImageRemapper {
 public:
  HmSingleImageRemapper() : m_advancedOptions(){};

  /** create a remapped pano image.
   *
   *  The image ownership is transferred to the caller.
   */
  virtual std::unique_ptr<HmRemappedPanoImage<ImageType, AlphaType>> getRemapped(
      const HuginBase::PanoramaData& pano,
      const HuginBase::PanoramaOptions& opts,
      unsigned int imgNr,
      const std::shared_ptr<hm::MatrixRGB>& image,
      vigra::Rect2D outputROI,
      AppBase::ProgressDisplay* progress) = 0;

  virtual ~HmSingleImageRemapper(){};

  void setAdvancedOptions(
      const HuginBase::Nona::AdvancedOptions advancedOptions) {
    m_advancedOptions = advancedOptions;
  }

  ///
  //virtual void release(HmRemappedPanoImage<ImageType, AlphaType>* d) = 0;

 protected:
  HuginBase::Nona::AdvancedOptions m_advancedOptions;
  std::size_t pass_{0};
};

/** functor to create a remapped image, loads image from disk */
template <typename ImageType, typename AlphaType>
class HmFileRemapper : public HmSingleImageRemapper<ImageType, AlphaType> {
 public:
  HmFileRemapper() : HmSingleImageRemapper<ImageType, AlphaType>() {
    m_remapped = 0;
  }

  virtual ~HmFileRemapper(){};

  typedef std::vector<float> LUT;

 public:
  ///
  void loadImage(
      const HuginBase::PanoramaOptions& opts,
      vigra::ImageImportInfo& info,
      ImageType& srcImg,
      AlphaType& srcAlpha) {}

  ///
  std::unique_ptr<HmRemappedPanoImage<ImageType, AlphaType>> getRemapped(
      const HuginBase::PanoramaData& pano,
      const HuginBase::PanoramaOptions& opts,
      unsigned int imgNr,
      const std::shared_ptr<hm::MatrixRGB>& image,
      vigra::Rect2D outputROI,
      AppBase::ProgressDisplay* progress) override;

  ///
  // virtual void release(HmRemappedPanoImage<ImageType, AlphaType>* d) {
  //   // if (d == m_remapped) {
  //   //   m_remapped = null;ptr;
  //   // }
  //   // delete d;
  // }

 protected:
  AlphaType srcAlpha_;
  std::unique_ptr<HmRemappedPanoImage<ImageType, AlphaType>> m_remapped;
  std::vector<std::unique_ptr<vigra::ImageImportInfo>> image_import_infos_;
};

} // namespace hm