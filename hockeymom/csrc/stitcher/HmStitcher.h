#pragma once

#include "hockeymom/csrc/stitcher/HmRemappedPanoImage.h"

//#include "panodata/Panorama.h"
#include "hugin/src/hugin_base/nona/Stitcher.h"
#include "hugin/src/hugin_base/nona/StitcherOptions.h"

#include <cstdint>
#include <string>

namespace hm {

/** remap a set of images, and store the individual remapped files. */
template <typename ImageType, typename AlphaType>
class HmMultiImageRemapper
    : public HuginBase::Nona::MultiImageRemapper<ImageType, AlphaType> {
  // using namespace HuginBase;
  // using namespace HuginBase::Nona;
 public:
  using Base = HuginBase::Nona::MultiImageRemapper<ImageType, AlphaType>;
  using MultiImageRemapper =
      HuginBase::Nona::MultiImageRemapper<ImageType, AlphaType>;
  using SingleImageRemapper =
      HuginBase::Nona::SingleImageRemapper<ImageType, AlphaType>;
  using PanoramaData = HuginBase::PanoramaData;
  using PanoramaOptions = HuginBase::PanoramaOptions;
  using AdvancedOptions = HuginBase::Nona::AdvancedOptions;
  using UIntSet = HuginBase::UIntSet;
  using Base::m_basename;

  HmMultiImageRemapper(
      const PanoramaData& pano,
      AppBase::ProgressDisplay* progress)
      : MultiImageRemapper(pano, progress) {}

  // virtual ~MultiImageRemapper()
  // {
  // }

  virtual void stitch(
      const PanoramaOptions& opts,
      UIntSet& images,
      const std::string& basename,
      SingleImageRemapper& remapper,
      const AdvancedOptions& advOptions) {
    Base::stitch(opts, images, basename, remapper);
    DEBUG_ASSERT(
        opts.outputFormat == PanoramaOptions::TIFF_multilayer ||
        opts.outputFormat == PanoramaOptions::TIFF_m ||
        opts.outputFormat == PanoramaOptions::JPEG_m ||
        opts.outputFormat == PanoramaOptions::PNG_m ||
        opts.outputFormat == PanoramaOptions::HDR_m ||
        opts.outputFormat == PanoramaOptions::EXR_m);

    m_basename = basename;

    // setup the output.
    prepareOutputFile(opts, advOptions);

    // remap each image and save
    int i = 0;
    for (UIntSet::const_iterator it = images.begin(); it != images.end();
         ++it) {
      // get a remapped image.
      PanoramaOptions modOptions(opts);
      if (HuginBase::Nona::GetAdvancedOption(
              advOptions, "ignoreExposure", false)) {
        modOptions.outputExposureValue =
            Base::m_pano.getImage(*it).getExposureValue();
        modOptions.outputRangeCompression = 0.0;
      };
      HmRemappedPanoImage<ImageType, AlphaType>* remapped =
          remapper.getRemapped(
              Base::m_pano, modOptions, *it, Base::m_rois[i], Base::m_progress);
      try {
        saveRemapped(
            *remapped, *it, Base::m_pano.getNrOfImages(), opts, advOptions);
      } catch (vigra::PreconditionViolation& e) {
        // this can be thrown, if an image
        // is completely out of the pano
        std::cerr << e.what();
      }
      // free remapped image
      remapper.release(remapped);
      i++;
    }
    finalizeOutputFile(opts);
    if (Base::m_progress) {
      Base::m_progress->taskFinished();
    }
  }

  /** prepare the output file (setup file structures etc.) */
  virtual void prepareOutputFile(
      const PanoramaOptions& opts,
      const AdvancedOptions& advOptions) {
    // Base::m_progress->setMessage("Multiple images output");
  }

  /** save a remapped image, or layer */
  virtual void saveRemapped(
      HmRemappedPanoImage<ImageType, AlphaType>& remapped,
      unsigned int imgNr,
      unsigned int nImg,
      const PanoramaOptions& opts,
      const AdvancedOptions& advOptions) {
    HuginBase::Nona::detail::saveRemapped(
        remapped,
        imgNr,
        nImg,
        opts,
        m_basename,
        HuginBase::Nona::GetAdvancedOption(advOptions, "useBigTIFF", false),
        Base::m_progress);

    if (opts.saveCoordImgs) {
      vigra::UInt16Image xImg;
      vigra::UInt16Image yImg;

      Base::m_progress->setMessage("creating coordinate images");

      remapped.calcSrcCoordImgs(xImg, yImg);
      vigra::UInt16Image dist;
      if (!opts.tiff_saveROI) {
        dist.resize(opts.getWidth(), opts.getHeight());
        vigra::copyImage(
            srcImageRange(xImg),
            vigra_ext::applyRect(remapped.boundingBox(), destImage(dist)));
        std::ostringstream filename2;
        filename2 << m_basename << std::setfill('0') << std::setw(4) << imgNr
                  << "_x.tif";

        vigra::ImageExportInfo exinfo(filename2.str().c_str());
        exinfo.setXResolution(150);
        exinfo.setYResolution(150);
        exinfo.setCompression(opts.tiffCompression.c_str());
        vigra::exportImage(srcImageRange(dist), exinfo);

        vigra::copyImage(
            srcImageRange(yImg),
            vigra_ext::applyRect(remapped.boundingBox(), destImage(dist)));
        std::ostringstream filename3;
        filename3 << m_basename << std::setfill('0') << std::setw(4) << imgNr
                  << "_y.tif";

        vigra::ImageExportInfo exinfo2(filename3.str().c_str());
        exinfo2.setXResolution(150);
        exinfo2.setYResolution(150);
        exinfo2.setCompression(opts.tiffCompression.c_str());
        vigra::exportImage(srcImageRange(dist), exinfo2);
      } else {
        std::ostringstream filename2;
        filename2 << m_basename << std::setfill('0') << std::setw(4) << imgNr
                  << "_x.tif";
        vigra::ImageExportInfo exinfo(filename2.str().c_str());
        exinfo.setXResolution(150);
        exinfo.setYResolution(150);
        exinfo.setCompression(opts.tiffCompression.c_str());
        vigra::exportImage(srcImageRange(xImg), exinfo);
        std::ostringstream filename3;
        filename3 << m_basename << std::setfill('0') << std::setw(4) << imgNr
                  << "_y.tif";
        vigra::ImageExportInfo exinfo2(filename3.str().c_str());
        exinfo2.setXResolution(150);
        exinfo2.setYResolution(150);
        exinfo2.setCompression(opts.tiffCompression.c_str());
        vigra::exportImage(srcImageRange(yImg), exinfo2);
      }
    }
  }

  virtual void finalizeOutputFile(const PanoramaOptions& opts) {
    Base::m_progress->taskFinished();
  }

  // protected:
  //     std::string m_basename;
};

} // namespace hm
