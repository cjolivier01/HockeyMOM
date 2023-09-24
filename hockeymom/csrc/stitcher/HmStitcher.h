#pragma once

#include "hockeymom/csrc/common/MatrixRGB.h"
#include "hockeymom/csrc/stitcher/HmRemappedPanoImage.h"

#include "hugin/src/hugin_base/nona/Stitcher.h"
#include "hugin/src/hugin_base/nona/StitcherOptions.h"

#include <cstdint>
#include <string>

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
  virtual HmRemappedPanoImage<ImageType, AlphaType>* getRemapped(
      const HuginBase::PanoramaData& pano,
      const HuginBase::PanoramaOptions& opts,
      unsigned int imgNr,
      vigra::Rect2D outputROI,
      AppBase::ProgressDisplay* progress) = 0;

  virtual ~HmSingleImageRemapper(){};

  void setAdvancedOptions(
      const HuginBase::Nona::AdvancedOptions advancedOptions) {
    m_advancedOptions = advancedOptions;
  }

  ///
  virtual void release(HmRemappedPanoImage<ImageType, AlphaType>* d) = 0;

 protected:
  HuginBase::Nona::AdvancedOptions m_advancedOptions;
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
  virtual HmRemappedPanoImage<ImageType, AlphaType>* getRemapped(
      const HuginBase::PanoramaData& pano,
      const HuginBase::PanoramaOptions& opts,
      unsigned int imgNr,
      vigra::Rect2D outputROI,
      AppBase::ProgressDisplay* progress);

  ///
  virtual void release(HmRemappedPanoImage<ImageType, AlphaType>* d) {
    delete d;
  }

 protected:
  HmRemappedPanoImage<ImageType, AlphaType>* m_remapped;
};

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
  using PanoramaData = HuginBase::PanoramaData;
  using PanoramaOptions = HuginBase::PanoramaOptions;
  using AdvancedOptions = HuginBase::Nona::AdvancedOptions;
  using UIntSet = HuginBase::UIntSet;
  using Base::m_basename;

  HmMultiImageRemapper(
      const PanoramaData& pano,
      AppBase::ProgressDisplay* progress)
      : MultiImageRemapper(pano, progress) {}

  void set_images(
      std::shared_ptr<hm::MatrixRGB> image1,
      std::shared_ptr<hm::MatrixRGB> image2) {
    images_ = std::vector<std::shared_ptr<hm::MatrixRGB>>{
        std::move(image1), std::move(image2)};
  }

  void stitch(
      const PanoramaOptions& opts,
      UIntSet& images,
      const std::string& basename,
      HmSingleImageRemapper<ImageType, AlphaType>& remapper,
      const AdvancedOptions& advOptions) {
    ++pass_;
    // Skip over direct base class stitch call
    if (pass_ == 1) {
      // Base::Base::stitch(opts, images, basename, remapper);
      Base::Base::m_images = images;
      Base::calcOutputROIS(opts, images);
    }
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
      if (pass_ == 1) {
        mod_options_.clear();
        PanoramaOptions modOptions(opts);
        if (HuginBase::Nona::GetAdvancedOption(
                advOptions, "ignoreExposure", false)) {
          modOptions.outputExposureValue =
              Base::m_pano.getImage(*it).getExposureValue();
          modOptions.outputRangeCompression = 0.0;
        };
        mod_options_.emplace_back(std::move(modOptions));
      }
      HmRemappedPanoImage<ImageType, AlphaType>* remapped =
          remapper.getRemapped(
              Base::m_pano,
              mod_options_.at(i),
              *it,
              Base::m_rois[i],
              Base::m_progress);
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
 private:
  std::vector<std::shared_ptr<hm::MatrixRGB>> images_;
  std::vector<PanoramaOptions> mod_options_;
  std::size_t pass_{0};
};

template <typename ImageType, typename AlphaType>
HmRemappedPanoImage<ImageType, AlphaType>* HmFileRemapper<
    ImageType,
    AlphaType>::
    getRemapped(
        const HuginBase::PanoramaData& pano,
        const HuginBase::PanoramaOptions& opts,
        unsigned int imgNr,
        vigra::Rect2D outputROI,
        AppBase::ProgressDisplay* progress) {
  typedef typename ImageType::value_type PixelType;

  // typedef typename vigra::NumericTraits<PixelType>::RealPromote RPixelType;
  //         typedef typename vigra::BasicImage<RPixelType> RImportImageType;
  typedef typename vigra::BasicImage<float> FlatImgType;

  FlatImgType ffImg;
  AlphaType srcAlpha;

  // choose image type...
  const HuginBase::SrcPanoImage& img = pano.getImage(imgNr);

  vigra::Size2D destSize(opts.getWidth(), opts.getHeight());

  m_remapped = new HmRemappedPanoImage<ImageType, AlphaType>;

  // load image

  vigra::ImageImportInfo info(img.getFilename().c_str());

  int width = info.width();
  int height = info.height();

  if (opts.remapUsingGPU) {
    // Extend image width to multiple of 8 for fast GPU transfers.
    const int r = width % 8;
    if (r != 0)
      width += 8 - r;
  }

  ImageType srcImg(width, height);
  m_remapped->m_ICCProfile = info.getICCProfile();

  if (info.numExtraBands() > 0) {
    srcAlpha.resize(width, height);
  }
  // int nb = info.numBands() - info.numExtraBands();
  bool alpha = info.numExtraBands() > 0;
  std::string type = info.getPixelType();

  HuginBase::SrcPanoImage src = pano.getSrcImage(imgNr);

  // import the image
  progress->setMessage("loading", hugin_utils::stripPath(img.getFilename()));

  if (alpha) {
    vigra::importImageAlpha(
        info, vigra::destImage(srcImg), vigra::destImage(srcAlpha));
  } else {
    vigra::importImage(info, vigra::destImage(srcImg));
  }
  // check if the image needs to be scaled to 0 .. 1,
  // this only works for int -> float, since the image
  // has already been loaded into the output container
  double maxv = vigra_ext::getMaxValForPixelType(info.getPixelType());
  if (maxv != vigra_ext::LUTTraits<PixelType>::max()) {
    double scale = ((double)vigra_ext::LUTTraits<PixelType>::max()) / maxv;
    // std::cout << "Scaling input image (pixel type: " << info.getPixelType()
    // << " with: " << scale << std::endl;
    transformImage(
        vigra::srcImageRange(srcImg),
        destImage(srcImg),
        vigra::functor::Arg1() * vigra::functor::Param(scale));
  }

  // load flatfield, if needed.
  if (img.getVigCorrMode() & HuginBase::SrcPanoImage::VIGCORR_FLATFIELD) {
    // load flatfield image.
    vigra::ImageImportInfo ffInfo(img.getFlatfieldFilename().c_str());
    progress->setMessage(
        "flatfield vignetting correction",
        hugin_utils::stripPath(img.getFilename()));
    vigra_precondition(
        (ffInfo.numBands() == 1),
        "flatfield vignetting correction: "
        "Only single channel flatfield images are supported\n");
    ffImg.resize(ffInfo.width(), ffInfo.height());
    vigra::importImage(ffInfo, vigra::destImage(ffImg));
  }
  m_remapped->setAdvancedOptions(
      HmSingleImageRemapper<ImageType, AlphaType>::m_advancedOptions);
  // remap the image

  remapImage(
      srcImg,
      srcAlpha,
      ffImg,
      pano.getSrcImage(imgNr),
      opts,
      outputROI,
      *m_remapped,
      progress);
  return m_remapped;
}

} // namespace hm
