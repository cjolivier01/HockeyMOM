#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <vector>

namespace hm {

class StitchingDataLoader {
  StitchingDataLoader(std::vector<std::string> video_files);
  ~StitchingDataLoader();
};

}

