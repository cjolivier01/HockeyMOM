#pragma once

//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "opencv2/videoio.hpp"

#include <string>
#include <vector>

namespace hm {

class StitchingDataLoader {
 public:
  StitchingDataLoader(std::vector<std::string> video_files);
  ~StitchingDataLoader();
 private:
};

}

