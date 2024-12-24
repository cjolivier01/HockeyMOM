set(SOURCES 
  ResizingBox.cpp
  PlayerTrack.cpp
  PlayTracker.cpp
  PlayDetector.cpp 
  TranslatingBox.cpp
  LivingBox.cpp
  LivingBoxImpl.cpp
  BoxUtils.cpp
)
set(HEADERS
  LivingBox.h
  PlayerTrack.h
  PlayTracker.h
  PlayDetector.h
  LivingBoxImpl.h 
  BoxUtils.h
)

add_library(play_tracker STATIC ${SOURCES} ${HEADERS})
