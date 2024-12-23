set(SOURCES ResizingBox.cpp PlayerTrack.cpp TranslatingBox.cpp LivingBox.cpp LivingBoxImpl.cpp PlayTracker.cpp BoxUtils.cpp)
set(HEADERS LivingBox.h PlayerTrack.h LivingBoxImpl.h PlayTracker.h BoxUtils.h)

add_library(play_tracker STATIC ${SOURCES} ${HEADERS})
