set(SOURCES LivingBox.cpp PlayTracker.cpp BoxUtils.cpp)
set(HEADERS LivingBox.h PlayTracker.h BoxUtils.h)

add_library(play_tracker STATIC ${SOURCES} ${HEADERS})
