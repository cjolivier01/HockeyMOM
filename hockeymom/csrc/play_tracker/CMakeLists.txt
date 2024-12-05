set(SOURCES LivingBox.cpp PlayTracker.cpp)
set(HEADERS LivingBox.h PlayTracker.h)

add_library(play_tracker STATIC ${SOURCES} ${HEADERS})
