set(SOURCES MovingBox.cpp PlayTracker.cpp)
set(HEADERS MovingBox.h PlayTracker.h)

add_library(play_tracker STATIC ${SOURCES} ${HEADERS})
