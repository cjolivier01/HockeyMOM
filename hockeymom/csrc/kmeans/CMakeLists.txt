set(SOURCES csrc/kmeans/kmeans.cpp
  csrc/kmeans/file_io.cpp
  csrc/kmeans/wtime.cpp
  csrc/kmeans/seq_kmeans.cpp
  csrc/kmeans/omp_kmeans.cpp
  csrc/kmeans/wtime.cpp)
set(HEADERS kmeans.h)

add_library(kmeans STATIC ${SOURCES} ${HEADERS})
