#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "caffe" for configuration "Release"
set_property(TARGET caffe APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(caffe PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "proto;proto;/usr/lib/libboost_system-mt.so;/usr/lib/libboost_thread-mt.so;-lpthread;/usr/local/lib/libglog.so;/usr/local/lib/libgflags.a;/usr/lib/libprotobuf.so;-lpthread;/usr/lib/libhdf5_hl.so;/usr/lib/libhdf5.so;/usr/local/lib/liblmdb.so;/usr/lib/libleveldb.a;/usr/lib/libsnappy.so;/usr/local/cuda-7.0/lib64/libcudart.so;/usr/local/cuda-7.0/lib64/libcurand.so;/usr/local/cuda-7.0/lib64/libcublas.so;opencv_core;opencv_highgui;opencv_imgproc;opencv_imgcodecs;/usr/lib/liblapack_atlas.so;/usr/lib/atlas-base/libptcblas.a;/usr/lib/libatlas.so"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffe.so"
  IMPORTED_SONAME_RELEASE "libcaffe.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffe )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffe "${_IMPORT_PREFIX}/lib/libcaffe.so" )

# Import target "proto" for configuration "Release"
set_property(TARGET proto APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(proto PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libproto.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS proto )
list(APPEND _IMPORT_CHECK_FILES_FOR_proto "${_IMPORT_PREFIX}/lib/libproto.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
