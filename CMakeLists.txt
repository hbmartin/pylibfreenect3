cmake_minimum_required(VERSION 3.26...4.4)

project(pylibfreenect3 LANGUAGES CXX)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Cython 3.2.8 REQUIRED)
include(UseCython)
find_package(Freenect2 0.4 REQUIRED)

Cython_transpile(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/pylibfreenect3/_native.pyx"
  LANGUAGE CXX
  CYTHON_ARGS --3str --no-docstrings
  OUTPUT_VARIABLE native_source
)

Python_add_library(_native MODULE "${native_source}" WITH_SOABI)
target_compile_features(_native PRIVATE cxx_std_17)
target_compile_definitions(_native PRIVATE NPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION)
target_include_directories(_native PRIVATE "${Python_NumPy_INCLUDE_DIRS}")
target_link_libraries(_native PRIVATE Freenect2::Freenect2)

if(APPLE)
  set_target_properties(
    _native PROPERTIES
    BUILD_RPATH "${Freenect2_LIBRARY_DIR}"
    INSTALL_RPATH "${Freenect2_LIBRARY_DIR};@loader_path/.dylibs"
  )
elseif(UNIX)
  set_target_properties(
    _native PROPERTIES
    BUILD_RPATH "${Freenect2_LIBRARY_DIR}"
    INSTALL_RPATH "${Freenect2_LIBRARY_DIR};$ORIGIN/../pylibfreenect3.libs"
  )
endif()

install(TARGETS _native LIBRARY DESTINATION pylibfreenect3)
