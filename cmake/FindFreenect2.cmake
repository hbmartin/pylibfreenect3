include(FindPackageHandleStandardArgs)

set(_Freenect2_saved_prefix_path "${CMAKE_PREFIX_PATH}")
if(Freenect2_ROOT)
  list(PREPEND CMAKE_PREFIX_PATH "${Freenect2_ROOT}")
  set(freenect2_ROOT "${Freenect2_ROOT}")
endif()

find_package(freenect2 CONFIG QUIET)

if(TARGET freenect2::freenect2)
  set(Freenect2_DISCOVERY "CMake package")
  get_target_property(Freenect2_INCLUDE_DIR freenect2::freenect2 INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(Freenect2_LIBRARY freenect2::freenect2 IMPORTED_LOCATION)
  add_library(Freenect2::Freenect2 INTERFACE IMPORTED)
  set_property(TARGET Freenect2::Freenect2 PROPERTY INTERFACE_LINK_LIBRARIES freenect2::freenect2)
else()
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PC_Freenect2 QUIET freenect2)
  endif()

  find_path(
    Freenect2_INCLUDE_DIR
    NAMES libfreenect2/libfreenect2.hpp
    HINTS "${Freenect2_ROOT}" "${PC_Freenect2_INCLUDEDIR}"
    PATH_SUFFIXES include
  )
  find_library(
    Freenect2_LIBRARY
    NAMES freenect2 freenect2static
    HINTS "${Freenect2_ROOT}" "${PC_Freenect2_LIBDIR}"
    PATH_SUFFIXES lib lib64
  )

  if(PC_Freenect2_FOUND)
    set(Freenect2_DISCOVERY "pkg-config")
  elseif(Freenect2_ROOT)
    set(Freenect2_DISCOVERY "Freenect2_ROOT")
  else()
    set(Freenect2_DISCOVERY "standard prefixes")
  endif()

  if(Freenect2_INCLUDE_DIR AND Freenect2_LIBRARY)
    add_library(Freenect2::Freenect2 UNKNOWN IMPORTED)
    set_target_properties(
      Freenect2::Freenect2 PROPERTIES
      IMPORTED_LOCATION "${Freenect2_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${Freenect2_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${PC_Freenect2_LINK_LIBRARIES}"
    )
  endif()
endif()

set(CMAKE_PREFIX_PATH "${_Freenect2_saved_prefix_path}")

if(Freenect2_INCLUDE_DIR)
  set(_Freenect2_config "${Freenect2_INCLUDE_DIR}/libfreenect2/config.h")
  if(EXISTS "${_Freenect2_config}")
    file(STRINGS "${_Freenect2_config}" _Freenect2_version_line REGEX "^#define LIBFREENECT2_VERSION ")
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" Freenect2_VERSION "${_Freenect2_version_line}")
  endif()
endif()

find_package_handle_standard_args(
  Freenect2
  REQUIRED_VARS Freenect2_INCLUDE_DIR Freenect2_LIBRARY Freenect2_VERSION
  VERSION_VAR Freenect2_VERSION
  REASON_FAILURE_MESSAGE
    "Install libfreenect2-metal 0.3.x or set Freenect2_ROOT to its installation prefix."
)

if(Freenect2_FOUND)
  if(NOT Freenect2_VERSION MATCHES "^0\\.3\\.")
    message(FATAL_ERROR "libfreenect2 0.3.x headers are required; found '${Freenect2_VERSION}'")
  endif()

  get_filename_component(Freenect2_LIBRARY_DIR "${Freenect2_LIBRARY}" DIRECTORY)
  set(_Freenect2_probe "${CMAKE_CURRENT_BINARY_DIR}/freenect2_probe.cpp")
  file(WRITE "${_Freenect2_probe}" [=[
#include <iostream>
#include <libfreenect2/libfreenect2.hpp>
int main() {
  std::cout << libfreenect2::getVersion() << ";"
            << libfreenect2::getApiVersion() << ";"
            << libfreenect2::getBuildRevision();
  return 0;
}
]=])
  try_run(
    Freenect2_PROBE_RUN Freenect2_PROBE_COMPILE
    SOURCES "${_Freenect2_probe}"
    LINK_LIBRARIES Freenect2::Freenect2
    CXX_STANDARD 17
    RUN_OUTPUT_VARIABLE Freenect2_PROBE_OUTPUT
    COMPILE_OUTPUT_VARIABLE Freenect2_PROBE_BUILD_OUTPUT
  )
  if(NOT Freenect2_PROBE_COMPILE)
    message(FATAL_ERROR "Could not compile the libfreenect2 probe:\n${Freenect2_PROBE_BUILD_OUTPUT}")
  endif()
  if(NOT Freenect2_PROBE_RUN EQUAL 0)
    message(FATAL_ERROR "Could not run the libfreenect2 probe (exit ${Freenect2_PROBE_RUN})")
  endif()
  string(STRIP "${Freenect2_PROBE_OUTPUT}" Freenect2_PROBE_OUTPUT)
  set(_Freenect2_probe_values "${Freenect2_PROBE_OUTPUT}")
  list(GET _Freenect2_probe_values 0 Freenect2_RUNTIME_VERSION)
  list(GET _Freenect2_probe_values 1 Freenect2_RUNTIME_API)
  list(GET _Freenect2_probe_values 2 Freenect2_BUILD_REVISION)
  if(NOT Freenect2_RUNTIME_VERSION MATCHES "^0\\.3\\." OR NOT Freenect2_RUNTIME_API STREQUAL "3")
    message(FATAL_ERROR
      "libfreenect2 runtime 0.3.x with API 3 is required; found "
      "${Freenect2_RUNTIME_VERSION} with API ${Freenect2_RUNTIME_API}"
    )
  endif()
  message(STATUS "pylibfreenect3 native core:")
  message(STATUS "  discovery: ${Freenect2_DISCOVERY}")
  message(STATUS "  headers: ${Freenect2_INCLUDE_DIR}")
  message(STATUS "  library: ${Freenect2_LIBRARY}")
  message(STATUS "  runtime: ${Freenect2_RUNTIME_VERSION} (API ${Freenect2_RUNTIME_API})")
  message(STATUS "  revision: ${Freenect2_BUILD_REVISION}")
endif()

mark_as_advanced(Freenect2_INCLUDE_DIR Freenect2_LIBRARY)
