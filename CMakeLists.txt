cmake_minimum_required(VERSION 3.16)

project(detectOnVideo VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)

set(USE_PACKAGES Widgets
  OpenGLWidgets
  Core
  Gui
  Qml
  Quick
  QuickControls2)

find_package(QT NAMES Qt6 REQUIRED COMPONENTS ${USE_PACKAGES})
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS ${USE_PACKAGES})
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
pkg_check_modules(JANSSON REQUIRED jansson)

if(${OpenCV_FOUND})
    message(STATUS "OpenCV library status:")
    message(STATUS "    config: ${OpenCV_DIR}")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV не найден! Установите OpenCV 4.x.")
endif()

find_package(onnxruntime REQUIRED)
if(onnxruntime_FOUND)
    message(STATUS "onnxruntime найден!")
else()
    message(FATAL_ERROR "onnxruntime не найден! Установите onnxruntime-dev.")
endif()

file(GLOB SRC_FILES "*.cpp")
file(GLOB HDR_FILES "*.h")
file(GLOB UI_FILES "*.ui")

set(PROJECT_SOURCES ${SRC_FILES} ${HDR_FILES} ${UI_FILES})

add_executable(detectOnVideo
  ${PROJECT_SOURCES} resources.qrc
  QVideoPlayer.h QVideoPlayer.cpp
)

target_link_libraries(detectOnVideo PRIVATE
  Qt${QT_VERSION_MAJOR}::Widgets
  Qt${QT_VERSION_MAJOR}::OpenGLWidgets
  Qt${QT_VERSION_MAJOR}::Qml
  Qt${QT_VERSION_MAJOR}::Quick
  Qt${QT_VERSION_MAJOR}::QuickControls2
  pthread
  ${OpenCV_LIBS}
  onnxruntime::onnxruntime
)

if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.vvs.adaptivePIDview)
endif()

IF (WIN32)
  set_target_properties(detectOnVideo PROPERTIES
      ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE)
endif()

include(GNUInstallDirs)
install(TARGETS detectOnVideo
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(detectOnVideo)
endif()
