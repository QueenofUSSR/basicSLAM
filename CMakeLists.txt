cmake_minimum_required(VERSION 3.16)
project(basicSLAM VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)    # 对编辑器友好

# 如果你源码安装了 OpenCV 到 /usr/local，CMake 有时需要明确路径：
# set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")

find_package(OpenCV REQUIRED COMPONENTS core highgui features2d imgproc imgcodecs calib3d video)

# Optional: detect g2o via pkg-config and enable g2o-based optimizer when available


# 源码和头文件
set(PROJ_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
file(GLOB_RECURSE PROJ_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV DIR: ${OpenCV_DIR}")

# 创建库（静态或共享，根据需要切换 STATIC/SHARED/omit）
add_library(basicslam_lib STATIC ${PROJ_SOURCES})
target_include_directories(basicslam_lib
    PUBLIC
        $<BUILD_INTERFACE:${PROJ_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_compile_options(basicslam_lib PRIVATE -Wall -Wextra -Wpedantic)

# 链接 OpenCV 到库（可选：也可以链接到每个测试可执行）
target_link_libraries(basicslam_lib
    PUBLIC
        ${OpenCV_LIBS}
)

# Try to detect OpenCV SFM (optional, for BA with Ceres)
set(OPENCV_SFM_FOUND OFF)
find_package(OpenCV COMPONENTS sfm QUIET)
if(TARGET opencv_sfm OR (OpenCV_LIBS MATCHES "opencv_sfm"))
    set(OPENCV_SFM_FOUND ON)
    message(STATUS "OpenCV SFM module found. BA backend will be enabled.")
    target_compile_definitions(basicslam_lib PUBLIC USE_OPENCV_SFM)
    # Ensure we link against sfm if present in OpenCV libs
    target_link_libraries(basicslam_lib PUBLIC opencv_sfm)
endif()

# Optional: detect g2o and enable g2o-based optimizer when available.
# First try pkg-config, then fallback to manual detection of headers/libs.
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(G2O g2o QUIET)
    if(G2O_FOUND)
        message(STATUS "g2o found: ${G2O_VERSION}")
        target_include_directories(basicslam_lib PUBLIC ${G2O_INCLUDE_DIRS})
        target_link_libraries(basicslam_lib PUBLIC ${G2O_LIBRARIES})
        target_compile_definitions(basicslam_lib PUBLIC USE_G2O)
    else()
        message(STATUS "g2o not found via pkg-config; attempting manual detection")
        find_path(G2O_INCLUDE_DIR NAMES g2o/core/sparse_optimizer.h HINTS /usr/include /usr/local/include)
        find_library(G2O_CORE_LIB NAMES g2o_core g2o_core.so g2o_core.a HINTS /usr/lib /usr/local/lib)
        find_library(G2O_STUFF_LIB NAMES g2o_stuff g2o_stuff.so g2o_stuff.a HINTS /usr/lib /usr/local/lib)
        find_library(G2O_TYPES_SBA_LIB NAMES g2o_types_sba g2o_types_sba.so g2o_types_sba.a HINTS /usr/lib /usr/local/lib)
        find_library(G2O_SOLVER_DENSE_LIB NAMES g2o_solver_dense g2o_solver_dense.so g2o_solver_dense.a HINTS /usr/lib /usr/local/lib)
        if(G2O_INCLUDE_DIR AND G2O_CORE_LIB)
            message(STATUS "g2o detected manually: include=${G2O_INCLUDE_DIR} core=${G2O_CORE_LIB}")
            # Ensure Eigen include is available (g2o headers expect Eigen/Core)
            find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core HINTS /usr/include/eigen3 /usr/local/include)
            if(EIGEN3_INCLUDE_DIR)
                target_include_directories(basicslam_lib PUBLIC ${G2O_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
            else()
                target_include_directories(basicslam_lib PUBLIC ${G2O_INCLUDE_DIR})
            endif()
            # Try to find common g2o components to link
            find_library(G2O_TYPES_SBA_LIB NAMES g2o_types_sba g2o_types_sba.so g2o_types_sba.a HINTS /usr/lib /usr/local/lib)
            # the VertexPointXYZ vtable is provided by g2o_types_slam3d on most distributions
            find_library(G2O_TYPES_SLAM3D_LIB NAMES g2o_types_slam3d g2o_types_slam3d.so g2o_types_slam3d.a HINTS /usr/lib /usr/local/lib)
            find_library(G2O_STUFF_LIB NAMES g2o_stuff g2o_stuff.so g2o_stuff.a HINTS /usr/lib /usr/local/lib)
            find_library(G2O_SOLVER_DENSE_LIB NAMES g2o_solver_dense g2o_solver_dense.so g2o_solver_dense.a HINTS /usr/lib /usr/local/lib)
            # Prefer solver_cholmod or solver_csparse if available for better performance
            find_library(G2O_SOLVER_CHOLMOD_LIB NAMES g2o_solver_cholmod g2o_solver_cholmod.so g2o_solver_cholmod.a HINTS /usr/lib /usr/local/lib)
            find_library(G2O_SOLVER_CSPARSE_LIB NAMES g2o_solver_csparse g2o_solver_csparse.so g2o_solver_csparse.a HINTS /usr/lib /usr/local/lib)

            set(G2O_LINK_LIBS ${G2O_CORE_LIB})
            # Prefer linking the SLAM3D types library (contains VertexPointXYZ vtable/typeinfo)
            if(G2O_TYPES_SLAM3D_LIB)
                list(APPEND G2O_LINK_LIBS ${G2O_TYPES_SLAM3D_LIB})
            endif()
            # Also link the SBA types (contains CameraParameters, EdgeProjectXYZ2UV, VertexSE3Expmap)
            if(G2O_TYPES_SBA_LIB)
                list(APPEND G2O_LINK_LIBS ${G2O_TYPES_SBA_LIB})
            endif()
            if(G2O_STUFF_LIB)
                list(APPEND G2O_LINK_LIBS ${G2O_STUFF_LIB})
            endif()
            if(G2O_SOLVER_CHOLMOD_LIB)
                list(APPEND G2O_LINK_LIBS ${G2O_SOLVER_CHOLMOD_LIB})
            elseif(G2O_SOLVER_CSPARSE_LIB)
                list(APPEND G2O_LINK_LIBS ${G2O_SOLVER_CSPARSE_LIB})
            elseif(G2O_SOLVER_DENSE_LIB)
                list(APPEND G2O_LINK_LIBS ${G2O_SOLVER_DENSE_LIB})
            endif()

            if(G2O_LINK_LIBS)
                message(STATUS "Linking g2o libs: ${G2O_LINK_LIBS}")
                target_link_libraries(basicslam_lib PUBLIC ${G2O_LINK_LIBS})
                target_compile_definitions(basicslam_lib PUBLIC USE_G2O)
            else()
                message(STATUS "g2o headers found but no runtime libs located; skipping g2o link")
            endif()
        else()
            message(STATUS "g2o not found; building without g2o support")
        endif()
    endif()
else()
    message(STATUS "pkg-config not found; skipping g2o detection")
endif()

# 设置输出目录（可选）
set_target_properties(basicslam_lib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ---- Tests: 为 Tests/*.cpp 每个创建一个可执行并链接 basicslam_lib ----
enable_testing()

file(GLOB TEST_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/Tests/*.cpp)

foreach(test_src IN LISTS TEST_SOURCES)
    # test_src is like "Tests/test_foo.cpp"
    get_filename_component(test_name ${test_src} NAME_WE) # test_foo
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE basicslam_lib ${OpenCV_LIBS})
    target_include_directories(${test_name} PRIVATE ${PROJ_INCLUDE_DIR})

    # 将可执行放在统一目录（可选）
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )

    # 注册到 CTest（可选）
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()