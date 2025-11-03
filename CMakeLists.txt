cmake_minimum_required(VERSION 3.16)
project(basicSLAM VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)    # 对编辑器友好

# 如果你源码安装了 OpenCV 到 /usr/local，CMake 有时需要明确路径：
# set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")

find_package(OpenCV REQUIRED COMPONENTS core highgui features2d imgproc imgcodecs calib3d video)

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