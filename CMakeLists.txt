cmake_minimum_required(VERSION 3.29)
project(pongsasos)

set(CMAKE_CXX_STANDARD 20)

set(RAYLIB_ROOT "/opt/homebrew/opt/raylib")
list(APPEND CMAKE_PREFIX_PATH "${RAYLIB_ROOT}/lib/cmake/raylib")

find_package(raylib CONFIG REQUIRED)

add_executable(pongsasos main.cpp)

target_link_libraries(pongsasos PRIVATE raylib)

if(APPLE)
    target_link_libraries(pongsasos PRIVATE
            "-framework Cocoa"
            "-framework IOKit"
            "-framework CoreFoundation"
            "-framework CoreVideo"
            "-framework OpenGL")
endif()