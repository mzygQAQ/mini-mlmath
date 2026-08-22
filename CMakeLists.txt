# mini-mlmath —— 教学用迷你矩阵库（header-only）
#
# 目录契约（标准 C++ 库布局）：
#   include/mini_mlmath/   公开头文件（库本体，全部 header-only）
#   tests/                 测试程序（每个模块一个 *_test.cpp）
#
# 用法一（独立构建）：
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ./build/matrix_test      # 矩阵乘法：正确性 + 性能对比
#   ./build/softmax_test     # softmax：数值稳定性 + 按行归一化
#   ./build/vector_test      # 向量：点积 / 模长 / 余弦相似度
#   ./build/logic_gate       # 感知机学逻辑门（AND/OR 可学，XOR 线性不可分）
#
# 用法二（挂进已有项目，推荐）：
#   本库做成 INTERFACE target（无编译产物，纯传播头文件路径和 C++ 标准），
#   在你的 CMakeLists 里 add_subdirectory 或 fetchcontent 之后直接：
#       target_link_libraries(your_app PRIVATE mini_mlmath)
#   即可 include <mini_mlmath/matrix.h>，零安装、零依赖。
#
# 用 VS2022 打开运行（Windows 用户）：
#   直接用 VS2022「打开本地文件夹」指向本目录即可（CMake 会自动配置）。
#   两个注意点：
#     1) 把顶部配置切到 Release —— Debug 无优化，benchmark 数字没意义；
#     2) 平台选 x64 —— /arch:AVX2 只在 64 位生效。

cmake_minimum_required(VERSION 3.14)
project(mini_mlmath LANGUAGES CXX)

# =============================================================================
# 库本体：INTERFACE 库。
#   - 头文件路径通过 target_include_directories 暴露给所有链接者
#     （include/ 是根，所以调用方写 #include <mini_mlmath/matrix.h>）；
#   - cxx_std_17 作为编译要求传播出去，链接方自动拿到 C++17。
# =============================================================================
add_library(mini_mlmath INTERFACE)
target_include_directories(mini_mlmath INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(mini_mlmath INTERFACE cxx_std_17)

# 测试统一挂在 BUILD_TESTING 开关下（cmake -DBUILD_TESTING=OFF 可关闭），
# 这是把测试程序和库本体解耦的标准做法。
include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
