# tests —— mini-mlmath 的测试程序（每模块一个 *_test.cpp）
#
# 每个 *_test.cpp 都是独立可执行：正确性验证 + 可复现的 benchmark。
# 头文件写进 sources 是为了在 IDE 里能看到（教学项目，方便直接点开读）。

add_executable(matrix_test matrix_test.cpp)
add_executable(softmax_test softmax_test.cpp)
add_executable(vector_test vector_test.cpp)
add_executable(logic_gate logic_gate.cpp)
add_executable(random_check random_check.cpp)   # int 离散均匀抽样 smoke test
add_executable(perceptron_test perceptron_test.cpp)   # 端到端测 Perceptron 类
add_executable(knn_test knn_test.cpp)   # KNN：多数投票 + metric / strategy 扩展点

foreach(tgt matrix_test softmax_test vector_test logic_gate random_check perceptron_test knn_test)
  target_link_libraries(${tgt} PRIVATE mini_mlmath)

# =====================================================================
# 性能相关编译选项（这是本库的第一条真实性能教训！）
#
# 教训 1：GCC/Clang 的 -O2 不会自动向量化。
#   -O2 会开 inline、常量折叠、公共子表达式消除等，但**不开 loop vectorize /
#   SLP vectorize**。本库三种乘法的内层都是「读一批连续 double、乘加」，
#   不开向量化时编译器生成的就是纯标量 SSE 逐个处理，CPU 的 AVX2 寄存器
#   一次能算 4 个 double（或 8 个 float）却只在用 1/4 的吞吐。
#   必须 -O3（等价于 -O2 + -ftree-vectorize -fopt-info 等）才能向量化，
#   否则内层是标量循环、内存根本跑不满，下面的分块/打包优化毫无意义。
#
#   实测（本机 g++ 13.3，N=2048 的 double 乘法，单位 ms）：
#     -O2 时：packed 4128、blocked 4040、naive 4073 —— 三版几乎打平！
#     -O3 时：packed  941、blocked 1887、naive 2845（packed 快约 4.4 倍）
#   关键现象：-O2 下「分块/打包全部失效」，三版一样慢。因为内层没有向量化，
#   标量乘加已把 CPU 塞满，内存带宽根本不是瓶颈 —— 优化内存访问毫无意义。
#
# 教训 2：-march=native 只对当前机器有效！
#   native = 「按本机 CPU 探测到的全部指令集编译」，本机是 x86 的
#   AVX2/FMA 时代，编译器才会发 vfmadd / vmulpd / vaddpd。
#   注意：把源码/编译产物拷到别的机器（尤其 CI、服务器、别人的电脑）
#   要么删掉这行，要么写死成具体型号如 -march=haswell / -march=znver4，
#   否则程序可能在旧 CPU 上直接 SIGILL（非法指令崩溃）。
# =====================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(${tgt} PRIVATE -O3 -march=native)
elseif(MSVC)
  # MSVC 没有 -O3 / -march 的概念，对应物是 /O2 和 /arch:AVX2。
  # 注意 /utf-8 是**必须项**：本库注释和打印的中文全是 UTF-8 编码，MSVC
  # 默认按系统代码页（中文 Windows 是 GBK/936）解析源文件，不指定会乱码，
  # 严重的还会报 C4819 警告甚至编译错误。GCC/Clang 默认 UTF-8 不需要。
  target_compile_options(${tgt} PRIVATE /O2 /arch:AVX2 /utf-8)
endif()
endforeach()

# 真实世界里：Eigen / OpenBLAS 编译时都会用 -march=native（或更精细的
# -mavx2 -mfma -fopenmp-simd），这决定了「同样的源码在不同机器上速度
# 可能差出好几倍」。
