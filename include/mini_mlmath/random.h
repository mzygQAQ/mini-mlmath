// ============================================================================
//  random.h —— 类似 numpy.random 的随机数工具（header-only，随 Matrix.h 一起用）
//
//  干的事：生成随机标量 / 随机 Matrix。教学上最常见的用途就是
//    - 造带噪声的数据（给模型加一点高斯扰动）
//    - 生成随机矩阵做矩阵乘法 benchmark（matrix_test.cpp 里那批随机数据）
//    - 初始化权重（机器学习里 W ~ N(0, 1) 之类的）
//
//  和 numpy 的对应关系：
//    numpy.random.uniform(low, high, size)  ->  rng.uniform<T>(a, b)
//    numpy.random.normal(loc, scale, size)  ->  rng.normal<T>(mean, variance)
//    numpy.random.rand(d0, d1)              ->  rng.uniform_matrix<T>(rows, cols)
//    numpy.random.randn(d0, d1)             ->  rng.normal_matrix<T>(rows, cols)
//    注意：numpy 的 normal 第 2 个参数是标准差 scale（std），而本库直接给
//    **方差** variance —— 因为方差才是更常用、也更直观的说法
//    （var = std²，内部实现里会 sqrt 回去）。
//
//  标量版是模板：rng.uniform<T>(lo, hi) 的 T 从 lo/hi 实参推导，
//  返回类型 = T。矩阵版同理。这样 Perceptron<float> / Perceptron<double>
//  都能直接调用，不用手动 static_cast。
//
//  为什么用「类」？
//    随机数生成是典型的有状态运算：mt19937 引擎要记住上一次的输出才能产生
//    下一个「看起来无关」的数。所以把引擎包进 Random 对象（和 vector.h 的
//    Vector、feature_selection 的 VarianceThreshold 同逻辑）。
//
//  两个值得记的要点：
//    1) 伪随机 ≠ 真随机：mt19937 是确定性算法，给定种子，序列完全固定。
//       好处是**可复现** —— 传同一个 seed，两次运行得到一模一样的矩阵，
//       教学实验和 benchmark 都靠它保证公平对比。
//    2) 用分布对象（std::uniform_real_distribution / std::normal_distribution）
//       而不是手写公式：normal_distribution 内部是 Box-Muller / ziggurat，
//       手写容易出数值问题，交给标准库即可。
//    3) T 必须是浮点类型（float / double / long double），不是「任何数」。
//       原因：T = int 时 int(-0.5) = int(0.5) = 0，uniform<int>(-0.5, 0.5)
//       会静默退化成「永远返回 0」—— 随机初始化变成全 0 初始化，
//       而且能编译通过。T = bool / 整数家族都不能用。
//       所以接口层 static_assert 拦住，编译期就报错。
// ============================================================================
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>

#include "mini_mlmath/matrix.h"

class Random {
public:
    // 默认种子 42：开箱即可复现（教学/benchmark 友好）。想每次真随机，
    // 可传 std::random_device{}()（见文件头注释）。
    explicit Random(std::uint64_t seed = 42) : engine_(seed) {}

    // 重新播种，之后重新生成（可复现实验的标准操作）
    void seed(std::uint64_t s) { engine_.seed(s); }

    // ---- 标量（模板：T 从实参推导，例 rng.uniform<double>(-1, 1)）----

    // [low, high) 内均匀分布的一个标量 T（numpy 的 uniform）
    template <typename T>
    T uniform(T low = T(0), T high = T(1));

    // 高斯分布的一个标量 T：均值 mean、方差 variance
    // （内部取 std::sqrt(variance) 转成 normal_distribution 需要的标准差）
    template <typename T>
    T normal(T mean = T(0), T variance = T(1));

    // ---- Matrix 生成器（模板，T 从 low/high 或 mean/variance 实参推导）----

    // rows×cols 矩阵，每个元素 iid ~ U(low, high)（numpy 的 rand / uniform + size）
    template <typename T>
    Matrix<T> uniform_matrix(std::size_t rows, std::size_t cols,
                             T low = T(0), T high = T(1));

    // rows×cols 矩阵，每个元素 iid ~ N(mean, variance)
    template <typename T>
    Matrix<T> normal_matrix(std::size_t rows, std::size_t cols,
                            T mean = T(0), T variance = T(1));

private:
    std::mt19937 engine_;   // 梅森旋转伪随机引擎（状态全在这）
};

// ----------------------------------------------------------------------------
// 矩阵生成器实现（模板，定义必须留在头文件里；直接走 data() 裸指针填充，
// 和 matrix.h 乘法内层一个道理 —— 绕开 operator() 的越界检查）
// ----------------------------------------------------------------------------
template <typename T>
Matrix<T> Random::uniform_matrix(std::size_t rows, std::size_t cols,
                                 T low, T high) {
    static_assert(std::is_floating_point_v<T>,
                  "Random::uniform_matrix<T> requires floating-point T "
                  "(float/double/long double). For integer matrices, fill manually.");
    Matrix<T> m(rows, cols);
    std::uniform_real_distribution<T> dist(low, high);
    T* d = m.data();
    for (std::size_t i = 0; i < rows * cols; ++i) d[i] = dist(engine_);
    return m;
}

template <typename T>
Matrix<T> Random::normal_matrix(std::size_t rows, std::size_t cols,
                                T mean, T variance) {
    static_assert(std::is_floating_point_v<T>,
                  "Random::normal_matrix<T> requires floating-point T "
                  "(float/double/long double). For integer matrices, fill manually.");
    Matrix<T> m(rows, cols);
    std::normal_distribution<T> dist(mean, std::sqrt(variance));
    T* d = m.data();
    for (std::size_t i = 0; i < rows * cols; ++i) d[i] = dist(engine_);
    return m;
}

// ----------------------------------------------------------------------------
// 标量版实现（同样模板化留在头文件）。两个版本都直接用 std::xxx_distribution<T>
// —— 不再走「先转 double 再转回 T」的弯路，long double 也能保留精度。
// T 必须是浮点类型（static_assert 已拦，编译期就报错），不绕开。
// ----------------------------------------------------------------------------
template <typename T>
T Random::uniform(T low, T high) {
    static_assert(std::is_floating_point_v<T>,
                  "Random::uniform<T> requires floating-point T "
                  "(float/double/long double). int(-0.5) silently truncates to 0, "
                  "so int weights would always sample 0 — use T=double/float instead.");
    std::uniform_real_distribution<T> dist(low, high);
    return dist(engine_);
}

template <typename T>
T Random::normal(T mean, T variance) {
    static_assert(std::is_floating_point_v<T>,
                  "Random::normal<T> requires floating-point T "
                  "(float/double/long double). Integer weights lose precision silently.");
    std::normal_distribution<T> dist(mean, std::sqrt(variance));
    return dist(engine_);
}
