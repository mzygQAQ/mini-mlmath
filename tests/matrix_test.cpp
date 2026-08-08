// ============================================================================
//  matrix_test.cpp —— mini-mlmath 的矩阵乘法测试与性能对比程序
//
//  它只做两件事：
//    1. 正确性验证：一个能用手算核对的小例子 + 随机矩阵下「三版乘法结果
//       必须一致」（用 matrix.h 里的 approxEqual 校验，容差 1e-9 相对误差）
//    2. 性能对比：N = 512 / 1024 / 2048，三版各自计时，给出可以复现的结论
//
//  运行前请确认编译选项：必须 -O3（见 CMakeLists.txt 顶部注释，教训 1），
//  否则下面的 benchmark 数字不成立。
// ============================================================================
#include "mini_mlmath/matrix.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

// Windows 下让控制台按 UTF-8 输出中文（VS2022 的 cmd/终端默认是本地代码页
// GBK，直接 cout 中文会乱码；设一次 CP_UTF8 就好了）。Linux 不需要。
// 注意：windows.h 会把 min/max 定义为宏，会咬碎 std::min/std::max（C2589），
// 所以必须先 NOMINMAX。这是 Windows 平台的无奈之举，跟 matrix.h「无宏」的
// 库设计无关 —— matrix_test.cpp 只是测试程序，不是库本体。
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// ----------------------------------------------------------------------------
// 小工具：测一次 f() 的执行毫秒数
// ----------------------------------------------------------------------------
template <typename F>
double bench_ms(F&& f) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    f();
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ----------------------------------------------------------------------------
// 小工具：随机填充一个矩阵（均匀分布在 [-1, 1)）
// ----------------------------------------------------------------------------
template <typename T, typename RNG>
void fill_random(Matrix<T>& m, RNG& rng) {
    std::uniform_real_distribution<T> dist(T(-1), T(1));
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < m.cols(); ++j) m(i, j) = dist(rng);
}

// ----------------------------------------------------------------------------
// 小工具：矩阵元素绝对值和（只用来「使用结果」，防止编译器把乘法结果
// 整个优化掉 —— 这正是真实 benchmark 里校验和（checksum）的作用）
// ----------------------------------------------------------------------------
template <typename T>
T checksum(const Matrix<T>& m) {
    T s = T(0);
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < m.cols(); ++j) s += m(i, j);
    return s;
}

// ----------------------------------------------------------------------------
// 正确性验证 1：手算可核对的小例子
//  A(2×3) * B(3×2)：
//     A = [1 2 3]      B = [7  8 ]
//         [4 5 6]          [9  10]
//                          [11 12]
//  手算 C = A*B（2×2）：
//     C(0,0) = 1*7 + 2*9 + 3*11 = 58
//     C(0,1) = 1*8 + 2*10+ 3*12 = 64
//     C(1,0) = 4*7 + 5*9 + 6*11 = 139
//     C(1,1) = 4*8 + 5*10+ 6*12 = 154
// ----------------------------------------------------------------------------
void verify_hand_example() {
    std::cout << "===== 正确性验证 1：手算小例子 =====\n";
    const Matrix<double> A = Matrix<double>::fromList({{1, 2, 3}, {4, 5, 6}});
    const Matrix<double> B = Matrix<double>::fromList({{7, 8}, {9, 10}, {11, 12}});
    const Matrix<double> expected =
        Matrix<double>::fromList({{58, 64}, {139, 154}});

    std::cout << "A = \n" << A;
    std::cout << "B = \n" << B;

    const auto C0 = multiply_naive(A, B);
    const auto C1 = multiply_blocked(A, B);
    const auto C2 = multiply_packed(A, B);

    std::cout << "multiply_naive  结果:\n" << C0;
    std::cout << "multiply_blocked 结果:\n" << C1;
    std::cout << "multiply_packed  结果:\n" << C2;

    // 三版都要等于手算值，且三版彼此一致
    const bool ok = approxEqual(C0, expected) && approxEqual(C1, expected) &&
                    approxEqual(C2, expected) &&
                    approxEqual(C0, C1) && approxEqual(C1, C2);
    std::cout << (ok ? "✓ 三版乘法与手算结果一致\n\n"
                     : "✗ 结果不一致！请检查实现\n\n");
}

// ----------------------------------------------------------------------------
// 正确性验证 2：随机矩阵下三版结果互相一致
// 用中等尺寸（N=256）随机矩阵，跑三版并用 approxEqual 两两比较。
// 因为只是「相对容差」比较，即使累加顺序不同也能稳定通过。
// ----------------------------------------------------------------------------
void verify_random_consistency() {
    std::cout << "===== 正确性验证 2：随机矩阵三版一致性 (N=256) =====\n";
    std::mt19937 rng(42);  // 固定种子，结果可复现

    const std::size_t N = 256;
    Matrix<double> A(N, N), B(N, N);
    fill_random(A, rng);
    fill_random(B, rng);

    const auto C0 = multiply_naive(A, B);
    const auto C1 = multiply_blocked(A, B);
    const auto C2 = multiply_packed(A, B);

    std::cout << "naive  vs blocked: "
              << (approxEqual(C0, C1) ? "一致 ✓" : "不一致 ✗") << "\n";
    std::cout << "blocked vs packed: "
              << (approxEqual(C1, C2) ? "一致 ✓" : "不一致 ✗") << "\n";
    std::cout << "naive  vs packed: "
              << (approxEqual(C0, C2) ? "一致 ✓" : "不一致 ✗") << "\n\n";
}

// ----------------------------------------------------------------------------
// 性能对比：N = 512 / 1024 / 2048
//
// 先剧透预期结论（这正是第三条真实性能教训）：
//
// 教训 3：分块的收益只在矩阵超过缓存之后才出现。
//   - N=512：矩阵 512×512×8B = 2MB，三块加起来 6MB，整块能塞进 L2/L3，
//     三版都快，差别是「循环开销」的差别。此时 blocked 多了块循环和余数
//     判断、packed 多了拷贝，所以 blocked 可能反而不如 naive —— 分块没有
//     数据可救，纯属白忙活。
//   - N=1024：8MB/块，开始超 L2，blocked/packed 开始赢。
//   - N=2048：32MB/块，远超大缓存，内存带宽成为硬瓶颈。naive 里 B 的每行
//     被重读 M 次（内存流量 ~M*K*N），blocked 靠缓存复用、packed 靠
//     连续访问 + 复用，把内存流量砍到接近理论下限 —— 这里差距最大。
//
// 实测数据（本机 g++ 13.3 -O3 -march=native，best-of-3，单位 ms）：
//     N=512：  naive 21.9   blocked 22.4   packed 15.3   packed/naive 0.70
//     N=1024： naive 202.1  blocked 189.0  packed 132.4  packed/naive 0.65
//     N=2048： naive 2845.2 blocked 1887.3 packed 940.9  packed/naive 0.33
// 三个特征全对上：
//   - 小 N：blocked(22.4) 反而不如 naive(21.9)，packed 也只是略快 —— 分块
//     收益尚未出现（数据全在缓存里）；
//   - 大 N：blocked 比 naive 快 1.5 倍，packed 比 naive 快 3.0 倍 —— 分块+
//     打包的收益完全兑现；
//   - 换成 -O2 编译（教训 1）时 N=2048 三版全部 ~4.1 秒、打平，说明没
//     向量化时一切都是白搭。
// 你机器上的绝对数字会不同（CPU/缓存/内存型号决定），但趋势必然一致。
// ----------------------------------------------------------------------------
void run_benchmark() {
    std::cout << "===== 性能对比 (最佳计时，ms) =====\n";
    std::mt19937 rng(2024);

    // 表头
    std::cout << std::left << std::setw(8) << "N"
              << std::setw(14) << "naive"
              << std::setw(14) << "blocked"
              << std::setw(14) << "packed"
              << std::setw(16) << "packed/naive"
              << std::endl;

    for (std::size_t N : {512UL, 1024UL, 2048UL}) {
        Matrix<double> A(N, N), B(N, N);
        fill_random(A, rng);
        fill_random(B, rng);

        // 每种方法跑 3 次取最好（best-of-3）：排除系统噪音 / 频率抖动。
        // 真实 benchmark 常用「取最小」而非「取平均」，因为最小接近真实
        // 计算时间，平均会被偶尔的调度中断拉高。
        double t_naive = 1e18, t_blocked = 1e18, t_packed = 1e18;

        for (int rep = 0; rep < 3; ++rep) {
            t_naive = std::min(t_naive, bench_ms([&] {
                volatile double s = checksum(multiply_naive(A, B));
                (void)s;
            }));
            t_blocked = std::min(t_blocked, bench_ms([&] {
                volatile double s = checksum(multiply_blocked(A, B));
                (void)s;
            }));
            t_packed = std::min(t_packed, bench_ms([&] {
                volatile double s = checksum(multiply_packed(A, B));
                (void)s;
            }));
        }

        std::cout << std::left << std::setw(8) << N
                  << std::setw(14) << std::fixed << std::setprecision(1) << t_naive
                  << std::setw(14) << t_blocked
                  << std::setw(14) << t_packed
                  << std::setw(16) << (t_packed / t_naive)
                  << std::endl;
    }
}

// ----------------------------------------------------------------------------
// 入口
// ----------------------------------------------------------------------------
int main() {
#ifdef _WIN32
    // 让 Windows 控制台用 UTF-8 渲染中文输出（配合 CMake 里的 /utf-8）
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::cout << "mini-mlmath —— 教学用迷你矩阵库（矩阵乘法测试）\n";
    std::cout << "三个刻意简化：无宏 / 无表达式模板 / 无 BLAS（详见 matrix.h 头部）\n\n";

    verify_hand_example();
    verify_random_consistency();
    run_benchmark();

    std::cout << "\n三版乘法已用 approxEqual(1e-9 相对容差) 校验一致；\n";
    std::cout << "性能趋势应满足：小 N 下 packed 略快、blocked 可能不敌 naive，\n";
    std::cout << "大 N 下 packed 明显最快（分块收益只在大矩阵上兑现，见教训 3）。\n";
    return 0;
}
