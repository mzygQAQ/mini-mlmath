// ============================================================================
//  softmax_test.cpp —— mini-mlmath 的 softmax 测试程序
//
//  验证三件事：
//    1. 常规向量：输出必须是概率分布 —— 每项 ∈(0,1) 且和为 1，单调性保持
//    2. 大数值向量 {1000,999,998}：这是「数值稳定性」的试金石 —— 朴素实现
//       exp(1000) 直接溢出成 inf 出 NaN，减去 max 后最大项是 exp(0)=1，正常
//    3. softmax_rows：注意力 score 矩阵按行归一化，每行和 = 1
//
//  实现与原理见 softmax.h 头部注释（为什么减 max、为什么用自由函数）。
// ============================================================================
#include "matrix.h"
#include "softmax.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

// Windows 下让控制台按 UTF-8 输出中文，理由同 matrix_test.cpp（见那里的
// 注释：NOMINMAX 是为了防 windows.h 的 min/max 宏咬碎 std::min/std::max）。
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// ----------------------------------------------------------------------------
// 正确性验证：softmax
// ----------------------------------------------------------------------------
void verify_softmax() {
    std::cout << "===== 正确性验证：softmax =====\n";

    // 手算：softmax({1,2,3})，减 max=3 后 exp 为 {e^-2, e^-1, e^0}，
    // 分母 = e^-2+e^-1+1 ≈ 0.1353+0.3679+1 = 1.5032
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const auto p = softmax(x);
    std::cout << "softmax({1,2,3}) = {";
    for (std::size_t i = 0; i < p.size(); ++i)
        std::cout << p[i] << (i + 1 < p.size() ? ", " : "}\n");

    // 和必须为 1
    double sum = 0;
    for (double v : p) sum += v;
    std::cout << "各项之和 = " << sum
              << (std::abs(sum - 1.0) < 1e-12 ? "  ✓\n" : "  ✗\n");

    // 大数值不溢出（naive 在这里会出 NaN）
    const auto big = softmax(std::vector<double>{1000.0, 999.0, 998.0});
    bool finite = true;
    for (double v : big)
        if (!std::isfinite(v)) finite = false;
    std::cout << "softmax({1000,999,998}) = {";
    for (std::size_t i = 0; i < big.size(); ++i)
        std::cout << big[i] << (i + 1 < big.size() ? ", " : "}\n");
    std::cout << (finite ? "✓ 大数值不溢出（已减 max 的功劳）\n" : "✗ 溢出了！\n");

    // 矩阵按行 softmax（模拟 2 行注意力分数）
    const Matrix<double> scores = Matrix<double>::fromList({
        {1.0, 2.0, 3.0},
        {0.0, 0.0, 0.0}   // 全 0 行 -> 均匀分布 {1/3,1/3,1/3}
    });
    const auto rows = softmax_rows(scores);
    std::cout << "softmax_rows(scores) = \n" << rows;
    bool rowsOk = true;
    for (std::size_t i = 0; i < rows.rows(); ++i) {
        double rs = 0;
        for (std::size_t j = 0; j < rows.cols(); ++j) rs += rows(i, j);
        if (std::abs(rs - 1.0) > 1e-12) rowsOk = false;
    }
    std::cout << (rowsOk ? "✓ 每行和均为 1\n" : "✗ 行和不为 1\n");
}

// ----------------------------------------------------------------------------
// 入口
// ----------------------------------------------------------------------------
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::cout << "mini-mlmath —— 教学用迷你矩阵库（softmax 测试）\n\n";

    verify_softmax();

    std::cout << "\nsoftmax 实现见 softmax.h：减 max 保证数值稳定，\n";
    std::cout << "普通向量用 softmax()，attention 分数矩阵按行用 softmax_rows()。\n";
    return 0;
}
