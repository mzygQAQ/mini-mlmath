// ============================================================================
//  SoftMax.h —— 数值稳定的 softmax（header-only，随 Matrix.h 一起用）
//
//  softmax 在注意力/神经网络里几乎无处不在（KV cache 的 attention score
//  就要过 softmax），所以给 mini-mlmath 配一个教学版实现。
//
//  为什么是「自由函数」而不是 class？
//   softmax 是一个**无状态**的纯函数运算：输入一批数、输出概率分布，
//  没有任何内部状态要保存。为无状态运算硬造一个类只会多一层没用的包装，
//  这正是「朴素 > 花哨」的教学取向 —— 也和本库「无表达式模板、一眼看懂
//  在算什么」的定位一致。等需要缓存中间统计量（比如在线 softmax，见下文）
//  时才值得变成类。
//
//  数学定义：softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
//  一个致命的实现陷阱 —— **数值溢出**：
//   如果 x_i 很大（比如注意力分数能到几百上千），exp(x_i) 直接变成 inf，
//   inf/inf = NaN，全废。而 exp 对「整体平移」是不变的：
//      softmax(x)_i = exp(x_i - c) / Σ_j exp(x_j - c)   （任意常数 c）
//   所以标准做法是先减去最大值 c = max(x)，让最大的指数变成 exp(0) = 1，
//   既防溢出又防下溢（最小项也不至于全变 0 而丢精度）。
//  这就是为什么几乎所有实现（PyTorch、flash-attn 的 online softmax）都要
//  先求 max 再 exp —— 减 max 那步不是多余，是保命符。
// ============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "matrix.h"

// ----------------------------------------------------------------------------
// softmax(std::vector) —— 一维向量版，返回归一化后的概率分布
// 保证：输出各项 ∈ (0,1)，且和为 1；输入全等时退化为均匀分布。
// ----------------------------------------------------------------------------
template <typename T>
std::vector<T> softmax(const std::vector<T>& x) {
    if (x.empty()) return {};

    // 数值稳定性：先减去最大值，见文件头注释
    const T maxVal = *std::max_element(x.begin(), x.end());

    std::vector<T> e(x.size());
    T sum = T(0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        e[i] = std::exp(x[i] - maxVal);
        sum += e[i];
    }
    for (T& v : e) v /= sum;   // 归一化：除以所有 exp 之和
    return e;
}

// ----------------------------------------------------------------------------
// softmax_rows(Matrix) —— 按行 softmax（attention 的标准用法）
// 每行是一个独立的概率分布：对 attention score 矩阵，第 i 行就是第 i 个
// query 对全部 key 的注意力权重，行和必须为 1。
// 返回一个新矩阵，不修改入参（eager 语义，与 Matrix.h 一致）。
// ----------------------------------------------------------------------------
template <typename T>
Matrix<T> softmax_rows(const Matrix<T>& x) {
    Matrix<T> r(x.rows(), x.cols());
    const T* xd = x.data();
    T* rd = r.data();
    const std::size_t cols = x.cols();

    for (std::size_t i = 0; i < x.rows(); ++i) {
        const T* row = xd + i * cols;
        T* out = rd + i * cols;

        // 1) 求行最大值（用于数值稳定）
        T maxVal = row[0];
        for (std::size_t j = 1; j < cols; ++j) maxVal = std::max(maxVal, row[j]);

        // 2) exp(x - max) 并累加分母
        T sum = T(0);
        for (std::size_t j = 0; j < cols; ++j) {
            out[j] = std::exp(row[j] - maxVal);
            sum += out[j];
        }

        // 3) 归一化
        for (std::size_t j = 0; j < cols; ++j) out[j] /= sum;
    }
    return r;
}

// 在线 softmax（flash-attention 用的就是它）：如果你想深入，
// 思路是「边扫边维持 running max 和 running sum」，这样注意力矩阵可以
// 分块流式处理、不落内存。那是下一个教学主题，这里先不展开。
