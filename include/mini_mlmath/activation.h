// ============================================================================
//  activation.h —— 激活函数（header-only，随 Matrix.h 一起用）
//
//  干的事：把「线性分数 z = w·x + b」变成神经元的非线性输出。
//  感知机原始用的是阶跃函数 sign(z)（输出非 0 即 1，不可导）；
//  但多层网络训练要反向传播、要算梯度，阶跃函数在 0 处不可导，
//  MLP 时代改用**可导**的平滑激活函数 —— sigmoid 是第一个，也是
//  和「概率」联系最直接的一个（输出 ∈ (0,1)，可以直接当概率看）。
//  之后你会遇到的 tanh、ReLU 都是同一族：可导的非线性函数。
//
//  sigmoid — 平滑 S 形，输出 ∈ (0,1)
//    定义：σ(z) = 1 / (1 + e^(-z))
//    值域 (0,1)；z → +∞ 趋近 1，z → -∞ 趋近 0，z = 0 时正好 0.5。
//    所以「sigmoid(z) > 0.5」等价于「z > 0」—— 和 sign 的决策边界
//    完全一致，只是把跳变换成了平滑过渡，还多了个概率解释。
//
//    数值稳定性（和 softmax.h 同一类陷阱，务必记住）：
//      直接写 1/(1+exp(-z))，当 z 很负（如 -1000）时 exp(1000) 上溢成 inf，
//      1/(1+inf) = 0 碰巧结果还对，但中间值爆了。标准做法是分段：
//        z >= 0: 1/(1+exp(-z))      —— exp(-z) 落在 (0,1]，安全
//        z <  0: exp(z)/(1+exp(z))  —— exp(z) 落在 (0,1)，安全
//      两段数学上恒等，但保证中间值永不溢出。PyTorch / numpy 都是这么干的。
//
//  ReLU — 修正线性单元，**现代深度学习的默认激活**
//    定义：relu(z) = max(0, z)              （把负数「砍掉」留正数）
//    值域 [0, +∞)；z > 0 时斜率恒为 1，z < 0 时输出恒为 0。
//    求导：relu'(z) = 1 if z > 0, else 0  （z = 0 处按惯例取 0 即可）
//
//    为什么 sigmoid 后还要 ReLU？三个实操原因：
//      1) **不饱和** —— sigmoid 在 z 很大 / 很小时梯度趋近于 0（饱和区），
//         多层反向传播时这些小数连乘起来梯度会指数级衰减（vanishing gradient），
//         浅层权重几乎不动。ReLU 在 z > 0 一侧梯度恒为 1，不存在这个问题。
//      2) **计算便宜** —— 没有 exp()，就一次比较 + 一次取数，sigmoid 的
//         几次 exp + 除法在它面前是几十倍的开销。CNN / Transformer 这种
//         几十上百层的网络，激活函数调用是热点。
//      3) **稀疏性** —— 一半左右的输入（z < 0 那部分）直接变 0，等价于
//         「随机关掉一半神经元」，带来一定的正则化效果（事实上 ResNet /
//         GPT 的隐藏层清一色 ReLU/GELU，没人再用 sigmoid 当隐藏层激活）。
//
//    缺点：z < 0 那一侧梯度为 0，对应神经元「死掉」再也不会更新
//    （dying ReLU 问题）。后续改进：LeakyReLU（负数区给个小斜率 0.01）、
//    PReLU（斜率也学）、GELU（GPT 用）、SiLU/Swish。本文件不实现这些。
// ============================================================================
#pragma once

#include <cmath>
#include <cstddef>

#include "mini_mlmath/matrix.h"

// sigmoid 标量版：单个数 z -> (0,1)。分段求值防溢出，见文件头注释。
template <typename T>
T sigmoid(T z) {
    if (z >= T(0)) {
        const T e = std::exp(-z);
        return T(1) / (T(1) + e);
    } else {
        const T e = std::exp(z);
        return e / (T(1) + e);
    }
}

// sigmoid 矩阵版：逐元素应用，返回新矩阵（eager，不改入参）。
// 多层网络里就是对「隐藏层线性分数 H = X·W1」整矩阵激活：
//   H_activated = sigmoid(H)
// 这就是 MLP 里「每个神经元先加权求和、再过激活」的矩阵化写法。
template <typename T>
Matrix<T> sigmoid(const Matrix<T>& m) {
    Matrix<T> r(m.rows(), m.cols());
    for (std::size_t i = 0; i < m.rows() * m.cols(); ++i)
        r.data()[i] = sigmoid(m.data()[i]);
    return r;
}

// ReLU 标量版：relu(z) = max(0, z)。
//   极简到「看过一次就不会忘」：正数原样放行，负数砍成 0。
//   不需要数值稳定技巧（max 不溢出），不需要分支以外的开销。
//   注：下面用 `z > 0` 而不是 `z >= 0`，让 z == 0 走「负数分支」返回 0，
//   这是 dying ReLU 边界处最常见的实现选择（relu'(0) = 0 的子梯度约定）。
//   浮点里 z 精确等于 0 极少触发，所以两种写法对实际训练没区别。
template <typename T>
T relu(T z) {
    return z > T(0) ? z : T(0);
}

// ReLU 矩阵版：逐元素应用，返回新矩阵（eager，不改入参）。
// 现代 CNN / Transformer 隐藏层激活的标配：
//   H_activated = relu(X · W1 + b1)
template <typename T>
Matrix<T> relu(const Matrix<T>& m) {
    Matrix<T> r(m.rows(), m.cols());
    for (std::size_t i = 0; i < m.rows() * m.cols(); ++i)
        r.data()[i] = relu(m.data()[i]);
    return r;
}