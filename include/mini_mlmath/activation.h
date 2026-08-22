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
//  数学定义：sigmoid(z) = 1 / (1 + e^(-z))
//    值域 (0,1)；z → +∞ 趋近 1，z → -∞ 趋近 0，z = 0 时正好 0.5。
//    所以「sigmoid(z) > 0.5」等价于「z > 0」—— 和 sign 的决策边界
//    完全一致，只是把跳变换成了平滑过渡，还多了个概率解释。
//
//  数值稳定性（和 softmax.h 同一类陷阱，务必记住）：
//    直接写 1/(1+exp(-z))，当 z 很负（如 -1000）时 exp(1000) 上溢成 inf，
//    1/(1+inf) = 0 碰巧结果还对，但中间值爆了。标准做法是分段：
//      z >= 0: 1/(1+exp(-z))      —— exp(-z) 落在 (0,1]，安全
//      z <  0: exp(z)/(1+exp(z))  —— exp(z) 落在 (0,1)，安全
//    两段数学上恒等，但保证中间值永不溢出。PyTorch / numpy 都是这么干的。
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
