// ============================================================================
//  logic_gate.cpp —— 用感知机学逻辑门（mini-mlmath 的感知机测试程序）
//
//  逻辑门是感知机最经典的入门测试（《统计学习方法》例 2.1 就是 AND 门）：
//    输入两个 0/1 值 x1, x2，学一个模型输出 0 或 1。
//
//  值得验证的四件事（关键结论先写在注释里，跑通后对照）：
//    1. AND 门：线性可分，感知机一定能学会（x1 & x2，只有 (1,1) 输出 1）
//    2. OR  门：线性可分，感知机一定能学会（x1 | x2，只有 (0,0) 输出 0）
//    3. NAND 门：线性可分（AND 的反相，同样能学）
//    4. XOR 门：**线性不可分**，感知机学不会（会一直震荡不收敛）——
//       这正是当年 Minsky & Papert 证明感知机局限、导致神经网络
//       第一次寒冬的著名反例，也是后来要加隐藏层（MLP）的原因。
//
//  提示：本库 Perceptron 的标签约定是 +1/-1，所以 0/1 标签
//  要先映射成 -1/+1（见 ml/perceptron.h 头注释）。
//
//  实现与原理见 ml/perceptron.h 头部注释。
// ============================================================================
//#include "mini_mlmath/ml/perceptron.h"

#include "mini_mlmath/matrix.h"
#include "mini_mlmath/check.h"

#include <iostream>

// and_gate：AND 门前向验证（固定手写参数，不训练；等 Perceptron::fit
// 写完后可换成学出来的权重）。
//
// 参数：w = (0.5, 0.5)，bias = -0.75，即决策线 x1 + x2 = 1.5，手算核对：
//   (0,0): 0.5*0 + 0.5*0 - 0.75 = -0.75 < 0 -> 0
//   (0,1): 0.5*0 + 0.5*1 - 0.75 = -0.25 < 0 -> 0
//   (1,0): 0.5*1 + 0.5*0 - 0.75 = -0.25 < 0 -> 0
//   (1,1): 0.5*1 + 0.5*1 - 0.75 =  0.25 > 0 -> 1
//
// bias folding：X 增广一列 1（with_ones_column），bias 变成权重第三分量，
// 整个前向就是一次矩阵乘，没有任何标量加法。
// 注意这里用 sign（分数>0 判 1）而非 sigmoid：单层感知机不反向传播、
// 不需要可导激活，经典感知机本来就用阶跃函数。sigmoid 等可导激活
// 留给多层网络（activation.h 里已备好）。
auto and_gate(const Matrix<float> &dataset) -> Matrix<float> {
    CHECK(dataset.cols() == 2) << "AND gate expects 2 inputs per row";
    CHECK(dataset.rows() > 0) << "dataset must not be empty";

    const Matrix<float> feature = dataset.with_ones_column(); // n×3: [x1, x2, 1]
    const Matrix<float> weight({{0.5f, 0.5f, -0.75f}});       // 1×3，第三分量 = bias
    const Matrix<float> scores = feature * weight.transposed(); // n×1，纯矩阵乘

    // sign：分数 > 0 判 1，否则 0（分数恰好为 0 落在决策线上，归 0）
    Matrix<float> pred(dataset.rows(), 1);
    for (std::size_t i = 0; i < dataset.rows(); ++i)
        pred(i, 0) = scores(i, 0) > 0.0f ? 1.0f : 0.0f;
    return pred;
}

int main() {
    // test_and：真值表全 4 个组合
    Matrix<float> dataset = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
    };
    const auto pred = and_gate(dataset);
    for (std::size_t i = 0; i < dataset.rows(); ++i)
        std::cout << "AND(" << dataset(i, 0) << ", " << dataset(i, 1)
                  << ") = " << pred(i, 0) << "\n";
    return 0;
}
