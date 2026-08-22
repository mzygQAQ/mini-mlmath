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

auto or_gate(const Matrix<float> &dataset) -> Matrix<float> {
    CHECK(dataset.cols() == 2) << "OR gate expects 2 inputs per row";
    CHECK(dataset.rows() > 0) << "dataset must not be empty";

    const Matrix<float> feature = dataset.with_ones_column();
    const Matrix<float> weight = {{0.5, 0.5, -0.4}};
    const Matrix<float> scores = feature * weight.transposed();

    // sign：分数 > 0 判 1，否则 0（分数恰好为 0 落在决策线上，归 0）
    Matrix<float> pred(dataset.rows(), 1);
    for (std::size_t i = 0; i < dataset.rows(); ++i)
        pred(i, 0) = scores(i, 0) > 0.0f ? 1.0f : 0.0f;
    return pred;
}

// NOT AND，就是 AND 的输出取反：只有两个输入都是 1 时输出 0，其余三种情况都输出 1。
// 0 0 -> 1
// 0 1 -> 1
// 1 0 -> 1
// 1 1 -> 0
auto nand_gate(const Matrix<float> &dataset) -> Matrix<float> {
    CHECK(dataset.cols() == 2) << "NAND gate expects 2 inputs per row";
    CHECK(dataset.rows() > 0) << "dataset must not be empty";

    const Matrix<float> feature = dataset.with_ones_column();
    const Matrix<float> weight = {{-1.0f, -1.0f, 1.1f}};
    const Matrix<float> scores = feature * weight.transposed();

    // sign：分数 > 0 判 1，否则 0（分数恰好为 0 落在决策线上，归 0）
    Matrix<float> pred(dataset.rows(), 1);
    for (std::size_t i = 0; i < dataset.rows(); ++i)
        pred(i, 0) = scores(i, 0) > 0.0f ? 1.0f : 0.0f;
    return pred;
}

// XOR（异或）：两个输入**不同**时输出 1，相同输出 0。
// 0 0 -> 0
// 0 1 -> 1
// 1 0 -> 1
// 1 1 -> 0        （注意：不是 OR！只有 (1,1) 这一行和 OR 不同）
//
// 为什么单层感知机（一条直线）学不会 XOR？—— 线性不可分，三重视角：
//
// 1) 几何直觉（画图一眼看出）：
//    把四个点画在平面上，(0,0) 和 (1,1) 是「对角」两个顶点、同为 0 类，
//    (0,1) 和 (1,0) 是另外两个对角顶点、同为 1 类。两类点互相穿插，
//    任何一条直线只能把平面切成两半，永远无法把「对角的两组」分开。
//    对比 AND/OR/NAND：它们的 0 类和 1 类各自聚在直线某一侧，一条线够用。
//
// 2) 代数证明（把真值表写成不等式组，看它无解）：
//    设决策线 w1*x1 + w2*x2 + b = 0，> 0 判 1。逐行代入：
//      (0,0)->0:   b <= 0
//      (0,1)->1:   w2 + b > 0
//      (1,0)->1:   w1 + b > 0
//      (1,1)->0:   w1 + w2 + b <= 0
//    中间两式相加：w1 + w2 + 2b > 0；代入第四式（w1+w2 <= -b）得
//      -b + 2b > 0  =>  b > 0
//    与第一式 b <= 0 矛盾 —— 不等式组无解，不存在任何 (w1, w2, b)
//    能分对四个点。这就是「线性不可分」的严格定义。
//
// 3) 收敛定理视角（考试常考）：
//    感知机收敛定理（Novikoff 1962）的前提是「数据线性可分」；XOR 不可分，
//    所以训练时权重永远震荡、不收敛。这正是 Minsky & Papert 1969 年证明
//    感知机局限、导致神经网络第一次寒冬的著名反例。
//
// 解法：加隐藏层（MLP）。XOR = (x1 OR x2) AND NAND(x1, x2)：
//   隐藏层两个神经元分别学 OR 和 NAND，输出层用 AND 组合 ——
//   隐藏层先把输入非线性地「掰」到可分的位置，多层网络的意义就在这。
auto xor_gate(const Matrix<float> &dataset) -> Matrix<float> {

    Matrix<float> res;
    return res;
}

int main() {
    // test_and：真值表全 4 个组合
    Matrix<float> dataset = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
    };

    // AND
    const auto pred = and_gate(dataset);
    for (std::size_t i = 0; i < dataset.rows(); ++i)
        std::cout << "AND(" << dataset(i, 0) << ", " << dataset(i, 1)
                  << ") = " << pred(i, 0) << "\n";

    // OR
    const auto or_pred = or_gate(dataset);
    for (std::size_t i = 0; i < dataset.rows(); ++i)
        std::cout << "OR(" << dataset(i, 0) << ", " << dataset(i, 1)
                  << ") = " << or_pred(i, 0) << "\n";

    // NAND
    const auto nand_pred = nand_gate(dataset);
    for (std::size_t i = 0; i < dataset.rows(); ++i)
        std::cout << "NAND(" << dataset(i, 0) << ", " << dataset(i, 1)
                  << ") = " << nand_pred(i, 0) << "\n";
    return 0;
}
