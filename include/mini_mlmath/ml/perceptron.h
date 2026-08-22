// ============================================================================
//  perceptron.h —— 感知机 Perceptron（header-only，随 Matrix.h 一起用）
//
//  干的事（对应 scikit-learn 的 Perceptron）：给一批带标签的样本 (x_i, y_i)，
//  学一个**线性二分类器**：一条（超）平面把两类样本分开。
//
//  为什么感知机值得单独一个头文件？
//    - 它是「神经网络」的祖宗：单个神经元 = 加权求和 + 阶跃激活，
//      MLP / 深度学习里的权重、偏置、激活、梯度更新，全都能在感知机上
//      找到最朴素的原型，理解它等于理解神经网络的地基。
//    - 它是「在线学习」（每看一个样本就更新一次）的最简例子，
//      和「扫完全部样本再更新」的 batch 梯度下降正好形成对照。
//
//  数学定义（经典 Rosenblatt 感知机，1958）：
//    决策函数（模型输出）:   f(x) = w·x + b
//    预测标签:                y_hat = sign(f(x))，取 +1 / -1
//    更新规则（只在**预测错误**时更新，即 y_i * f(x_i) <= 0 时）:
//        w <- w + lr * y_i * x_i
//        b <- b + lr * y_i
//    直觉：判错了就把权重往「让这次输入得分更高/更低」的方向掰一下，
//    掰的幅度正比于学习率 lr。经典感知机取 lr = 1 即可（收敛性与 lr 大小
//    无关，只有符号起作用 —— 这是它和梯度下降最大的不同）。
//
//  收敛性（重要结论，考试常考）：
//    若训练数据**线性可分**，感知机在有限步内收敛（感知机收敛定理，
//    Novikoff 1962）；若不可分，算法不收敛、权重会一直震荡 ——
//    这也是后来 SVM（找最大间隔）和逻辑回归（软化阶跃）出现的原因之一。
//
//  约定（与 VarianceThreshold 一致）：
//    - 数据布局：Matrix<T>，每行 = 一个样本，每列 = 一个特征
//    - 标签：std::vector<T>，取值 **+1 / -1**（sign 的定义域；
//      如果你的标签是 0/1，调用前先自行映射成 ±1）
//    - 权重初始化为全 0（经典做法，sklearn 也默认从 0 开始）
//
//  用法：
//    Perceptron<double> p(/* learning_rate = */ 1.0, /* max_iter = */ 100);
//    p.fit(X_train, y_train);                // X_train: n×d，y_train: n 个 ±1
//    auto pred   = p.predict(X_test);        // 返回 ±1 的预测标签
//    auto scores = p.decision_function(X_test); // 原始分数 w·x + b（画决策边界用）
//
//  延伸（本文件不实现，留给感兴趣的读者）：
//    - partial_fit（在线学习）：训练集很大时不用重训，来一个样本更新一次，
//      sklearn 的 Perceptron.partial_fit 就是干这个的；
//    - 打乱样本顺序（shuffle）：感知机的收敛路径依赖样本顺序，随机化
//      通常收敛更快（sklearn 默认 shuffle=True）。
// ============================================================================
#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "mini_mlmath/check.h"
#include "mini_mlmath/matrix.h"

template <typename T>
class Perceptron {
public:
    using value_type = T;
    using size_type = std::size_t;

    // ---- 构造 ----

    // learning_rate：更新步长，默认 1.0（经典取值，见文件头更新规则）；
    // max_iter：最多迭代多少**轮**（每轮 = 完整扫一遍全部样本）。
    explicit Perceptron(T learning_rate = T(1), size_type max_iter = 1000);

    // ---- 训练阶段 ----

    // 在样本 X（n 行 × d 列）和标签 y（n 个，取值 ±1）上训练，
    // 把学到的参数写进 weights_ / bias_。返回 *this 便于链式。
    // 实现步骤见类外定义处的 TODO。
    Perceptron& fit(const Matrix<T>& X, const std::vector<T>& y);

    // ---- 推理阶段 ----

    // 决策函数：对 X 的每一行返回原始分数 w·x + b（predict 就是取它的符号）。
    // 单独暴露出来，方便画决策边界 / 看样本到超平面的距离。
    // X 的行数任意，列数必须等于 fit 时的特征数 d。
    std::vector<T> decision_function(const Matrix<T>& X) const;

    // 对 X 的每一行返回预测标签（+1 / -1）。
    std::vector<T> predict(const Matrix<T>& X) const;

    // ---- 查询 ----

    // 学到的权重 w 的副本（长度 = 特征数 d；fit 之前为空）
    std::vector<T> weights() const { return weights_; }

    // 学到的偏置 b
    T bias() const { return bias_; }

    // 是否已 fit 过（predict / decision_function 之前必须为 true）
    bool fitted() const { return fitted_; }

private:
    T learning_rate_ = T(1);    // 学习率
    size_type max_iter_ = 1000; // 最大迭代轮数
    std::vector<T> weights_;    // 权重，fit 后长度 = 特征数 d
    T bias_ = T(0);             // 偏置，fit 后学出
    bool fitted_ = false;       // 是否已 fit 过
};

// ============================================================================
//  类外实现（模板必须留在头文件里，不能拆 .cpp —— 每个翻译单元都要看到定义
//  才能实例化）。以下是**空骨架**：函数体只有 TODO 注释和占位的 throw，
//  算法步骤写在注释里，实现留给你自己写（写完记得删掉 throw）。
// ============================================================================

template <typename T>
Perceptron<T>::Perceptron(T learning_rate, size_type max_iter)
    : learning_rate_(learning_rate), max_iter_(max_iter) {}

template <typename T>
Perceptron<T>& Perceptron<T>::fit(const Matrix<T>& X, const std::vector<T>& y) {
    // TODO(你)：实现感知机训练，建议步骤：
    //  1) 校验：X 至少 1 行 1 列；y.size() == X.rows()（用 CHECK，见 check.h）；
    //  2) 初始化：weights_ 清空后填 d 个 0（d = X.cols()），bias_ = 0；
    //  3) 外层循环 max_iter_ 轮，内层遍历每个样本 i：
    //        score = w·x_i + b
    //        若 y_i * score <= 0（预测错了）就更新：
    //           weights_[j] += learning_rate_ * y_i * X(i, j)   （对每个 j）
    //           bias_        += learning_rate_ * y_i
    //     （经典感知机：只在出错时更新，见文件头）
    //  4) fitted_ = true，返回 *this。
    throw std::logic_error("Perceptron::fit() not implemented yet");
}

template <typename T>
std::vector<T> Perceptron<T>::decision_function(const Matrix<T>& X) const {
    // TODO(你)：实现决策函数，建议步骤：
    //  1) 校验：fitted_ 必须为 true（先 fit 再推理）；
    //     X.cols() == weights_.size()（列数必须和 fit 时一致）；
    //  2) 对每一行算 w·x_i + b，返回长度 = X.rows() 的分数向量。
    throw std::logic_error("Perceptron::decision_function() not implemented yet");
}

template <typename T>
std::vector<T> Perceptron<T>::predict(const Matrix<T>& X) const {
    // TODO(你)：实现预测，建议步骤：
    //  1) 先调用 decision_function(X) 拿到每行分数（或直接复用其逻辑）；
    //  2) 对每个分数取符号：score > 0 判 +1，否则判 -1
    //     （score == 0 恰好落在平面上，归到哪类自己定，注释里写清楚即可）。
    throw std::logic_error("Perceptron::predict() not implemented yet");
}
