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
//    - 权重随机初始化：每个权重 ~ U(-0.5, 0.5)，用成员里的 Random 实例生成
//      （默认种子 42，可复现；想换种子可改 Random rand_ 字段或外部传 seed）
//    - 偏置 b 初始化为 0
//    - 注：经典 Rosenblatt 感知机其实从 0 开始（更新规则 w += lr·y·x 是
//      加性的，从 0 起步也能让算法收敛）；这里用随机初始化是为了在
//      Perceptron 演进到 MLP / 深层模型时不需要再改代码 —— 深度学习里
//      随机初始化是必要的（避免对称性），现在就把 API 摆好。
//
//  用法：
//    Perceptron<> p(/* learning_rate = */ 1.0f, /* max_iter = */ 100);  // 默认 T=float
//    // 或显式指定：Perceptron<double> p(1.0, 100);  // 需要 double 精度时
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
#include <type_traits>
#include <vector>

#include "mini_mlmath/check.h"
#include "mini_mlmath/random.h"
#include "mini_mlmath/matrix.h"

// 约束：Perceptron 的 T 必须是浮点类型（float / double / long double），
// 整数类型会在编译期就报错。原因和 random.h 一样：int 权重会让
// `int(-0.5) = int(0.5) = 0` 静默退化，weights 全 0。
// 用类内 static_assert，错误指向 Perceptron 自己的实例化点，
// 比 random 里的 static_assert 更早、更明显。
//
// T 默认 float：感知机算的是 w·x + b 这种简单线性运算，不需要 double 精度；
// float 内存和带宽都更省，GPU 友好。需要 double 时显式写 Perceptron<double>。
template<typename T = float>
class Perceptron {
    static_assert(std::is_floating_point_v<T>,
                  "Perceptron<T> requires floating-point T "
                  "(float / double / long double). int weights silently "
                  "truncate to 0 in random init — use T=double (recommended) or T=float.");

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
    Perceptron &fit(const Matrix<T> &X, const std::vector<T> &y);

    // ---- 推理阶段 ----

    // 决策函数：对 X 的每一行返回原始分数 w·x + b（predict 就是取它的符号）。
    // 单独暴露出来，方便画决策边界 / 看样本到超平面的距离。
    // X 的行数任意，列数必须等于 fit 时的特征数 d。
    std::vector<T> decision_function(const Matrix<T> &X) const;

    // 对 X 的每一行返回预测标签（+1 / -1）。
    std::vector<T> predict(const Matrix<T> &X) const;

    // ---- 查询 ----

    // 学到的权重 w 的副本（长度 = 特征数 d；fit 之前为空）
    std::vector<T> weights() const { return weights_; }

    // 学到的偏置 b
    T bias() const { return bias_; }

    // 是否已 fit 过（predict / decision_function 之前必须为 true）
    bool fitted() const { return fitted_; }

private:
    Random rand_;
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

template<typename T>
Perceptron<T>::Perceptron(T learning_rate, size_type max_iter)
        : learning_rate_(learning_rate), max_iter_(max_iter) {}

template<typename T>
Perceptron<T> &Perceptron<T>::fit(const Matrix<T> &X, const std::vector<T> &y) {
    //  1) 校验：X 至少 1 行 1 列；y.size() == X.rows()（用 CHECK，见 check.h）；
    CHECK(y.size() == X.rows());
    CHECK(X.cols() >= 1) << "Perceptron::fit: need at least 1 feature column";

    //  2) 初始化：bias_ = 0；weights_ 长度 d = X.cols()，每个元素 ~ U(-0.5, 0.5)
    //     （用成员里的 Random rand_：默认种子 42，实验可复现）
    bias_ = T(0);
    weights_.resize(X.cols());
    for (size_type j = 0; j < X.cols(); ++j) {
        weights_[j] = rand_.uniform<T>(T(-0.5), T(0.5));
    }

    //  3) Bias folding：把 weights_ 和 bias_ 拼成 (d+1) 维的「增广权重」，
    //     同时给 X 右边拼一列 1（with_ones_column）。这样：
    //       - 算分数 = features * w_aug，一次矩阵乘搞定
    //       - bias 更新自动并入 w_aug[d]（因为 features(i, d) = 1）
    //       - 不再需要单独管 bias，代码更干净
    std::vector<T> w_aug = weights_;
    w_aug.push_back(bias_);                                // 长度 d+1，最后一个是 bias
    Matrix<T> features = X.with_ones_column();             // n × (d+1)

    //  4) 外层循环 max_iter_ 轮。每轮：
    //       - 用当前 w_aug 算所有样本的分数（一次矩阵乘）
    //       - 逐样本检查 y_i * score_i > 0 是否成立
    //           - > 0：正确，啥也不做
    //           - <= 0：错，更新 w_aug 的所有 d+1 个分量
    //       - 更新规则对所有 j 统一：w_aug[j] += lr * y_i * features(i, j)
    //         j=d 时 features(i, d) = 1，所以 bias 更新自动包含在内
    for (size_type epoch = 0; epoch < max_iter_; ++epoch) {
        Matrix<T> scores = features * Matrix<T>::from_column(w_aug);  // n×1
        for (size_type i = 0; i < X.rows(); ++i) {
            // y_i × score_i > 0 → 正确；<= 0 → 错
            if (y[i] * scores(i, 0) <= T(0)) {
                for (size_type j = 0; j < features.cols(); ++j) {
                    w_aug[j] += learning_rate_ * y[i] * features(i, j);
                }
            }
        }
    }

    //  5) 训练完，把 w_aug 拆回 weights_ 和 bias_（最后一位是 bias）
    bias_ = w_aug.back();
    w_aug.pop_back();
    weights_ = w_aug;

    //  6) 设置训练完毕
    fitted_ = true;
    return *this;
}

template<typename T>
std::vector<T> Perceptron<T>::decision_function(const Matrix<T> &X) const {
    CHECK(fitted_) << "must call fit() before decision_function()";
    CHECK(X.cols() == weights_.size())
            << "Perceptron::decision_function: feature count mismatch, got "
            << X.cols() << " cols but trained on " << weights_.size();

    // 用 bias folding 一次矩阵乘算所有分数（复用 fit 里的模式）
    std::vector<T> w_aug = weights_;
    w_aug.push_back(bias_);
    Matrix<T> features = X.with_ones_column();
    Matrix<T> scores = features * Matrix<T>::from_column(w_aug);  // n×1

    // 拉平 n×1 → std::vector
    std::vector<T> result(X.rows());
    for (size_type i = 0; i < X.rows(); ++i) result[i] = scores(i, 0);
    return result;
}

template<typename T>
std::vector<T> Perceptron<T>::predict(const Matrix<T> &X) const {
    // 取 decision_function 的符号。score == 0 落在平面上，归 -1
    // （和 sklearn 的 Perceptron 行为一致：默认 0 归负类）
    std::vector<T> scores = decision_function(X);
    std::vector<T> result(X.rows());
    for (size_type i = 0; i < X.rows(); ++i) {
        result[i] = scores[i] > T(0) ? T(1) : T(-1);
    }
    return result;
}
