// ============================================================================
//  variance_threshold.h —— 基于方差的特征选择（header-only，随 Matrix.h 一起用）
//
//  干的事（对应 scikit-learn 的 VarianceThreshold）：
//    feature selection 里最朴素的一招 —— 把「取值几乎不变」的特征丢掉。
//    方差越小，说明这个特征的所有样本挤在同一点附近，携带的信息量越低；
//    方差低于阈值的列，对模型基本没贡献，删掉还能降维、防过拟合。
//
//  为什么是「类」而不是「自由函数」？（对照 softmax.h 里那句「无状态运算
//  才用自由函数」）
//    softmax 是无状态纯函数：算完即走。而 VarianceThreshold 自带状态：
//    fit 阶段算出的「每列方差」和「哪些列被选中」要存下来，供后续
//    transform 反复使用 —— 这就是需要缓存中间统计量的场景，值得做成类。
//
//  和 sklearn 的 API 对齐（教学版，约定与 Matrix.h 一致）：
//    - 数据布局：Matrix<T>，每行 = 一个样本，每列 = 一个特征
//    - 无宏、无表达式模板、eager 求值
//
//  数学定义（每个特征列独立算）：
//    var_j = (1/n) Σ_i (X(i,j) - mean_j)²      （n = 样本数）
//    保留所有 var_j >= threshold 的列。
//
//  用法：
//    VarianceThreshold<double> vt(0.1);
//    vt.fit(X);                       // 算出每列方差，存进 variances_
//    auto Y = vt.transform(X);        // 只保留方差达标的列
//    // 或一步到位：auto Y = vt.fit_transform(X);
// ============================================================================
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mini_mlmath/check.h"
#include "mini_mlmath/matrix.h"

template<typename T>
class VarianceThreshold {
public:
    using value_type = T;
    using size_type = std::size_t;

    // ---- 构造 ----

    // threshold：方差低于该值的特征列将被剔除，默认 0.0（sklearn 语义：
    // 默认只删「方差为 0」的常数列）
    explicit VarianceThreshold(T threshold = T(0)) : threshold_(threshold) {}

    // ---- 训练阶段 ----

    // 对样本矩阵 X（n 行 × d 列）逐列算方差，存进 variances_；
    // 并按 threshold_ 决定哪些列保留（记录到 support_，即 sklearn 的
    // get_support()）。返回 *this 便于链式。实现自己写。
    VarianceThreshold &fit(const Matrix<T> &X);

    // 训练 + 一次性变换，一步到位
    Matrix<T> fit_transform(const Matrix<T> &X);

    // ---- 推理阶段 ----

    // 对任意矩阵 X（列数必须和 fit 时一致），只保留被选中的列返回
    Matrix<T> transform(const Matrix<T> &X) const;

    // ---- 查询 ----

    // 每列方差的副本（长度 = fit 时的特征数 d）
    std::vector<T> variances() const { return variances_; }

    // 掩码：第 j 列为非 0 表示该列被保留（长度 = d）。
    std::vector<std::uint8_t> get_support() const { return support_; }

private:
    T threshold_ = T(0);        // 方差阈值
    bool fitted_ = false;       // 是否已 fit 过（transform 前必须为 true）
    std::vector<T> variances_;  // 每列方差，fit 后填充
    std::vector<std::uint8_t> support_; // 每列是否保留（1/0），fit 后填充
};

// ============================================================================
//  类外实现（模板必须留在头文件里，不能拆 .cpp —— 每个翻译单元都要看到定义
//  才能实例化）。以下是空骨架，待你自己填实现。
// ============================================================================

template<typename T>
VarianceThreshold<T> &VarianceThreshold<T>::fit(const Matrix<T> &X) {
    // 1) 校验：X 至少有一列（n 行 × d 列）
    const auto n_samples = X.rows();
    const auto n_features = X.cols();
    CHECK(n_samples > 0 && n_features > 0) << "must have one features/samples";

    support_.clear();
    variances_.clear();

    // 2) 逐列算方差，填入 variances_：var_j = (1/n) Σ_i (X(i,j) - mean_j)²
    for (std::size_t fet = 0; fet < n_features; ++fet) {
        T total = {0};
        for (std::size_t sam = 0; sam < n_samples; ++sam) {
            total += X(sam, fet);
        }

        T mean = total / static_cast<T>(n_samples);
        T sum_sq = {0};
        for (std::size_t sam = 0; sam < n_samples; ++sam) {
            const T diff = X(sam, fet) - mean;
            sum_sq += diff * diff;
        }
        const T var = sum_sq / static_cast<T>(n_samples);
        variances_.push_back(var);
        support_.push_back(var > threshold_ ? 1 : 0);
    }

    fitted_ = true;
    return *this;
}

template<typename T>
Matrix<T> VarianceThreshold<T>::transform(const Matrix<T> &X) const {
    // 必须先用 fit 学出筛选规则，才能谈「套用到新数据」
    CHECK(fitted_) << "must call fit() before transform()";

    // 校验 X.cols() == support_.size()（列数必须和 fit 时一致）
    CHECK(X.cols() == support_.size()) << "X must be have same cols with the train dataset";

    // 新建输出矩阵：行数不变，列数 = support_ 中非 0 的个数
    const std::size_t n_kept =
            static_cast<std::size_t>(std::count_if(support_.begin(), support_.end(),
                                                   [](std::uint8_t s) { return s != 0; }));
    Matrix<T> Y(X.rows(), n_kept);

    // 逐列搬运被选中的列，返回。注意 new_col 只在选中列时自增，
    // 否则没选中的列会把新列位置顶偏（错位/越界）。
    for (std::size_t row = 0; row < X.rows(); ++row) {
        std::size_t new_col = 0;
        for (std::size_t col = 0; col < X.cols(); ++col) {
            if (support_[col] != 0) {
                Y(row, new_col++) = X(row, col);
            }
        }
    }

    return Y;
}

template<typename T>
Matrix<T> VarianceThreshold<T>::fit_transform(const Matrix<T> &X) {
    return fit(X).transform(X);
}


