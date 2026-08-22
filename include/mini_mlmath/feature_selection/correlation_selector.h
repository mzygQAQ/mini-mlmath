// ============================================================================
//  correlation_selector.h —— 相关系数法特征选择（header-only，特征选择模块）
//
//  干的事：用「特征和目标变量的相关程度」来筛特征 —— 和 y 高度相关的特征
//  值得留，几乎无关的特征删掉。这比 VarianceThreshold（只看自身方差、不管
//  y）多利用了一层监督信息。
//
//  数学定义（皮尔逊相关系数，每列特征独立和 y 算）：
//      r_j = Σ_i (X(i,j) - x̄_j)(y_i - ȳ)
//            ──────────────────────────────────────────
//            sqrt( Σ_i (X(i,j)-x̄_j)²  ·  Σ_i (y_i-ȳ)² )
//    r ∈ [-1, 1]。|r| 越大线性关系越强；|r|≈0 说明该特征和 y 基本线性无关。
//    常数列（x̄_j 处所有点一样）分母为 0，r 无定义——应视为最无关，丢弃。
//
//  和成熟库的对应（API 模仿这些）：
//    sklearn:   SelectKBest(score_func=f_regression, k)  —— 按 F 统计量取 top-k
//    pandas:    df.corrwith(y)                            —— 逐列与 y 的相关度
//    本类：     CorrelationSelector(threshold)            —— |r| >= threshold 保留
//    和 sklearn 不同之处：sklearn 用 k（留前 k 个），这里用阈值（留「够相关」的），
//    语义更贴近「相关系数法」这个说法。想改成 top-k 自己加个参数即可。
//
//  和 VarianceThreshold 的关系（两者都在 feature_selection 模块）：
//    方差法    ：无监督，只看特征自身的波动（不需要 y）
//    相关系数法：有监督，看特征与目标的关联（必须给 y）
//    实际流程常先用方差法去常数列，再用相关系数法去无关列。
//
//  【可插拔算法：policy（策略）模式】
//    「算相关系数」是一个无状态运算，被抽成独立的策略类型（Pearson /
//    Spearman），只提供静态方法 measure(x, y)。选择器的骨架逻辑
//    （fit 框架 / transform / 阈值筛选）和具体算法无关，通过第二个模板参数
//    Measure 注入 —— 这就是 template trait / policy 的典型用法：
//        CorrelationSelector<double>              // 默认皮尔逊
//        CorrelationSelector<double, Spearman>    // 斯皮尔曼
//    想加新算法（如 Kendall τ、互信息），再写一个带 measure 的 struct 即可，
//    fit/transform 一行不用改。
//
//    关键洞察：斯皮尔曼相关系数 = 对 x、y 先各自取秩（rank），再在秩上算
//    皮尔逊。所以 Spearman::measure 内部复用 Pearson::measure，秩变换抽成
//    公共辅助函数 rank_vector —— 策略模式正好把这些关系表达出来。
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mini_mlmath/check.h"
#include "mini_mlmath/matrix.h"
#include "mini_mlmath/vector.h"

// ----------------------------------------------------------------------------
// 相关系数策略：每个都是无状态的 struct，只提供静态 measure(x, y)。
// ----------------------------------------------------------------------------

// 皮尔逊相关系数：直接用原始值算（见文件头公式）
struct Pearson {
    template<typename T>
    static T measure(const Vector<T> &x, const Vector<T> &y);
};

// 斯皮尔曼秩相关系数：x、y 先各自取秩，再对秩算皮尔逊。
// 对单调非线性关系也敏感（皮尔逊只对线性关系敏感）。
struct Spearman {
    template<typename T>
    static T measure(const Vector<T> &x, const Vector<T> &y);
};

// 秩变换辅助：把向量 v 的每个元素替换成它在全向量中的名次（秩）。
// 升序排名：最小值秩 1，最大值秩 n；并列（tie）取平均秩。自己实现。
template<typename T>
Vector<T> rank_vector(const Vector<T> &v) {
    return v;
}

// ----------------------------------------------------------------------------
// 相关系数选择器：骨架逻辑固定，算法由模板参数 Measure 注入
// ----------------------------------------------------------------------------
template<typename T, typename Measure = Pearson>
class CorrelationSelector {
public:
    using value_type = T;
    using size_type = std::size_t;

    // ---- 构造 ----

    // threshold：|r| 低于该值的特征列被剔除。默认 0.5（|r|>=0.5 通常算
    // 「中等偏强相关」，可改）。
    explicit CorrelationSelector(T threshold = T(0.5)) : threshold_(threshold) {}

    // ---- 训练阶段 ----

    // 对样本矩阵 X（n 行 × d 列）和目标向量 y（长度 n，每个样本一个标签）
    // 逐列调用 Measure::measure 算相关系数，存进 correlations_；并按
    // threshold_ 决定保留哪些列（记录到 support_）。返回 *this 便于链式。
    CorrelationSelector &fit(const Matrix<T> &X, const Vector<T> &y);

    // 训练 + 一次性变换（有监督版，入参多一个 y）
    Matrix<T> fit_transform(const Matrix<T> &X, const Vector<T> &y);

    // ---- 推理阶段 ----

    // 对任意矩阵 X（列数必须和 fit 时一致），只保留被选中的列返回
    Matrix<T> transform(const Matrix<T> &X) const;

    // ---- 查询 ----

    // 每列与 y 的相关系数（长度 = fit 时的特征数 d）
    std::vector<T> correlations() const { return correlations_; }

    // 掩码：第 j 列为非 0 表示该列被保留（长度 = d）
    std::vector<std::uint8_t> get_support() const { return support_; }

private:
    T threshold_ = T(0.5);      // |r| 阈值
    bool fitted_ = false;       // 是否已 fit 过（transform 前必须为 true）
    std::vector<T> correlations_;       // 每列相关系数，fit 后填充
    std::vector<std::uint8_t> support_; // 每列是否保留（1/0），fit 后填充
};

// ============================================================================
//  类外实现（模板必须留在头文件里）。以下是空骨架，待你自己填实现。
// ============================================================================

template<typename T>
T Pearson::measure(const Vector<T> &x, const Vector<T> &y) {
    // 1) CHECK(x.size() == y.size())
    // 2) 公式见文件头：r = Σ(x-x̄)(y-ȳ) / sqrt(Σ(x-x̄)² · Σ(y-ȳ)²)
    // 3) 分母为 0（任一是常数列）时 r 无定义，返回 T(0)（视为无关）
    // 4) 返回 r
    return T(0);
}

template<typename T>
T Spearman::measure(const Vector<T> &x, const Vector<T> &y) {
    // 斯皮尔曼 = 在秩上算皮尔逊：
    //     return Pearson::measure(rank_vector(x), rank_vector(y));
    return T(0);
}

template<typename T, typename Measure>
CorrelationSelector<T, Measure> &
CorrelationSelector<T, Measure>::fit(const Matrix<T> &X, const Vector<T> &y) {
    // 1) 校验：X 至少一行一列；y 长度必须等于 X.rows()
    // 2) 对每列：抽出列向量 col（n 个元素，col[i] = X(i, fet)），
    //    调 Measure::measure(col, y) 得 r_j，存入 correlations_
    // 3) support_[fet] = (|r_j| >= threshold_) ? 1 : 0
    // 4) fitted_ = true
    // 5) return *this;
    return *this;
}

template<typename T, typename Measure>
Matrix<T> CorrelationSelector<T, Measure>::fit_transform(const Matrix<T> &X,
                                                         const Vector<T> &y) {
    // 一步到位：fit(X, y) 后再 transform(X)
    return transform(X);
}

template<typename T, typename Measure>
Matrix<T> CorrelationSelector<T, Measure>::transform(const Matrix<T> &X) const {
    // 1) CHECK(fitted_) << "must call fit() before transform()"
    // 2) CHECK(X.cols() == support_.size())
    // 3) 行数不变，列数 = support_ 中非 0 的个数
    // 4) 逐列搬运被选中的列（new_col 只在选中列时自增，参考 variance_threshold.h）
    return X;
}
