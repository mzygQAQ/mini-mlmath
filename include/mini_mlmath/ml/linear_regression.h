// ============================================================================
//  linear_regression.h —— 线性回归 Linear Regression（header-only）
//
//  干的事（对应 scikit-learn 的 sklearn.linear_model.LinearRegression）：
//  给一批带连续标签的样本 (x_i, y_i)，学一个**线性回归模型**，把 y 近似
//  表示成特征的线性组合。最基础的监督学习模型之一：
//      f(x) = w · x + b
//  其中 w ∈ R^d 是每个特征的权重，b 是偏置（截距）。
//
//  为什么值得单独一个头文件？
//    - 它是「监督学习」的入门款：损失函数（均方误差 MSE / SSE）、凸优化、
//      解析解（正规方程 normal equation）、数值解（梯度下降）这一整套
//      套路，在它身上全部能以最简形式出现。
//    - 它是后面所有「线性模型」的底座：Ridge / Lasso / ElasticNet / 逻辑
//      回归…… 都是在它的损失函数上加正则项或换 link function。把这一个
//      啃透，后面只是「在同一个损失上加东西」。
//    - 它和感知机形成「回归 vs 分类」对照：模型形式几乎一样（都是
//      w·x + b），区别只在「预测什么」和「怎么学」。
//
//  数学定义：
//    模型：             f(x) = w · x + b   （b 称为 intercept / 偏置）
//    损失（残差平方和，Residual Sum of Squares, RSS / SSE）：
//                       L(w, b) = Σ_i ( f(x_i) - y_i )²
//    闭式解（正规方程，normal equation；本类始终用 bias folding）：
//      增广形式  X_aug = [X | 1]  (n × (d+1))
//      w_aug     = (X_aug^T · X_aug)^(-1) · X_aug^T · y   （长度 d+1）
//      取出      w = w_aug[0..d-1],  b = w_aug[d]
//    评指标（R²，决定系数，coefficient of determination）：
//      R² = 1 - SS_res / SS_tot
//         = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²
//      R²=1 表示完美拟合；R²=0 表示和「直接猜均值」一样差；R²<0 表示
//      比猜均值还差（通常意味着模型没学到东西或数据无信号）。
//
//  约定（与 Perceptron 一致）：
//    - 数据布局：Matrix<T>，每行 = 一个样本，每列 = 一个特征
//    - 标签：std::vector<T>，n 个连续值（不像感知机要求 ±1）
//    - 默认 T = float：线性回归就是乘加，float 足够；需要更高精度时
//      显式写 LinearRegression<double>。
//
//  用法（API 形状照搬 sklearn）：
//    LinearRegression<> lr;                       // 默认就 learn bias（对应
//                                                // sklearn fit_intercept=True）
//    lr.fit(X_train, y_train);                    // 学出 weights / bias
//    auto pred  = lr.predict(X_test);             // 返回 y_hat（连续值）
//    double r2  = lr.score(X_test, y_test);       // R²，1 表示完美拟合
//    auto w     = lr.weights();                   // std::vector<T>，长度 d
//    T b        = lr.bias();                      // 标量
//
//  与 sklearn 的字段对应（API 形状对齐，命名按本项目习惯）：
//    lr.weights() ↔  sklearn: lr.coef_           (ndarray, shape (n_features,))
//    lr.bias()    ↔  sklearn: lr.intercept_      (float；多输出时 ndarray)
//  注：本项目统一用 weight / bias（和 Perceptron 一致），sklearn 的
//  coef_ / intercept_ 偏统计学味道，是 sklearn 继承自历史命名，本教学
//  项目不背这个包袱。
//
//  注：本类**始终学偏置 b**（用 bias folding 把 b 并入权重一起解），不暴露
//  fit_intercept 开关 —— sklearn 的 fit_intercept=False（强制过原点）模式
//  这里不支持，要那种行为直接在外面把 X 中心化。
//
//  配套讲解：docs/linear_regression.md
//    - 模型结构 / 正规方程 / 梯度下降 / R² / bias folding
//    - 和感知机的「同结构、异输出」对照
//
//  留给你写的部分（核心训练 / 推理，TODO 已标在下面）：
//    - fit(...)        ：从 X / y 解出 weights / bias
//                       （建议走正规方程 / 伪逆；想练梯度下降也可以）
//    - predict(...)    ：对 X 的每行算 w·x + b
//    - score(...)      ：在 (X, y) 上算 R²（用 predict + 一行公式即可）
//    - 想挑战可加：Ridge 正则（(X^T X + λI)⁻¹ X^T y），用 SVD 解（更稳）
// ============================================================================
#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "mini_mlmath/check.h"
#include "mini_mlmath/matrix.h"

// 约束：LinearRegression 的 T 必须是浮点类型（float / double / long double）。
// 整数 T 会在编译期就报错 —— 原因和 Perceptron 一样：int 权重会让 MSE
// 退化（梯度是 Σ2x_i(y_i - ŷ_i)，int 累加极易溢出；解正规方程时 int
// 矩阵也没法做浮点除法），用类内 static_assert 把错误钉在实例化点。
template<typename T = float>
class LinearRegression {
    static_assert(std::is_floating_point_v<T>,
                  "LinearRegression<T> requires floating-point T "
                  "(float / double / long double). int will overflow / "
                  "truncate silently during gradient and normal-equation solve.");

public:
    using value_type = T;
    using size_type = std::size_t;

    // ---- 构造 ----
    // 始终学偏置 b（bias folding 并入权重一起解）；无参数。
    LinearRegression() = default;

    // ---- 训练阶段 ----
    // 在样本 X（n 行 × d 列）和连续标签 y（n 个）上训练。
    // 学完后 weights_ 长度 = d，bias_ = b；fitted_ 置 true。
    // 实现细节留 TODO，见下方。
    LinearRegression &fit(const Matrix<T> &X, const std::vector<T> &y);

    // ---- 推理阶段 ----
    // 对 X 的每行返回预测值 ŷ = w·x + b（连续值；不是 ±1 那种分类标签）。
    // X 的列数必须等于 fit 时的特征数 d。
    std::vector<T> predict(const Matrix<T> &X) const;

    // ---- 评指标 ----
    // 在 (X, y) 上算 R²（决定系数）。
    // 公式： R² = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²
    // 等价于： R² = 1 - SSE / SST，其中 ȳ = mean(y)
    // （想顺便学一下，可以先调 predict 拿 ŷ，再和 y / ȳ 一起算）
    T score(const Matrix<T> &X, const std::vector<T> &y) const;

    // ---- 查询（命名和 Perceptron 统一：weights / bias）----
    // fit 之前调用行为未定义（看 fitted() 自己判）。
    std::vector<T> weights() const { return weights_; }     // 长度 d
    T bias() const { return bias_; }         // 标量

    bool fitted() const { return fitted_; }                  // 是否已 fit

private:
    std::vector<T> weights_;          // 权重 w，fit 后长度 = d
    T bias_ = T(0);      // 偏置 b
    bool fitted_ = false;   // 是否已 fit 过
};

// ============================================================================
//  类外实现（模板必须留在头文件里，不能拆 .cpp —— 每个翻译单元都要看到定义
//  才能实例化）。
//
//  下面三个函数的实现都留给你写：函数体已经填好参数校验和「学完后写回
//  成员变量」的注释骨架，**核心数学计算的部分用 TODO 标出**。把 TODO
//  替换成你的实现就行。
// ============================================================================

template<typename T>
LinearRegression<T> &LinearRegression<T>::fit(const Matrix<T> &X,
                                              const std::vector<T> &y) {
    // ---- 1) 参数校验（用 check.h 的 CHECK，习惯和 Perceptron 一致） ----
    CHECK(y.size() == X.rows())
        << "LinearRegression::fit: y.size() (" << y.size()
        << ") must equal X.rows() (" << X.rows() << ")";
    CHECK(X.cols() >= 1) << "LinearRegression::fit: need at least 1 feature column";
    CHECK(X.rows() >= static_cast<size_type>(X.cols() + 1))
        << "LinearRegression::fit: n_samples (" << X.rows()
        << ") must be >= n_features + 1 (for intercept) to solve the "
        << "normal equation (otherwise X^T X is singular)";

    // ---- 2) 准备设计矩阵 X_design：永远在 X 右边拼一列 1（bias folding）----
    // 把偏置 b 并入权重一起解，最后一位 = b。这样：
    //   - 一次矩阵乘算所有 ŷ（predict 同理）
    //   - 不需要单独管 bias，代码更干净（perceptron.h 也用同一招）
    // X_design 的形状：n × (d+1)
    Matrix<T> X_design = X.with_ones_column();
    const size_type D = X_design.cols();   // = d + 1

    // ---- 3) 把 y 变成 n×1 的列向量（矩阵运算需要） ----
    Matrix<T> Y = Matrix<T>::from_column(y);

    // ---- 4) 求解权重 w_aug（长度 D = d+1 的列向量） ----
    //   TODO —— 核心训练代码留给你写，二选一：
    //
    //   (a) 正规方程（解析解，最直接）：
    //         A = X_design^T · X_design              // (d+1) × (d+1)
    //         b = X_design^T · Y                      // (d+1) × 1
    //         w_aug = A^(-1) · b                      // (d+1) × 1
    //       库里的 Matrix<T> 已经有 transposed() / operator*，
    //       但**没有现成求逆**，所以这条路线要么你手写 Gauss-Jordan
    //       求逆，要么用「解线性方程组 A·w = b」代替显式求逆（更稳）——
    //       建议用后者：手写一个高斯消元解 A w_aug = b 即可。
    //
    //   (b) 梯度下降（数值解，更通用，便于以后加 Ridge / Lasso 正则）：
    //         随机初始化 w_aug；每轮算梯度 ∇L = (2/n) X^T (X w - y)；
    //         w_aug -= lr * ∇L；跑 N 轮。损失是凸的，必然收敛到全局最小。
    //       想练手或想顺便为 Ridge 打基础就选这条。
    //
    //   解出 w_aug 后，用下面这段把它拆回 weights / bias（已写好）：
    Matrix<T> w_aug;  // TODO: 把这一行替换成你的求解结果（(d+1) × 1）

    // w_aug 的最后一维是 bias，前面 d 维是 weight
    std::vector<T> w_vec(D);
    for (size_type i = 0; i < D; ++i) w_vec[i] = w_aug(i, 0);
    bias_ = w_vec.back();
    w_vec.pop_back();
    weights_ = std::move(w_vec);

    // ---- 5) 标记训练完成，返回 *this 便于链式 ----
    fitted_ = true;
    return *this;
}

template<typename T>
std::vector<T> LinearRegression<T>::predict(const Matrix<T> &X) const {
    // ---- 校验：必须先 fit；列数必须等于训练时的特征数 d ----
    CHECK(fitted_) << "LinearRegression::predict: must call fit() before predict()";
    CHECK(X.cols() == weights_.size())
        << "LinearRegression::predict: feature count mismatch, got "
        << X.cols() << " cols but trained on " << weights_.size();

    // 思路：算 ŷ = X · weights_ + bias_  （逐元素加偏置）
    // 想省一次循环可继续用 bias folding：把 weights_ 末尾补上 bias_
    // 得到 (d+1) 维 w_aug，给 X 拼一列 1 后一次矩阵乘得到 ŷ 的列向量，
    // 再把 n×1 拉平成 std::vector<T> 返回。
    std::vector<T> weights = weights_;
    weights.push_back(bias_);
    Matrix<T> w_aug = Matrix<T>::from_row(weights);
    Matrix<T> features = X.with_ones_column();

    std::vector<T> result(X.rows(), T(0));
    Matrix<T> result_mat = features * w_aug.transposed();
    for (auto row = 0; row < result_mat.rows(); row++) {
        result[row] = result_mat(row, 0);
    }
    return result;
}

template<typename T>
T LinearRegression<T>::score(const Matrix<T> &X, const std::vector<T> &y) const {
    // R² = 1 - SSE / SST
    //   SSE = Σ_i (y_i - ŷ_i)²   （模型残差平方和，越小越好）
    //   SST = Σ_i (y_i - ȳ)²     （总平方和；和「只猜均值 ȳ」比）
    //
    //   TODO —— 最自然的实现：先 predict 拿 ŷ，再两遍循环算 SSE 和 ȳ / SST。
    //   注意边界：SST == 0（y 全相等）时 R² 未定义，自己决定怎么兜底
    //   （sklearn 此时返回 0.0 + 警告或 1.0 + 警告，简单点直接返回 T(1)
    //   表示「完美拟合常数」也行 —— 这是个值得思考的设计点）。
    //
    //   实现完把下面这行替换掉：
    (void) X;
    (void) y;  // TODO: 删掉这一行
    return T(0);       // TODO: 替换为真正的 R²
}
