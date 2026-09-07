// ============================================================================
//  kmeans.h —— K 均值聚类 K-Means（header-only）
//
//  干的事（对应 scikit-learn 的 sklearn.cluster.KMeans）：
//  给一批**没有标签**的样本 {x_1..x_n}，把它们自动分成 k 簇。它不做分类、
//  不做回归，而是**无监督**地找出数据里「天然聚在一起的几坨」：
//      - 簇（cluster）没有预先定义的名字，只是「离得近的一组点」；
//      - 每个簇用它的**中心**（质心 centroid / cluster center）代表：
//        簇心 = 这个簇里所有样本的算术平均；
//      - 训练完每个样本被指派给离它最近的簇心（簇号 0..k-1）。
//
//  为什么值得单独一个头文件？
//    - 它是「无监督学习」的入门款：KNN 有监督（要标签）、k-means 无监督
//      （不要标签），两者是「监督 vs 无监督」最干净的对照。
//    - 它是「EM 算法」的雏形：交替「分配 → 更新」的两步结构就是 EM 的
//      expectation / maximization 两步，理解它等于给高斯混合模型（GMM）
//      打地基。
//    - 它逼你思考「什么叫好的聚类」：k 怎么选、特征怎么缩放、初始簇心怎么
//      放 —— 每个都是实践里的真实坑。
//
//  数学定义（Lloyd 算法，1957；「k-means」这个叫法是 MacQueen 1967 起的）：
//    目标：找 k 个簇心 c_1..c_k，最小化**总惯性**（total inertia，也叫
//    WCSS / within-cluster sum of squares / SSE）：
//              J(c_1..c_k) = Σ_{i=1}^{n} min_j ||x_i - c_j||²
//    直觉：让每个样本「离它所在一簇的簇心」尽量近。
//
//    算法（交替两个步骤，反复迭代直到收敛）：
//      初始化：挑 k 个初始簇心（怎么挑很关键，见下方「扩展点 A」）
//      循环直到簇心不再动（或达到 max_iter 轮）：
//        ① 分配步（assignment）：每个样本归到离它最近的簇心
//            label_i = argmin_j ||x_i - c_j||
//        ② 更新步（update）：每个簇心 = 该簇所有样本的算术平均
//            c_j = (1/|S_j|) Σ_{x ∈ S_j} x        （S_j = 第 j 簇的样本集）
//    每一步都**不增加**目标 J（k-means 的单调性保证），所以算法一定收敛到
//    某个**局部最优**；但局部最优 ≠ 全局最优，初始簇心不同可能收敛到不同
//    结果 —— 这正是下面 n_init（多起点）与 k-means++ 存在的理由。
//
//  为什么「算术平均」当簇心？（这是扩展点选择的地基，务必先读）
//    - 对欧氏距离平方 ||x - c||²，给定簇 S，让 Σ_{x∈S} ||x-c||² 最小的 c
//      **恰好是均值**（对 c 求导=0 可得）。所以「平方距离 ↔ 均值」天生一对。
//    - 换距离必须连「中心取什么」一起换（L1 距离对应中位数、余弦距离对应
//      单位化后的均值）。所以 k-means 的度量不像 KNN 那样可以随意换，本文件
//      固定欧氏距离 + 算术平均，自定义度量请自行保证配套（见「扩展」）。
//
//  扩展点（风格和 knn.h 一致，都是编译期策略模板）：
//    A. 初始化策略 init —— 放 namespace detail::init 下，默认 KMeansPlusPlus。
//       初始簇心怎么挑，直接决定结果好坏（差的初始化可能收敛进很差的局部
//       最优）。统一签名（KMeans 只依赖这一个接口，换策略零改动）：
//            template <typename T>
//            Matrix<T> operator()(const Matrix<T>& X, size_t k, Random& rng) const
//       返回 k 行 × d 列的初始簇心矩阵。本文件预置两个：
//          detail::init::RandomInit      从 X 里均匀随机抽 k 个不同样本
//          detail::init::KMeansPlusPlus  k-means++（sklearn 默认）：第一个
//                                        随机抽，之后每个新簇心按「到最近
//                                        已选簇心的距离平方」加权概率抽样，
//                                        让初始簇心尽量「撒得开」。
//    B. n_init —— 同样的数据跑 n_init 遍（每遍独立随机初始化），取**惯性
//       最小**的那遍当最终结果。朴素但有效的「多起点」技巧，专门对付局部
//       最优（sklearn 默认 10）。
//
//  约定（与 KNN / Perceptron 一致）：
//    - 数据布局：Matrix<T>，每行 = 一个样本，每列 = 一个特征
//    - **没有标签**：fit 只吃 X（无监督学习）
//    - 默认 T = float：聚类就是距离比较 + 求平均，float 足够；需要更高
//      精度时显式写 KMeans<double>
//
//  用法（API 形状对齐 sklearn）：
//    KMeans<> km(/* n_clusters = */ 3);             // 默认 k=8，这里改 3
//    km.fit(X);                                     // 学出簇心（无监督）
//    auto C      = km.centers();                    // 簇心矩阵 k×d
//    auto lab    = km.predict(X_test);              // 每行归哪簇（0..k-1）
//    auto lab2   = km.fit_predict(X);               // fit + predict 一步
//    T   iner    = km.inertia();                    // 总惯性（SSE，越小越好）
//    T   scr     = km.score(X);                     // 负惯性（sklearn 语义）
//
//  与 sklearn 的字段对应（命名按本项目习惯，不背 sklearn 的下划线包袱）：
//    km.centers()   ↔  sklearn: cluster_centers_    (k, n_features)
//    km.inertia()   ↔  sklearn: inertia_            (float)
//    km.labels()    ↔  sklearn: labels_             (每次 fit 的分配)
//    km.score()     ↔  sklearn: score()             (负惯性)
//
//  复杂度：
//    fit：O(n_init · max_iter · n · k · d)。每次迭代：分配步 O(n·k·d)、
//    更新步 O(n·d)。n = 样本数、k = 簇数、d = 特征维数（k、d 通常远小于 n）。
//
//  配套讲解：docs/kmeans.md（聚类目标、Lloyd 迭代、初始化策略、k 怎么选）
//
//  留给你实现的（骨架期方法体一律 throw，见类外定义处的 TODO 注释）：
//    - fit(...)              完整 Lloyd 迭代（分配→更新→收敛判断）
//                             + n_init 多起点选优
//    - predict(...)          给任意样本矩阵分配最近簇心
//    - assign(...)           分配步（私有工具，predict 复用）
//    - total_inertia(...)    惯性（WCSS）计算（私有工具，score 复用）
//    - score(...)            负惯性评指标
//    - detail::init::RandomInit / KMeansPlusPlus   两种初始化策略
//  每个方法体的 throw 换成真实现即可，参数校验和接口已全部就位；
//  写完用 tests/kmeans_test.cpp 当验收清单。
//
//  扩展（写完后可挑战）：
//    - 自定义 init 策略：照 detail::init 的签名新写一个 functor（比如
//      FarthestFirst、或从 sklearn 学的「按数据范围均匀撒点」）
//    - 自定义度量 + 配套中心：想换成 L1 / 余弦时，把「距离比较」和「中心
//      计算」一起改（L1→中位数、余弦→单位化均值），并保证测试仍过
//    - 数据标准化：k-means 对特征尺度敏感，fit 前对 X 做 z-score 常是
//      必要的（本库暂无 scaler）
// ============================================================================
#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "mini_mlmath/check.h"
#include "mini_mlmath/ml/knn.h" // 复用 detail::distance::EuclideanDistance
#include "mini_mlmath/random.h"

// ============================================================================
//  1. 初始化策略（Init）—— 扩展点 A，全部是函数对象 functor
//
//  统一签名（见文件头「扩展点 A」）：
//      template <typename T>
//      Matrix<T> operator()(const Matrix<T>& X, size_t k, Random& rng) const
//  X 是样本矩阵（n×d），k 是要挑的簇数，rng 是可复现的随机引擎，
//  返回值是 k×d 的初始簇心矩阵。
//
//  全部放在 namespace detail::init 下（detail = 实现细节，仅供 KMeans 使用）。
//  自定义策略照抄这个签名新增一个 struct 即可 —— 两个预置品本身就是模板示范。
// ============================================================================
namespace detail {
namespace init {

// 随机初始化：从 X 里均匀**不放回**地抽 k 个样本当初始簇心。
// 实现思路：先随机打乱行号（Fisher-Yates 洗牌，前 k 个就是答案），或者
// 逐行用 rng.uniform<int>(i, n-1) 交换，保证 k 个簇心互不重复。
struct RandomInit {
    /**
     * @brief 随机初始化：从样本中均匀抽 k 个不同样本当初始簇心
     * @param X 样本矩阵（n_samples × n_features）
     * @param k 需要的簇数
     * @param rng 随机数引擎（默认种子 42，可复现）
     * @return k × n_features 的初始簇心矩阵
     */
    template <typename T>
    Matrix<T> operator()(const Matrix<T> &X, std::size_t k, Random &rng) const {
        CHECK(k >= 1 && k <= X.rows())
            << "RandomInit: k (" << k << ") must be in [1, n_samples="
            << X.rows() << "]";

        const std::size_t n = X.rows();

        std::vector<std::size_t> perm(n);
        for (std::size_t i = 0; i < n; ++i)
            perm[i] = i;
        for (std::size_t i = 0; i < k; ++i) {
            std::size_t j = rng.uniform(i, n - 1); // 整数闭区间，上界 n-1
            std::swap(perm[i], perm[j]);
        }

        Matrix<T> centrals(k, X.cols());
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t c = 0; c < X.cols(); ++c)
                centrals(i, c) = X(perm[i], c);
        return centrals;
    }
};

// k-means++ 初始化（sklearn 默认）：让初始簇心「撒得开」。
// 实现思路：
//   1) 均匀随机抽第一个簇心 c_1；
//   2) 对每个样本 x，记 D(x) = 到最近一个已选簇心的距离，算 D(x)²；
//   3) 按「D(x)² / ΣD(x)²」做加权概率抽样下一个簇心（远点更容易被选中）；
//   4) 重复 2)~3) 直到凑满 k 个。
// 代价：每个新簇心要扫一遍全部样本算最近距离，O(k²·n·d)；换来初始簇心
// 通常离得足够开，Lloyd 收敛进好局部最优的概率大幅提升。
struct KMeansPlusPlus {
    /**
     * @brief k-means++ 初始化：让初始簇心尽量分散
     * @param X 样本矩阵（n_samples × n_features）
     * @param k 需要的簇数
     * @param rng 随机数引擎（默认种子 42，可复现）
     * @return k × n_features 的初始簇心矩阵
     */
    template <typename T>
    Matrix<T> operator()(const Matrix<T> &X, std::size_t k, Random &rng) const {
        // TODO —— 核心初始化留给你写（思路见结构体上方注释）：
        //   1) 均匀抽第一个簇心
        //   2) 循环：算每个样本到最近已选簇心的 D(x)²，按加权概率抽下一个
        // 写完把下面两行删掉即可。
        (void)rng; // TODO: 删掉这一行
        throw std::logic_error(
            "KMeansPlusPlus::operator(): not implemented yet — TODO: "
            "distance-squared weighted sampling, see comment above");
    }
};

} // namespace init
} // namespace detail

// ============================================================================
//  2. KMeans 聚类器本体
//
//  模板参数：
//    T    特征数值类型，默认 float（距离/均值运算，要求浮点）
//    Init 初始化策略 functor 类型，默认 k-means++（扩展点 A）
//
//  用法见文件头注释。
// ============================================================================
template <typename T = float,
          typename Init = detail::init::KMeansPlusPlus>
class KMeans {
    static_assert(std::is_floating_point_v<T>,
                  "KMeans<T> requires floating-point feature type T "
                  "(float / double / long double). Distances and means on int "
                  "will truncate silently.");

public:
    using value_type = T;
    using size_type = std::size_t;
    using label_type = int; // 簇号固定用 int（0..k-1，不像 KNN 用泛型 Label）

    /**
     * @brief 构造一个 KMeans 聚类器
     * @param n_clusters 簇数 k（sklearn 同名参数，默认 8）
     * @param max_iter 每遍 Lloyd 迭代的最大轮数（默认 300）
     * @param tol 收敛容差：连续两轮的惯性变化小于它即提前停止（默认 1e-4）
     * @param n_init 独立随机初始化的遍数，取惯性最小的一遍（默认 10）
     */
    KMeans(size_type n_clusters = 8,
           size_type max_iter = 300,
           T tol = T(1e-4),
           size_type n_init = 10);

    /**
     * @brief 在无标签样本 X 上训练，学出 k 个簇心
     * @param X 训练样本矩阵，每行一个样本、每列一个特征（n × d，无标签）
     * @return *this，便于链式调用
     * @note 无监督：X 不配标签；训练后 centers()/labels()/inertia() 可用
     */
    KMeans &fit(const Matrix<T> &X);

    /**
     * @brief 对给定样本矩阵逐行预测所属簇
     * @param X 查询样本矩阵，列数必须等于训练时的特征数 d
     * @return 长度 X.rows() 的标签数组，取值 0..n_clusters-1（离哪个簇心最近归谁）
     * @note 必须先调用 fit
     */
    std::vector<int> predict(const Matrix<T> &X) const;

    /**
     * @brief fit + predict 一步完成（sklearn 的 fit_predict）
     * @param X 训练（同时也是待预测）的样本矩阵
     * @return X 每行所属簇的标签，等价于 fit 后的 labels()
     */
    std::vector<int> fit_predict(const Matrix<T> &X);

    /**
     * @brief 评指标：负惯性（sklearn 语义，越大越好）
     * @param X 样本矩阵，列数必须等于训练时的特征数 d
     * @return -（样本到各自最近簇心的距离平方和）；惯性越小聚类越好，
     *         取负号让「越大越好」和其他模型的 score 统一方向
     * @note 必须先调用 fit
     */
    T score(const Matrix<T> &X) const;

    /**
     * @brief 学到的簇心矩阵
     * @return k × d 的 Matrix<T>，第 j 行是第 j 簇的中心（未 fit 时为空）
     */
    Matrix<T> centers() const { return centers_; }

    /**
     * @brief 最近一次 fit 时每个样本被分到的簇
     * @return 长度 n 的 int 数组，取值 0..n_clusters-1（未 fit 时为空）
     */
    std::vector<int> labels() const { return labels_; }

    /**
     * @brief 最近一次 fit 的总惯性（WCSS / SSE）
     * @return 每个样本到其簇心的距离平方和，越小表示簇越紧凑（未 fit 时为 0）
     */
    T inertia() const { return inertia_; }

    /**
     * @brief 是否已训练过
     * @return true 表示 fit 已成功执行
     */
    bool fitted() const { return fitted_; }

    /**
     * @brief 返回构造时设定的簇数
     * @return n_clusters 的值（k）
     */
    size_type n_clusters() const { return n_clusters_; }

private:
    /**
     * @brief 分配步：把 X 的每一行分到最近的簇心
     * @param X 样本矩阵（n × d）
     * @param centers 当前簇心矩阵（k × d）
     * @return 长度 n 的标签数组，label[i] = 离 x_i 最近的簇心行号
     * @note 用 detail::distance::EuclideanDistance（欧氏距离）比较远近
     */
    std::vector<int> assign(const Matrix<T> &X,
                            const Matrix<T> &centers) const;

    /**
     * @brief 计算总惯性（WCSS / SSE）
     * @param X 样本矩阵（n × d）
     * @param centers 簇心矩阵（k × d）
     * @param labels 每个样本所属簇（长度 n，通常来自 assign）
     * @return Σ_i ||x_i - c_{labels[i]}||²（距离平方和，越小越好）
     */
    T total_inertia(const Matrix<T> &X, const Matrix<T> &centers,
                    const std::vector<int> &labels) const;

    size_type n_clusters_ = 8; // 簇数 k
    size_type max_iter_ = 300; // Lloyd 迭代最大轮数
    T tol_ = T(1e-4);          // 收敛容差
    size_type n_init_ = 10;    // 多起点遍数
    Random rng_;               // 随机引擎（默认种子 42，实验可复现）
    Init init_;                // 初始化策略（默认 k-means++，扩展点 A）
    Matrix<T> centers_;        // 簇心，fit 后为 k×d
    std::vector<int> labels_;  // 最近一次 fit 的分配，长度 n
    T inertia_ = T(0);         // 最近一次 fit 的总惯性
    bool fitted_ = false;      // 是否已 fit 过
};

// ============================================================================
//  类外实现（模板必须留在头文件里 —— 每个翻译单元都要看到定义才能实例化）。
//
//  下面方法的函数体只有参数校验和 TODO 注释，核心算法留给你写：把每个
//  throw 换成真实现即可（接口、校验、成员变量已全部就位）。
// ============================================================================

template <typename T, typename Init>
KMeans<T, Init>::KMeans(size_type n_clusters, size_type max_iter,
                        T tol, size_type n_init)
    : n_clusters_(n_clusters), max_iter_(max_iter), tol_(tol),
      n_init_(n_init) {}

template <typename T, typename Init>
KMeans<T, Init> &KMeans<T, Init>::fit(const Matrix<T> &X) {
    // ---- 1) 参数校验（用 check.h 的 CHECK，习惯和 KNN 一致）----
    CHECK(X.rows() > 0 && X.cols() > 0)
        << "KMeans::fit: X must be non-empty (n_samples x n_features)";
    CHECK(n_clusters_ >= 1)
        << "KMeans::fit: n_clusters must be >= 1, got " << n_clusters_;
    CHECK(n_clusters_ <= X.rows())
        << "KMeans::fit: n_clusters (" << n_clusters_
        << ") must be <= n_samples (" << X.rows() << ")";

    // ---- 2) 核心训练：n_init 遍「随机初始化 + Lloyd 迭代」，取惯性最小 ----
    //   TODO —— 核心训练代码留给你写，推荐结构：
    //
    //   T best_inertia = 正无穷;  Matrix<T> best_centers;  vector<int> best_labels;
    //   for (t = 0; t < n_init_; ++t) {
    //       a) 初始化簇心：Matrix<T> centers = init_(X, n_clusters_, rng_);
    //       b) Lloyd 迭代（max_iter_ 轮，或提前收敛就 break）：
    //            ① 分配步：labels = assign(X, centers);
    //            ② 更新步：对每个簇 j，centers 第 j 行 = 该簇样本的算术平均
    //               （用 count + sum 两遍扫描累加，再除以簇内样本数）
    //            ③ 收敛判断：本轮 centers 相对上轮的位移（或惯性变化）
    //               < tol_ 就 break
    //       c) 算本次惯性：T iner = total_inertia(X, centers, labels);
    //       d) 若 iner < best_inertia：更新 best_*（连 labels 一起存！）
    //   }
    //   最后写回成员：centers_ = best_centers;  labels_ = best_labels;
    //                inertia_ = best_inertia;   fitted_ = true;   return *this;
    //
    //   设计点（实现前先想清楚，都是真实世界会踩的坑）：
    //     - 空簇：某簇心一个样本都没分到时，均值未定义。常见做法：① 该簇心
    //       重新随机初始化；② 把它挪到离所有簇心最远的样本附近；③ 丢掉一个
    //       簇（k 变小）。选一个并写清理由。
    //     - 收敛判据用「簇心位移」还是「惯性变化」：惯性量纲是距离平方，
    //       位移量纲是距离，注意 tol_ 的语义要和你的判据一致（本骨架默认
    //       按惯性变化 < tol_ 理解，见构造参数的 @brief）。
    //     - 更新步别忘了 count：只累加 sum 不数个数，除以 0 会得到 NaN。
    //     - 距离比较用 detail::distance::EuclideanDistance（knn.h 里那份），
    //       assign 里已经写了思路，直接实现它即可。

    (void)X; // TODO: 删掉这一行
    throw std::logic_error(
        "KMeans::fit: not implemented yet — TODO: Lloyd iteration + n_init "
        "best-of, see the TODO comment above");
}

template <typename T, typename Init>
std::vector<int> KMeans<T, Init>::predict(const Matrix<T> &X) const {
    CHECK(fitted_) << "KMeans::predict: must call fit() before predict()";
    CHECK(X.cols() == centers_.cols())
        << "KMeans::predict: feature count mismatch, got " << X.cols()
        << " cols but trained on " << centers_.cols();

    // 对 X 的每一行找最近的簇心，返回簇号（0..k-1）。
    //   TODO —— 其实就是一次分配步：return assign(X, centers_);
    (void)X; // TODO: 删掉这一行
    throw std::logic_error("KMeans::predict: not implemented yet");
}

template <typename T, typename Init>
std::vector<int> KMeans<T, Init>::fit_predict(const Matrix<T> &X) {
    // 组合函数：fit 会填好 labels_（见 fit 的 TODO），直接取即可。
    fit(X);
    return labels_;
}

template <typename T, typename Init>
T KMeans<T, Init>::score(const Matrix<T> &X) const {
    CHECK(fitted_) << "KMeans::score: must call fit() before score()";
    CHECK(X.cols() == centers_.cols())
        << "KMeans::score: feature count mismatch, got " << X.cols()
        << " cols but trained on " << centers_.cols();

    // sklearn 语义：返回负惯性 —— 惯性越小聚类越好，取负号后「越大越好」，
    // 和其他模型 score（越大越好）统一方向。
    //   TODO —— 一行即可：
    //   return -total_inertia(X, centers_, assign(X, centers_));
    (void)X; // TODO: 删掉这一行
    throw std::logic_error("KMeans::score: not implemented yet");
}

template <typename T, typename Init>
std::vector<int> KMeans<T, Init>::assign(const Matrix<T> &X,
                                         const Matrix<T> &centers) const {
    // 分配步：对 X 每一行，算到每个簇心的欧氏距离，取最小者的行号。
    //   TODO —— 实现思路（就是 knn.h 里 BruteForce::query 的简化版）：
    //   std::vector<int> labels(X.rows());
    //   detail::distance::EuclideanDistance metric;
    //   for (i = 0; i < X.rows(); ++i) {
    //       best_j = 0;  best_d = 正无穷;
    //       for (j = 0; j < centers.rows(); ++j) {
    //           d = metric(X.data() + i*X.cols(),
    //                      centers.data() + j*centers.cols(), X.cols());
    //           if (d < best_d) { best_j = j; best_d = d; }
    //       }
    //       labels[i] = best_j;
    //   }
    //   return labels;
    throw std::logic_error("KMeans::assign: not implemented yet");
}

template <typename T, typename Init>
T KMeans<T, Init>::total_inertia(const Matrix<T> &X,
                                 const Matrix<T> &centers,
                                 const std::vector<int> &labels) const {
    // 惯性（WCSS / SSE）：Σ_i ||x_i - c_{labels[i]}||²。
    //   TODO —— 实现思路：逐行找到对应簇心，累加**距离平方**（不是距离；
    //   分配步用「距离」比大小、这里用「平方」求数值，语义要分清）。
    //   提示：可以把 assign 里算距离的那段抽成小循环，或直接用
    //   EuclideanDistance 算出距离再平方。
    throw std::logic_error("KMeans::total_inertia: not implemented yet");
}
