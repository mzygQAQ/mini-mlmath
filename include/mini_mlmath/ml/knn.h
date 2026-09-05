// ============================================================================
//  knn.h —— K 近邻分类器 K-Nearest Neighbors（header-only）
//
//  干的事（对应 scikit-learn 的 sklearn.neighbors.KNeighborsClassifier）：
//  给一批带标签的样本 (x_i, y_i)，**不做任何训练/参数学习**，而是把数据原样
//  存下来；预测一个新样本 x 时，找出离它最近的 k 个训练样本，让它们**投票**
//  决定 x 的类别（多数类胜出）。这就是「懒学习 / 基于实例的学习」：
//      - 没有显式学出的模型（没有 weights / bias 那种参数）；
//      - fit 只是「记住数据」，真正的计算全部发生在 predict 时；
//      - 和 Perceptron / LinearRegression 的「学一个 w 出来」形成鲜明对照。
//
//  为什么值得单独一个头文件？
//    - 它是机器学习里**唯一一个零训练**的算法：一句话就能讲清原理，却在实际
//      问题（推荐、检索、小样本分类）里大量使用。适合当「监督学习入门第一课」。
//    - 它逼你思考「怎么定义两个样本的相似/距离」—— 特征缩放、距离度量、高维
//      诅咒（curse of dimensionality）这些概念全挂在它身上。
//    - 它是本库里展示**两个扩展点**的范例（本文件的教学重点，见下）：
//        ① 距离度量 metric（L1 / L2 / Chebyshev / 自定义）
//        ② 近邻搜索策略 strategy（BruteForce，预留 KDTree 位置）
//
//  两个扩展点是怎么设计的（请带着这个视角读代码）：
//    A. Metric（度量）：写成「函数对象 functor」。
//       一个 metric 就是一个定义了 operator() 的结构体，签名统一为
//            T operator()(const T* a, const T* b, size_t dim) const
//       传入两个点的内存首地址和维数，返回一个「越小越近」的距离标量。
//       好处：调换度量只改 KNN 的一个模板实参，不用动任何计算代码；要自定义
//       距离（比如「只看第 0 维」），照抄这个签名写个结构体塞进去即可。
//    B. Strategy（搜索策略）：编译期策略模板。
//       KNN 把「找 k 个最近邻居」这件脏活外包给一个独立类，默认是 BruteForce
//       （全量线性扫描，见 knn.h 里 BruteForce 类的头注释）。所有策略类必须
//       暴露同一组方法（build / query / sample_count / feature_count，见下），
//       KNN 只通过这组接口调它，完全不知道内部是「全量扫」还是「KDTree 剪枝」。
//       本文件已用 KDTreeSearch 骨架占好位：接口和 BruteForce 一字不差，方法体
//       暂时 throw「not implemented」（见文件里的 3b 节）。将来做 KDTree 时把
//       build 换成建树、query 换成剪枝搜索即可 —— 一个字符都不用动 KNN。
//
//  数学定义（多数投票）：
//    给定训练集 {x_1..x_n}、标签 {y_1..y_n}、正整数 k、距离度量 d(·,·)。
//    对查询点 q：
//      1) 算 q 到每个训练样本的距离，取最小的 k 个（集合记 N_k(q)）；
//      2) 预测标签 = N_k(q) 里出现次数最多的那个（多数投票 majority vote）。
//    两个设计点（对应 sklearn 的 weights 参数，本教学版只实现 uniform）：
//      - uniform（默认，本文件）：每个邻居一票，权重相同 —— 直觉是「近邻区域
//        里哪个类人多就归哪个类」；
//      - distance：按距离倒数加权（越近权重越大），sklearn 有但这里留作扩展。
//    平票处理：本实现取「出现次数并列最多时、标签字典序最小」的那个，保证
//    输出确定可复现（和 statistics.h 的 mode 同一套决策）。
//
//  复杂度：
//    fit：O(1)（只是拷贝数据）。
//    predict（BruteForce 策略）：每个查询 O(n·d + n·log k)，
//      n = 训练样本数、d = 特征维数。没有训练开销、查询全量扫 —— 这就是
//      「懒学习」的代价，也是将来想用 KDTree 把查询压到 O(log n) 的动力。
//
//  约定（与 Perceptron / LinearRegression 一致）：
//    - 数据布局：Matrix<T>，每行 = 一个样本，每列 = 一个特征
//    - 标签：std::vector<Label>，n 个离散类别（int / string 等皆可；
//      要求支持 operator<（做平票排序）与 operator==）
//    - 默认 T = float：距离就是浮点加减乘 + 一次开方，float 足够
//    - 默认 Label = int：分类标签通常就是 0/1/2…
//
//  用法（API 形状对齐 sklearn）：
//    KNN<> knn(/* k = */ 3);                       // 默认 int 标签 + float + 欧氏 + 暴力
//    knn.fit(X_train, y_train);                     // 记住数据（零训练）
//    auto pred = knn.predict(X_test);               // 每行一个多数投票结果
//    double acc = knn.score(X_test, y_test);        // 分类准确率
//    auto nbrs = knn.kneighbors(X_test);            // 想看最近邻居长啥样时
//
//  换度量 / 换策略（本文件两个扩展点的用法示范）：
//    KNN<int, float, ManhattanDistance>  knn_l1(5);      // L1 曼哈顿距离
//    KNN<int, float, ChebyshevDistance>  knn_linf(5);    // L∞ 切比雪夫距离
//    KNN<int, float, EuclideanDistance,
//        KDTreeSearch>                    knn_fast(5);   // 切 KDTree：当前是 throw 骨架
//    自定义度量：struct MyMetric { template<typename T>
//        T operator()(const T* a, const T* b, size_t d) const { ... } };
//    然后 KNN<int, float, MyMetric> 即可。
//
//  配套讲解：docs/knn.md（常见 metric、搜索策略、参数选择见该文档）
//
//  留给你扩展的（不影响当前可用，标在注释里）：
//    - distance 权重投票（weights='distance'）
//    - 回归模式 KNeighborsRegressor（取 k 近邻标签的均值而非投票）
//    - KDTreeSearch：接口骨架已在文件里占位（见 3b），把 build / query 从
//      throw 换成建树 + 剪枝搜索即可；BallTreeSearch 同理照抄接口。
// ============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "mini_mlmath/check.h"
#include "mini_mlmath/matrix.h"

// ============================================================================
//  1. 近邻查询结果：一次 query 找到的一个邻居
// ============================================================================
template <typename T>
struct NeighborHit {
    std::size_t index = 0;   // 训练样本的行号（第几个样本）
    T distance = T(0);       // 它到查询点的距离（由 Metric 算出）
};

// 邻居排序规则：先比距离（小 = 近），距离相同再比行号（保证确定性）。
// BruteForce::query 里用它把邻居按「距离近 → 远」排好交给 KNN 投票。
template <typename T>
struct NeighborHitLess {
    bool operator()(const NeighborHit<T>& a, const NeighborHit<T>& b) const {
        if (a.distance != b.distance) return a.distance < b.distance;
        return a.index < b.index;
    }
};

// ============================================================================
//  2. 距离度量（Metric）—— 扩展点 A，全部是函数对象 functor
//
//  统一签名（见文件头「两个扩展点」的 A）：
//      T operator()(const T* a, const T* b, size_t dim) const
//  a / b 是两个样本的首地址（可直接指向 Matrix 行：X.data() + i*X.cols()），
//  dim 是维数。返回值越小表示两个点越「近」。
//
//  本文件预置三个最常用度量，更多度量（余弦距离、Mahalanobis、自定义）只需
//  照抄这个签名新增一个 struct 即可 —— 三个预置品本身就是模板示范。
// ============================================================================

// L2 欧氏距离（默认）：sqrt( Σ_i (a_i - b_i)² )
// 几何意义是「直线距离」，最直观、默认选它。注意内部先累加平方、最后才开方，
// 所以「比较远近」时其实可以省掉 sqrt（比较单调），这里为了距离值可读保留。
struct EuclideanDistance {
    template <typename T>
    T operator()(const T* a, const T* b, std::size_t dim) const {
        T sum = T(0);
        for (std::size_t i = 0; i < dim; ++i) {
            const T diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
};

// L1 曼哈顿距离（Manhattan / city-block）： Σ_i |a_i - b_i|
// 几何意义是「只能沿坐标轴走」的路径长度，对异常值不如 L2 敏感
//（平方会让大偏差被放大），高维数据里有时比 L2 更稳。
struct ManhattanDistance {
    template <typename T>
    T operator()(const T* a, const T* b, std::size_t dim) const {
        T sum = T(0);
        for (std::size_t i = 0; i < dim; ++i) {
            sum += std::abs(a[i] - b[i]);
        }
        return sum;
    }
};

// L∞ 切比雪夫距离（Chebyshev）： max_i |a_i - b_i|
// 取「各维偏差里最大的那个」。象棋里的王每步只能走一格，走遍棋盘所需步数
// 就是切比雪夫距离，所以也叫棋盘距离（chessboard distance）。
struct ChebyshevDistance {
    template <typename T>
    T operator()(const T* a, const T* b, std::size_t dim) const {
        CHECK(dim >= 1) << "ChebyshevDistance: need dim >= 1";
        T best = std::abs(a[0] - b[0]);
        for (std::size_t i = 1; i < dim; ++i) {
            best = std::max(best, std::abs(a[i] - b[i]));
        }
        return best;
    }
};

// ============================================================================
//  3. 近邻搜索策略（Strategy）—— 扩展点 B，编译期策略模板
//
//  一个策略类 = 一个「邻居索引（neighbor index）」：fit 时把训练数据建成某种
//  便于查询的结构，predict 时快速吐出某点的 k 个最近邻居。
//
//  所有策略类必须暴露同一组公共接口（KNN 只依赖这些，换策略零改动）：
//      void build(const Matrix<T>& X);              // 记住/建索引
//      std::vector<NeighborHit<T>> query(const T* p, size_t k) const;
//                   // 找离点 p 最近的 k 个邻居，按距离升序返回
//      std::size_t sample_count() const;            // 训练样本数 n
//      std::size_t feature_count() const;           // 特征维数 d
//
//  本文件先实现最朴素的 BruteForce：不建任何索引，query 时把 p 和全部 n 个
//  训练样本逐一算距离，再挑出最小的 k 个 —— 保证正确、零代码量。
//  将来 KDTreeSearch 满足同一组签名就能无缝替换（见文件头「扩展点 B」）。
// ============================================================================
template <typename T, typename Metric>
class BruteForce {
public:
    using value_type = T;
    using size_type = std::size_t;

    // 策略是「编译期类型」，构造无参数：Metric 作为成员默认构造即可
    //（三个预置 metric 都是无状态 functor）。
    BruteForce() = default;

    // ---- fit：原样拷贝训练数据（懒学习，不做任何统计/建树）----
    void build(const Matrix<T> &X) {
        CHECK(X.rows() > 0 && X.cols() > 0)
            << "BruteForce::build: X must be non-empty (n_samples x n_features)";
        X_ = X;   // 深拷贝一份：predict 时训练集不能跑掉
    }

    // ---- query：线性扫描全部训练样本，返回距离最小的 k 个邻居 ----
    // p 是查询点的首地址（长度 feature_count()）。
    // 实现套路：先把「每个训练样本的距离」全算出来放进 all，再用 partial_sort
    // 只把前 k 个最小的排好（O(n log k)，比全排序 O(n log n) 省），resize 截断。
    std::vector<NeighborHit<T>> query(const T *p, size_type k) const {
        const size_type n = sample_count();
        CHECK(k >= 1 && k <= n)
            << "BruteForce::query: k (" << k << ") must be in [1, n_samples="
            << n << "]";

        std::vector<NeighborHit<T>> all(n);
        for (size_type i = 0; i < n; ++i) {
            all[i].index = i;
            all[i].distance = metric_(p, X_.data() + i * X_.cols(), X_.cols());
        }
        // partial_sort 把 [begin, begin+k) 排成升序且就是全局最小的 k 个，
        // 后面 k 个元素只保证「不小于它们」，我们不需要，直接裁掉。
        std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k),
                          all.end(), NeighborHitLess<T>());
        all.resize(k);
        return all;
    }

    // ---- 查询 ----
    size_type sample_count() const { return X_.rows(); }
    size_type feature_count() const { return X_.cols(); }

private:
    Matrix<T> X_;        // 训练数据副本（懒学习：fit 的唯一产物）
    Metric metric_{};    // 距离度量实例（默认构造；见文件头扩展点 A）
};

// ============================================================================
//  3b. KDTreeSearch —— 备选策略的骨架占位（接口已定，实现待写）
//
//  一个 KDTree 是干什么的（将来实现时再看这段，现在只用于理解设计意图）：
//  把 d 维空间递归地「沿某一维切一刀」分成两半：选当前子树里方差最大的维度，
//  取该维中位数做分裂点，左子树放小于分裂点的点、右子树放大于的 —— 重复到
//  每片只剩少量点。查询点 q 的 k 近邻时沿着树往下走，并维护「当前已找到的最
//  远邻居距离」，用「q 到另一边子树边界盒的最短距离」做剪枝：若该距离已经
//  大于最远邻居距离，整棵子树都不可能含更近的点，直接跳过 —— 平均把查询从
//  O(n)（BruteForce 全量扫）压到 O(log n)。
//
//  为什么现在就要占位？
//    - 让 KNN 的模板实参从 BruteForce 换成 KDTreeSearch **能直接编译过**，
//      证明策略接口真的一字不差、切换只改一处（见文件头「扩展点 B」）；
//    - 想真正提速时只需回来填两个方法，KNN 及调用方零改动。
//
//  当前状态：build / query 一律 throw std::logic_error（not implemented）。
//  注意 build 会抛 —— 所以「fit 一个用了 KDTreeSearch 的 KNN」会在 fit 阶段
//  直接失败，而不是等到 predict 才暴露。这正是「占位还没实现」的正确姿势：
//  失败点尽可能靠前、错误信息明确。
// ============================================================================
template <typename T, typename Metric>
class KDTreeSearch {
public:
    using value_type = T;
    using size_type = std::size_t;

    // 策略是「编译期类型」，构造无参数（和 BruteForce 一致，接口对齐）。
    KDTreeSearch() = default;

    // ---- fit：TODO —— 把 X 建成一棵 KDTree（沿方差最大维递归切分）----
    void build(const Matrix<T> &X) {
        (void) X;   // 占位：参数暂不使用
        throw std::logic_error(
            "KDTreeSearch::build: not implemented yet — KDTree coming soon, "
            "use BruteForce (the default) until then");
    }

    // ---- query：TODO —— 剪枝搜索（见 3b 节头注释），返回最近 k 个邻居 ----
    std::vector<NeighborHit<T>> query(const T *p, size_type k) const {
        (void) p;   // 占位：参数暂不使用
        (void) k;
        throw std::logic_error(
            "KDTreeSearch::query: not implemented yet — KDTree coming soon, "
            "use BruteForce (the default) until then");
    }

    // ---- 查询（未 build 前返回 0，接口与 BruteForce 一致）----
    size_type sample_count() const { return 0; }
    size_type feature_count() const { return 0; }

private:
    Metric metric_{};   // 距离度量实例（实现剪枝时算边界盒距离要用，先留着）
};

// ============================================================================
//  4. KNN 分类器本体
//
//  模板参数（按「最常改 → 最不常改」排序）：
//    Label    标签类型，默认 int
//    T        特征数值类型，默认 float（距离计算，要求浮点）
//    Metric   距离度量 functor 类型，默认欧氏（扩展点 A）
//    Search   近邻搜索策略类模板，默认暴力扫描（扩展点 B，可选 KDTreeSearch）
//
//  用法见文件头注释。
// ============================================================================
template <typename Label = int,
          typename T = float,
          typename Metric = EuclideanDistance,
          template <typename, typename> class Search = BruteForce>
class KNN {
    static_assert(std::is_floating_point_v<T>,
                  "KNN<T> requires floating-point feature type T "
                  "(float / double / long double). Distances need sqrt / abs "
                  "on fractional values — int features will truncate silently.");

public:
    using value_type = T;
    using label_type = Label;
    using size_type = std::size_t;

    // ---- 构造 ----
    // n_neighbors：k，投票的邻居个数（sklearn 同名参数），默认 5。
    explicit KNN(size_type n_neighbors = 5) : k_(n_neighbors) {}

    // ---- 训练阶段（懒学习：只拷贝数据，O(1) 逻辑）----
    // 在样本 X（n 行 × d 列）和离散标签 y（n 个）上 fit。
    // 实际工作是把 X 交给策略对象 build 起来（BruteForce 只是拷一份）。
    KNN &fit(const Matrix<T> &X, const std::vector<Label> &y) {
        CHECK(X.rows() > 0 && X.cols() > 0)
            << "KNN::fit: X must be non-empty (n_samples x n_features)";
        CHECK(y.size() == X.rows())
            << "KNN::fit: y.size() (" << y.size()
            << ") must equal X.rows() (" << X.rows() << ")";

        search_.build(X);
        y_ = y;
        fitted_ = true;
        return *this;
    }

    // ---- 推理阶段 ----
    // 对 X 的每一行：找 k 个最近邻居 → 多数投票 → 预测标签。
    // X 的列数必须等于 fit 时的特征数 d。
    std::vector<Label> predict(const Matrix<T> &X) const;

    // 分类准确率 = 预测对的样本数 / 总样本数（sklearn score 的默认指标）。
    double score(const Matrix<T> &X, const std::vector<Label> &y) const;

    // 想「亲眼看看」每个查询点找到了谁：返回每个查询点的 k 个最近邻居
    //（含行号与距离，按距离升序）。默认用构造时的 k。
    std::vector<std::vector<NeighborHit<T>>>
    kneighbors(const Matrix<T> &X, size_type k = 0) const;

    // ---- 查询 ----
    bool fitted() const { return fitted_; }              // 是否已 fit 过
    size_type n_neighbors() const { return k_; }         // 当前 k 值

private:
    // 私有工具：校验 + 把 X 的每行喂给策略查 k 近邻，逐行收集结果。
    // 返回 vector<vector<NeighborHit>>，外层按行、内层按距离升序。
    std::vector<std::vector<NeighborHit<T>>>
    query_rows(const Matrix<T> &X, size_type k) const;

    // 对一组已按距离升序排好的邻居投票，返回胜出标签（平票取字典序最小）。
    Label vote(const std::vector<NeighborHit<T>> &nbrs) const;

    size_type k_ = 5;              // 投票邻居数 k
    Search<T, Metric> search_;     // 近邻搜索策略（默认 BruteForce）
    std::vector<Label> y_;         // fit 时的标签副本（按行号对齐训练数据）
    bool fitted_ = false;          // 是否已 fit 过
};

// ============================================================================
//  类外实现（模板必须留在头文件里 —— 每个翻译单元都要看到定义才能实例化）
// ============================================================================

template <typename Label, typename T, typename Metric,
          template <typename, typename> class Search>
std::vector<std::vector<NeighborHit<T>>>
KNN<Label, T, Metric, Search>::query_rows(const Matrix<T> &X,
                                          size_type k) const {
    // 必须先 fit；且查询集的列数必须等于训练时的特征数
    CHECK(fitted_) << "KNN::predict: must call fit() before predict()";
    CHECK(X.cols() == search_.feature_count())
        << "KNN::predict: feature count mismatch, got " << X.cols()
        << " cols but trained on " << search_.feature_count();

    const size_type eff_k = (k == 0) ? k_ : k;
    CHECK(eff_k >= 1 && eff_k <= search_.sample_count())
        << "KNN::predict: k (" << eff_k << ") must be in [1, n_samples="
        << search_.sample_count() << "]";

    std::vector<std::vector<NeighborHit<T>>> out(X.rows());
    const std::size_t d = X.cols();
    for (std::size_t i = 0; i < X.rows(); ++i) {
        // Matrix 行主序：第 i 行起点 = data() + i*d
        out[i] = search_.query(X.data() + i * d, eff_k);
    }
    return out;
}

template <typename Label, typename T, typename Metric,
          template <typename, typename> class Search>
Label KNN<Label, T, Metric, Search>::vote(
    const std::vector<NeighborHit<T>> &nbrs) const {
    CHECK(!nbrs.empty()) << "KNN::vote: no neighbors to vote on";
    // map 让标签按 operator< 有序；遍历时先到先计票，
    // 「严格大于才替换」⇒ 并列时保留字典序更小的标签（确定性，同 mode）。
    std::map<Label, size_type> counts;
    for (const NeighborHit<T> &nb : nbrs) {
        CHECK(nb.index < y_.size())
            << "KNN::vote: neighbor index out of range (internal error)";
        ++counts[y_[nb.index]];
    }
    Label best = counts.begin()->first;
    size_type best_count = counts.begin()->second;
    for (const auto &kv : counts) {
        if (kv.second > best_count) {
            best = kv.first;
            best_count = kv.second;
        }
    }
    return best;
}

template <typename Label, typename T, typename Metric,
          template <typename, typename> class Search>
std::vector<Label> KNN<Label, T, Metric, Search>::predict(
    const Matrix<T> &X) const {
    // 逐行查邻居再投票，预测标签与 X 行一一对应
    const auto rows = query_rows(X, k_);
    std::vector<Label> pred(X.rows());
    for (std::size_t i = 0; i < X.rows(); ++i) pred[i] = vote(rows[i]);
    return pred;
}

template <typename Label, typename T, typename Metric,
          template <typename, typename> class Search>
double KNN<Label, T, Metric, Search>::score(
    const Matrix<T> &X, const std::vector<Label> &y) const {
    CHECK(y.size() == X.rows())
        << "KNN::score: y.size() (" << y.size()
        << ") must equal X.rows() (" << X.rows() << ")";
    const std::vector<Label> pred = predict(X);
    std::size_t correct = 0;
    for (std::size_t i = 0; i < pred.size(); ++i) {
        if (pred[i] == y[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(pred.size());
}

template <typename Label, typename T, typename Metric,
          template <typename, typename> class Search>
std::vector<std::vector<NeighborHit<T>>>
KNN<Label, T, Metric, Search>::kneighbors(const Matrix<T> &X,
                                          size_type k) const {
    return query_rows(X, k);
}
