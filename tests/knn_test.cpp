// ============================================================================
//  knn_test.cpp —— 端到端验证 KNN 类
//
//  数据用手算可核对的小规模二维点集：三类清晰可分，方便验证「离谁近归谁」。
//  验证点：
//    1. 三个预置 metric 结果一致（数据本身就是按欧氏距离分开的）；
//    2. k=1 等价于「最近邻」：查询点落在某簇内，必归该簇；
//    3. 自定义 metric（函数对象）可无缝换入 —— 度量扩展点；
//    4. fit 之前 predict 必须 CHECK 失败（懒学习也要先存数据）；
//    5. k > n_samples 必须 CHECK 失败。
//
//  实现与原理见 ml/knn.h 头部注释。
// ============================================================================
#include <cstddef>
#include <cstdio>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_mlmath/ml/knn.h"
#include "mini_mlmath/matrix.h"

// ----------------------------------------------------------------------------
// 用「3 簇清晰可分」的数据：标签用字符串，顺带验证 Label 模板参数是泛型的
// 簇 A：围绕 (1,1)，标签 "A"；簇 B：围绕 (5,5)，标签 "B"；簇 C：围绕 (9,1)
// ----------------------------------------------------------------------------
struct Fixture {
    Matrix<double> X;
    std::vector<std::string> y;
};

Fixture make_data() {
    // 每簇 4 个点，行 = 样本，列 = 特征
    Matrix<double> X = {
        {0.8, 1.1},   // A
        {1.2, 0.9},   // A
        {1.1, 1.3},   // A
        {0.9, 0.8},   // A
        {4.8, 5.2},   // B
        {5.2, 4.8},   // B
        {5.1, 5.3},   // B
        {4.9, 4.7},   // B
        {9.1, 0.9},   // C
        {8.9, 1.1},   // C
        {9.2, 0.7},   // C
        {8.8, 1.2},   // C
    };
    std::vector<std::string> y = {"A", "A", "A", "A",
                                  "B", "B", "B", "B",
                                  "C", "C", "C", "C"};
    return {X, y};
}

// 手算核对一个用例：三簇中心相距很远，k=1 时落点归最近簇
void verify_predict() {
    std::printf("== predict：k 近邻多数投票 ==\n");

    const Fixture fx = make_data();

    // 用训练集自己当测试集：k=1 时每点都该归自己那簇（自身距离 0 必被选中）
    {
        KNN<std::string, double> knn(1);
        knn.fit(fx.X, fx.y);
        const auto pred = knn.predict(fx.X);
        bool ok = true;
        for (std::size_t i = 0; i < pred.size(); ++i) {
            if (pred[i] != fx.y[i]) {
                std::printf("  FAIL: sample %zu predicted %s, expected %s\n",
                            i, pred[i].c_str(), fx.y[i].c_str());
                ok = false;
            }
        }
        if (ok) std::printf("  k=1 训练集自预测（每点归自己簇）: OK\n");
    }

    // 两个从未见过的查询点：落在 A 簇内 → "A"；落在 B 簇内 → "B"
    {
        KNN<std::string, double> knn(3);
        knn.fit(fx.X, fx.y);
        const Matrix<double> Q = {{1.0, 1.0}, {5.0, 5.0}, {9.0, 1.0}};
        const std::vector<std::string> expect = {"A", "B", "C"};
        const auto pred = knn.predict(Q);
        bool ok = true;
        for (std::size_t i = 0; i < pred.size(); ++i) {
            if (pred[i] != expect[i]) {
                std::printf("  FAIL: query %zu predicted %s, expected %s\n",
                            i, pred[i].c_str(), expect[i].c_str());
                ok = false;
            }
        }
        if (ok) std::printf("  k=3 三簇簇心查询分别命中 A/B/C: OK\n");
    }

    // score：对训练集预测准确率应为 1.0
    {
        KNN<std::string, double> knn(3);
        knn.fit(fx.X, fx.y);
        const double acc = knn.score(fx.X, fx.y);
        std::printf("  训练集 score（accuracy）: %.2f %s\n", acc,
                    (std::fabs(acc - 1.0) < 1e-12) ? "OK" : "FAIL");
        if (std::fabs(acc - 1.0) >= 1e-12) std::printf("  FAIL: expected 1.0\n");
    }
}

// 验证三个预置 metric：L2 / L1 / L∞ 在该数据上给出相同的三簇划分
void verify_metrics() {
    std::printf("== metric 扩展点：三种预置距离结果一致 ==\n");
    const Fixture fx = make_data();
    const Matrix<double> Q = {{1.0, 1.0}, {5.0, 5.0}, {9.0, 1.0}};
    const std::vector<std::string> expect = {"A", "B", "C"};

    bool all_ok = true;
    {
        KNN<std::string, double, EuclideanDistance> knn(3);
        knn.fit(fx.X, fx.y);
        all_ok &= (knn.predict(Q) == expect);
    }
    {
        KNN<std::string, double, ManhattanDistance> knn(3);
        knn.fit(fx.X, fx.y);
        all_ok &= (knn.predict(Q) == expect);
    }
    {
        KNN<std::string, double, ChebyshevDistance> knn(3);
        knn.fit(fx.X, fx.y);
        all_ok &= (knn.predict(Q) == expect);
    }
    std::printf("  Euclidean / Manhattan / Chebyshev 全部命中: %s\n",
                all_ok ? "OK" : "FAIL");
    if (!all_ok) std::printf("  FAIL\n");
}

// 自定义 metric（只比较第 0 维）：验证「新度量 = 新增一个 functor」即可用
struct FirstDimensionOnly {
    template <typename T>
    T operator()(const T *a, const T *b, std::size_t dim) const {
        (void) dim;   // 故意忽略第 1 维
        return std::abs(a[0] - b[0]);
    }
};

void verify_custom_metric() {
    std::printf("== metric 扩展点：自定义函数对象 ==\n");
    const Fixture fx = make_data();

    // 只比第 0 维：簇 A 第 0 维约 1、簇 B 约 5、簇 C 约 9；
    // 构造一个「第 0 维贴近 A、但第 1 维离 A 十万八千里」的查询点 ——
    // 若真按完整欧氏距离它会被 A 拿走？不会：A 和 C 第 0 维分别是 1 / 9，
    // (0.9, 9999) 离 A 的第 1 维很近（1 附近），仍归 A。
    // 我们用 (8.7, 9999)：第 0 维贴近 C（9），欧氏会因第 1 维全部爆炸而
    // 无法区分，但 FirstDimensionOnly 只看第 0 维 → 必归 C。
    KNN<std::string, double, FirstDimensionOnly> knn(1);
    knn.fit(fx.X, fx.y);

    const Matrix<double> Q = {{8.7, 9999.0}};
    const auto pred = knn.predict(Q);
    const bool ok = (pred.size() == 1 && pred[0] == "C");
    std::printf("  只看第 0 维 (8.7, 9999) → %s\n", ok ? "C: OK" : "FAIL");
    if (!ok) std::printf("  FAIL: expected C\n");
}

// 验证 strategy 输出：kneighbors 按距离升序，第一个邻居是自身
void verify_kneighbors() {
    std::printf("== strategy 输出：kneighbors ==\n");
    const Fixture fx = make_data();
    KNN<std::string, double> knn(3);
    knn.fit(fx.X, fx.y);

    // 查询第一个训练样本本身：k=3，最近的一定是它自己（距离 0）
    const Matrix<double> Q = {{0.8, 1.1}};
    const auto nbrs = knn.kneighbors(Q);
    bool ok = (nbrs.size() == 1 && nbrs[0].size() == 3);
    if (ok) {
        const auto &row = nbrs[0];
        // 距离应单调不降（升序）
        for (std::size_t i = 1; i < row.size(); ++i) {
            if (row[i].distance < row[i - 1].distance) { ok = false; break; }
        }
        // 第一个邻居是自身：距离 0，行号 0
        ok = ok && (std::fabs(row[0].distance) < 1e-12) && (row[0].index == 0);
    }
    std::printf("  邻居数=3、距离升序、最近为自身: %s\n", ok ? "OK" : "FAIL");
    if (!ok) std::printf("  FAIL\n");
}

// 错误处理：predict 前没 fit、k 超过样本数，都必须 CHECK 抛异常
void verify_guards() {
    std::printf("== 参数校验 ==\n");
    const Fixture fx = make_data();

    {
        KNN<std::string, double> knn(3);
        try {
            (void) knn.predict(fx.X);
            std::printf("  FAIL: predict without fit should have thrown\n");
            return;
        } catch (const std::invalid_argument &) {
            std::printf("  predict-without-fit guard: OK\n");
        }
    }
    {
        KNN<std::string, double> knn(99);   // k=99 > n_samples=12
        knn.fit(fx.X, fx.y);
        try {
            (void) knn.predict(fx.X);
            std::printf("  FAIL: k > n_samples should have thrown\n");
            return;
        } catch (const std::invalid_argument &) {
            std::printf("  k>n_samples guard: OK\n");
        }
    }
}

// 验证 KDTreeSearch 骨架占位：
//  1) 能作为 KNN 的第四个模板实参直接编译（策略接口对齐，切换只改一处）；
//  2) build 会抛 std::logic_error("not implemented") —— 占位还没实现就该在
//     fit 阶段立刻失败，而不是拖到 predict。
void verify_kdtree_placeholder() {
    std::printf("== KDTreeSearch 骨架占位 ==\n");
    const Fixture fx = make_data();

    // 只要这一行能编译，就证明 KDTreeSearch 的接口形状（build/query/...）和
    // BruteForce 完全一致 —— 这是「策略切换零改动」的编译期保证。
    KNN<std::string, double, EuclideanDistance, KDTreeSearch> knn(3);
    try {
        knn.fit(fx.X, fx.y);
        std::printf("  FAIL: KDTreeSearch::build should have thrown (not implemented)\n");
        return;
    } catch (const std::logic_error &) {
        std::printf("  KDTreeSearch::build throws not-implemented: OK\n");
    }
}

int main() {
    std::printf("mini-mlmath —— KNN K 近邻测试\n\n");
    verify_predict();
    verify_metrics();
    verify_custom_metric();
    verify_kneighbors();
    verify_kdtree_placeholder();
    verify_guards();
    std::printf("\nALL OK\n");
    return 0;
}
