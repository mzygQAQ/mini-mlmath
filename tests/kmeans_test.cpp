// ============================================================================
//  kmeans_test.cpp —— 端到端验证 KMeans 类
//
//  数据用手算可核对的小规模二维点集：三簇清晰可分（和 knn_test 同款布局）。
//  骨架期：功能测试里 fit/predict 会抛 std::logic_error("not implemented")，
//  被统一 catch 成 SKIPPED；等你实现完 kmeans.h，本文件自动变成验收清单 ——
//  每个 SKIPPED 会变成真实的 OK / FAIL，无需改动测试本身。
//
//  验证点：
//    1. 骨架占位：fit 抛 not-implemented（占位就该在 fit 阶段立刻失败）；
//    2. predict：训练后每个样本归自己那簇（按「每 4 个连续样本同簇」核对，
//       因为 k-means 的簇号 0/1/2 是任意排列的，不能死板对号）；
//    3. centers：学出的簇心接近各簇的算术平均（欧氏配均值的自洽性）；
//    4. fit_predict 与 predict 结果一致；
//    5. inertia / score：手算核对（三簇完全可分时惯性 ≈ 0.855，见下）；
//    6. 参数校验：没 fit 就 predict、n_clusters 超界，都必须 CHECK 失败。
//
//  实现与原理见 ml/kmeans.h 头部注释和 docs/kmeans.md。
// ============================================================================
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_mlmath/matrix.h"
#include "mini_mlmath/ml/kmeans.h"

// 三簇清晰可分（同 knn_test 的数据）：
//   簇 0：围绕 (1.0, 1.0)；簇 1：围绕 (5.0, 5.0)；簇 2：围绕 (9.0, 1.0)
// 每簇 4 个点，行 = 样本、列 = 特征。无监督：没有标签。
struct Fixture {
    Matrix<double> X;
};

/**
 * @brief 构造三簇清晰可分的二维测试数据
 * @return Fixture：12 个样本、2 维特征，前 4 行一簇、中间 4 行一簇、后 4 行一簇
 */
Fixture make_data() {
    Matrix<double> X = {
        {0.8, 1.1}, // 簇 0
        {1.2, 0.9}, // 簇 0
        {1.1, 1.3}, // 簇 0
        {0.9, 0.8}, // 簇 0
        {4.8, 5.2}, // 簇 1
        {5.2, 4.8}, // 簇 1
        {5.1, 5.3}, // 簇 1
        {4.9, 4.7}, // 簇 1
        {9.1, 0.9}, // 簇 2
        {8.9, 1.1}, // 簇 2
        {9.2, 0.7}, // 簇 2
        {8.8, 1.2}, // 簇 2
    };
    return {X};
}

/**
 * @brief 验证骨架占位：fit 必须抛 not-implemented（骨架期的正确姿势）
 */
void verify_placeholder() {
    std::printf("== 骨架占位：fit 抛 not-implemented ==\n");
    const Fixture fx = make_data();
    KMeans<double> km(3);
    try {
        km.fit(fx.X);
        std::printf("  FAIL: KMeans::fit should have thrown (not implemented)\n");
    } catch (const std::logic_error &) {
        std::printf("  KMeans::fit throws not-implemented: OK\n");
    }
}

/**
 * @brief 验证 predict：训练后每 4 个连续样本归同一簇（簇号任意但分组正确）
 */
void verify_predict() {
    std::printf("== predict：训练后每样本归自己簇 ==\n");
    const Fixture fx = make_data();
    try {
        KMeans<double> km(3);
        km.fit(fx.X);
        const auto lab = km.predict(fx.X);
        // k-means 的簇号 0/1/2 是任意排列的，所以按「每 4 个连续样本同簇」
        // 核对，而不是死板对号：0-3 同簇、4-7 同簇、8-11 同簇。
        bool ok = (lab.size() == fx.X.rows());
        for (std::size_t g = 0; g < 3 && ok; ++g) {
            for (std::size_t i = g * 4 + 1; i < g * 4 + 4; ++i) {
                if (lab[i] != lab[g * 4])
                    ok = false;
            }
        }
        std::printf("  三组各 4 点同簇: %s\n", ok ? "OK" : "FAIL");
        if (!ok)
            std::printf("  FAIL: expected each group of 4 rows to share a cluster\n");
    } catch (const std::logic_error &) {
        std::printf("  SKIPPED (not implemented)\n");
    }
}

/**
 * @brief 验证 centers：学出的簇心接近各簇的算术平均（簇号乱序需逐个匹配）
 */
void verify_centers() {
    std::printf("== centers：接近各簇算术平均 ==\n");
    const Fixture fx = make_data();
    // 每簇的精确均值（手算）：簇 0 = (1.0, 1.025)、簇 1 = (5.0, 5.0)、簇 2 = (9.0, 0.975)
    const std::vector<std::pair<double, double>> expect = {
        {1.0, 1.025}, {5.0, 5.0}, {9.0, 0.975}};
    try {
        KMeans<double> km(3);
        km.fit(fx.X);
        const Matrix<double> C = km.centers();
        bool ok = (C.rows() == 3 && C.cols() == 2);
        // 簇号 0/1/2 是任意排列的：对每个期望均值点，找离它最近且未用过的
        // 学出簇心，距离 < 0.05 才算命中（三簇相距很远，容差很安全）。
        std::vector<bool> used(C.rows(), false);
        for (const auto &e : expect) {
            double best = 1e30;
            std::size_t best_j = 0;
            for (std::size_t j = 0; j < C.rows(); ++j) {
                if (used[j])
                    continue;
                const double dx = C(j, 0) - e.first;
                const double dy = C(j, 1) - e.second;
                const double d = std::sqrt(dx * dx + dy * dy);
                if (d < best) {
                    best = d;
                    best_j = j;
                }
            }
            if (best < 0.05) {
                used[best_j] = true;
            } else
                ok = false;
        }
        std::printf("  簇心贴近各簇均值: %s\n", ok ? "OK" : "FAIL");
        if (!ok)
            std::printf("  FAIL: expected centers near (1,1.025)/(5,5)/(9,0.975)\n");
    } catch (const std::logic_error &) {
        std::printf("  SKIPPED (not implemented)\n");
    }
}

/**
 * @brief 验证 fit_predict：与「先 fit 再 predict」的结果一致
 */
void verify_fit_predict() {
    std::printf("== fit_predict 与 predict 一致 ==\n");
    const Fixture fx = make_data();
    try {
        KMeans<double> km(3);
        const auto fp = km.fit_predict(fx.X);
        const auto pr = km.predict(fx.X);
        const bool ok = (fp == pr);
        std::printf("  fit_predict == predict: %s\n", ok ? "OK" : "FAIL");
        if (!ok)
            std::printf("  FAIL\n");
    } catch (const std::logic_error &) {
        std::printf("  SKIPPED (not implemented)\n");
    }
}

/**
 * @brief 验证 inertia / score：三簇完全可分时惯性 ≈ 0.855，score = -inertia
 *
 * 手算（每簇 4 点到其均值的距离平方和）：
 *   簇 0：均值 (1.0, 1.025) → SSE = 0.2475
 *   簇 1：均值 (5.0, 5.0)   → SSE = 0.36
 *   簇 2：均值 (9.0, 0.975) → SSE = 0.2475
 *   总惯性 = 0.2475 + 0.36 + 0.2475 = 0.855
 */
void verify_inertia_score() {
    std::printf("== inertia / score：手算核对 ==\n");
    const Fixture fx = make_data();
    const double expected_inertia = 0.855;
    try {
        KMeans<double> km(3);
        km.fit(fx.X);
        const double iner = km.inertia();
        const double scr = km.score(fx.X);
        bool ok = (std::fabs(iner - expected_inertia) < 1e-3) && (std::fabs(scr + expected_inertia) < 1e-3);
        std::printf("  inertia=%.4f (期望 %.4f), score=%.4f: %s\n",
                    iner, expected_inertia, scr, ok ? "OK" : "FAIL");
        if (!ok)
            std::printf("  FAIL\n");
    } catch (const std::logic_error &) {
        std::printf("  SKIPPED (not implemented)\n");
    }
}

/**
 * @brief 验证参数校验：没 fit 就 predict、n_clusters 超界，都必须 CHECK 失败
 */
void verify_guards() {
    std::printf("== 参数校验 ==\n");
    const Fixture fx = make_data();

    {
        KMeans<double> km(3);
        try {
            (void)km.predict(fx.X);
            std::printf("  FAIL: predict without fit should have thrown\n");
            return;
        } catch (const std::invalid_argument &) {
            std::printf("  predict-without-fit guard: OK\n");
        }
    }
    {
        KMeans<double> km(99); // n_clusters=99 > n_samples=12
        try {
            (void)km.fit(fx.X);
            std::printf("  FAIL: n_clusters > n_samples should have thrown\n");
            return;
        } catch (const std::invalid_argument &) {
            std::printf("  n_clusters>n_samples guard: OK\n");
        }
    }
}

/**
 * @brief 测试程序入口
 * @return 0（骨架期功能测试显示 SKIPPED 是预期行为，不视为失败）
 */
int main() {
    std::printf("mini-mlmath —— K-Means K 均值聚类测试（骨架期）\n\n");
    verify_placeholder();
    verify_predict();
    verify_centers();
    verify_fit_predict();
    verify_inertia_score();
    verify_guards();
    std::printf("\nDONE（骨架期：功能测试为 SKIPPED，实现 kmeans.h 后自动验收）\n");
    return 0;
}
