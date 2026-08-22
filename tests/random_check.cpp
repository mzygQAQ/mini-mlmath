// random_check —— 验证 Random 对整数 T 的支持是否真离散均匀
// 不走 ctest 的硬断言（不做均值/方差校验），只看「整数真离散」「矩阵版
// 也能编译运行」「量化高斯有合理形状」三件事
#include <cstdio>
#include <map>
#include <mini_mlmath/random.h>

int main() {
    Random rng(42);

    // ---- 标量版：uniform<int> 应该是 [0, 10) 内的整数，每个等概率 ----
    std::map<int, int> hist;
    const int N = 100000;
    for (int i = 0; i < N; ++i) {
        hist[rng.uniform<int>(0, 10)]++;
    }
    std::printf("[uniform<int>(0, 10)] x %d 次：\n", N);
    for (const auto& [k, v] : hist) {
        std::printf("  %d: %d (%.1f%%)\n", k, v, 100.0 * v / N);
    }
    // 期望每个值约 10%，容忍 ±1% 就算 OK

    // ---- 矩阵版 ----
    auto m = rng.uniform_matrix<int>(2, 3, 0, 5);
    std::printf("\n[uniform_matrix<int>(2, 3, 0, 5)]:\n");
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            std::printf("  %d", static_cast<int>(m(i, j)));
        }
        std::printf("\n");
    }

    // ---- normal<int> 量化高斯 ----
    std::map<int, int> nhist;
    for (int i = 0; i < N; ++i) {
        nhist[rng.normal<int>(0, 1)]++;
    }
    std::printf("\n[normal<int>(0, 1)] x %d 次：\n", N);
    for (const auto& [k, v] : nhist) {
        std::printf("  %d: %d (%.1f%%)\n", k, v, 100.0 * v / N);
    }
    // 期望大部分集中在 -2..2，标准差 ~1

    return 0;
}
