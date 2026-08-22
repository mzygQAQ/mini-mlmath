// perceptron_test.cpp —— 端到端验证 Perceptron 类真的能学会线性可分问题
// 现有 logic_gate.cpp 用的是手写权重，没在测 Perceptron 本身；
// 本文件用 fit() 训练 AND / OR / NAND，验证 predict() 的输出和真值表一致。
// XOR 不在这里测 —— 线性不可分是预期行为（见 logic_gate.cpp 注释）。
#include <cstdio>
#include <vector>

#include "mini_mlmath/ml/perceptron.h"
#include "mini_mlmath/matrix.h"

int main() {
    // 4 个样本的布尔真值表（行 = 样本，列 = 特征）
    const Matrix<float> X = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    // 标签约定：+1 / -1（Perceptron 的 sign 函数定义域，见 perceptron.h 头注释）

    // ---- AND：只有 (1,1) 是 +1 ----
    {
        const std::vector<float> y_and = {-1, -1, -1, 1};
        Perceptron<> p(1.0f, 100);
        p.fit(X, y_and);
        auto pred = p.predict(X);
        std::printf("AND learned weights: [%.3f, %.3f], bias: %.3f\n",
                    p.weights()[0], p.weights()[1], p.bias());
        std::printf("AND predict: ");
        for (auto v : pred) std::printf("%+.0f ", v);
        std::printf("\n");
        for (std::size_t i = 0; i < 4; ++i) {
            if (pred[i] != y_and[i]) {
                std::printf("AND: FAIL at sample %zu\n", i);
                return 1;
            }
        }
        std::printf("AND: OK\n");
    }

    // ---- OR：只有 (0,0) 是 -1 ----
    {
        const std::vector<float> y_or = {-1, 1, 1, 1};
        Perceptron<> p(1.0f, 100);
        p.fit(X, y_or);
        auto pred = p.predict(X);
        std::printf("OR  learned weights: [%.3f, %.3f], bias: %.3f\n",
                    p.weights()[0], p.weights()[1], p.bias());
        for (std::size_t i = 0; i < 4; ++i) {
            if (pred[i] != y_or[i]) {
                std::printf("OR: FAIL at sample %zu\n", i);
                return 1;
            }
        }
        std::printf("OR:  OK\n");
    }

    // ---- NAND：和 AND 反相 ----
    {
        const std::vector<float> y_nand = {1, 1, 1, -1};
        Perceptron<> p(1.0f, 100);
        p.fit(X, y_nand);
        auto pred = p.predict(X);
        std::printf("NAND learned weights: [%.3f, %.3f], bias: %.3f\n",
                    p.weights()[0], p.weights()[1], p.bias());
        for (std::size_t i = 0; i < 4; ++i) {
            if (pred[i] != y_nand[i]) {
                std::printf("NAND: FAIL at sample %zu\n", i);
                return 1;
            }
        }
        std::printf("NAND: OK\n");
    }

    // ---- predict 前没 fit 必须 CHECK 失败（不返回 0，避免掩盖 bug）----
    {
        Perceptron<> p(1.0f, 100);
        try {
            p.predict(X);
            std::printf("FAIL: predict without fit should have thrown\n");
            return 1;
        } catch (const std::invalid_argument &) {
            std::printf("predict-without-fit guard: OK\n");
        }
    }

    std::printf("\nALL OK\n");
    return 0;
}
