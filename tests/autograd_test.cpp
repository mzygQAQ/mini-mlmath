// ============================================================================
//  autograd_test.cpp —— mini-mlmath 的自动求导测试
//
//  验证四件事：
//    1. 逐元素 mul + sum 的链式法则（手算可核对）
//    2. 常数张量（track=false）不累积梯度
//    3. 一个叶子被多个节点引用（fan-out）时梯度正确累加
//    4. 一个小型 MLP（matmul -> relu -> matmul -> sum）的梯度与
//       「纯数值有限差分」对拍 —— 这是最有力的正确性验证
//
//  实现见 autograd.h 头部注释。
// ============================================================================
#include "mini_mlmath/autograd.h"

#include <cmath>
#include <iostream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// ----------------------------------------------------------------------------
// 1) 逐元素乘 + 求和：手算
//   a = [1, 2], w = [3, 4]
//   y = a ⊙ w = [3, 8],  loss = sum(y) = 11
//   ∂loss/∂a_i = w_i = [3, 4],  ∂loss/∂w_i = a_i = [1, 2]
// ----------------------------------------------------------------------------
void verify_mul_sum() {
    std::cout << "===== 1) elementwise mul + sum（手算核对）=====\n";

    Tensor<double> a(Matrix<double>({{1.0, 2.0}}), true);
    Tensor<double> w(Matrix<double>({{3.0, 4.0}}), true);

    Tensor<double> y = a * w;        // 逐元素乘
    Tensor<double> loss = sum(y);    // 11
    loss.backward();

    std::cout << "loss = " << loss.item() << "（期望 11）\n";
    std::cout << "y    = " << y.value();
    std::cout << "a.grad = " << a.grad() << "（期望 [3, 4]）\n";
    std::cout << "w.grad = " << w.grad() << "（期望 [1, 2]）\n";

    const bool ok = (std::abs(loss.item() - 11.0) < 1e-12)
                 && (std::abs(a.grad()(0, 0) - 3.0) < 1e-12)
                 && (std::abs(a.grad()(0, 1) - 4.0) < 1e-12)
                 && (std::abs(w.grad()(0, 0) - 1.0) < 1e-12)
                 && (std::abs(w.grad()(0, 1) - 2.0) < 1e-12);
    std::cout << (ok ? "  ✓\n" : "  ✗\n");
    if (!ok) std::exit(1);
}

// ----------------------------------------------------------------------------
// 2) 常数不追踪：x 是常数（track=false），梯度只进 w
//   loss = sum(x ⊙ w)，∂loss/∂w_i = x_i
// ----------------------------------------------------------------------------
void verify_constant() {
    std::cout << "===== 2) 常数张量不累积梯度 =====\n";

    Tensor<double> x(Matrix<double>({{1.0, 2.0}}), false);  // 常数（数据）
    Tensor<double> w(Matrix<double>({{3.0, 4.0}}), true);   // 参数（要梯度）

    Tensor<double> loss = sum(x * w);
    loss.backward();

    std::cout << "w.grad = " << w.grad() << "（期望 [1, 2] = x）\n";

    const bool ok = (std::abs(w.grad()(0, 0) - 1.0) < 1e-12)
                 && (std::abs(w.grad()(0, 1) - 2.0) < 1e-12);
    std::cout << (ok ? "  ✓\n" : "  ✗\n");
    if (!ok) std::exit(1);
}

// ----------------------------------------------------------------------------
// 3) fan-out：同一个叶子 w 被两个算子引用
//   y1 = w*3, y2 = w*4, loss = sum(y1 + y2)
//   ∂loss/∂w = 3 + 4 = 7（每个消费者各贡献一份，正确累加）
// ----------------------------------------------------------------------------
void verify_fanout() {
    std::cout << "===== 3) fan-out（一个叶子被多个节点引用）=====\n";

    Tensor<double> w(Matrix<double>({{2.0}}), true);
    Tensor<double> y1 = w * 3.0;
    Tensor<double> y2 = w * 4.0;
    Tensor<double> loss = sum(y1 + y2);   // (6+8)=14
    loss.backward();

    std::cout << "loss   = " << loss.item() << "（期望 14）\n";
    std::cout << "w.grad = " << w.grad()(0, 0) << "（期望 7）\n";

    const bool ok = (std::abs(loss.item() - 14.0) < 1e-12)
                 && (std::abs(w.grad()(0, 0) - 7.0) < 1e-12);
    std::cout << (ok ? "  ✓\n" : "  ✗\n");
    if (!ok) std::exit(1);
}

// ----------------------------------------------------------------------------
// 4) 减 / 负号：手算
//   a = [5, 3], b = [2, 1]，c = a - b = [3, 2]，loss = 5
//   ∂loss/∂a = [1, 1], ∂loss/∂b = [-1, -1]；∂loss/∂a 用 -a 再验一遍
// ----------------------------------------------------------------------------
void verify_sub_neg() {
    std::cout << "===== 4) 减 / 一元负号 =====\n";

    Tensor<double> a(Matrix<double>({{5.0, 3.0}}), true);
    Tensor<double> b(Matrix<double>({{2.0, 1.0}}), true);
    Tensor<double> c = a - b;
    Tensor<double> loss = sum(c);
    loss.backward();

    const bool ok = (std::abs(loss.item() - 5.0) < 1e-12)
                 && (std::abs(a.grad()(0, 0) - 1.0) < 1e-12)
                 && (std::abs(a.grad()(0, 1) - 1.0) < 1e-12)
                 && (std::abs(b.grad()(0, 0) + 1.0) < 1e-12)
                 && (std::abs(b.grad()(0, 1) + 1.0) < 1e-12);
    std::cout << "a.grad = " << a.grad() << "（期望 [1, 1]）\n";
    std::cout << "b.grad = " << b.grad() << "（期望 [-1, -1]）\n";
    std::cout << (ok ? "  ✓\n" : "  ✗\n");
    if (!ok) std::exit(1);
}

// ----------------------------------------------------------------------------
// 5) zero_grad 与多轮累积语义
//   同一叶子 w 连续两次 backward，梯度累积（7 -> 9）；zero_grad 后归零。
// ----------------------------------------------------------------------------
void verify_accumulate_and_zero_grad() {
    std::cout << "===== 5) 多轮累积 + zero_grad =====\n";

    Tensor<double> w(Matrix<double>({{2.0}}), true);

    auto run = [&](double s) {
        auto loss = sum(w * s);
        loss.backward();
    };

    run(3.0);
    run(4.0);
    std::cout << "两次 backward 后 w.grad = " << w.grad()(0, 0)
              << "（期望 7 = 3+4，累积语义）\n";
    const bool ok1 = (std::abs(w.grad()(0, 0) - 7.0) < 1e-12);

    w.zero_grad();
    std::cout << "zero_grad 后 w.grad    = " << w.grad()(0, 0)
              << "（期望 0）\n";
    const bool ok2 = (std::abs(w.grad()(0, 0)) < 1e-12);

    std::cout << ((ok1 && ok2) ? "  ✓\n" : "  ✗\n");
    if (!(ok1 && ok2)) std::exit(1);
}

// ----------------------------------------------------------------------------
// 6) 小型 MLP 数值对拍（最强验证）
//   两层：h = relu(X @ W1)，out = h @ W2，loss = sum(out)
//   X: 1×2（常数），W1: 2×3（参数），W2: 3×1（参数）
//   用「纯数值有限差分」算 ∂loss/∂W1、∂loss/∂W2，与 autograd 对拍。
//   选的 W1 让 h 全 > 0，避开 relu 在 0 处不可导的边界。
// ----------------------------------------------------------------------------

// 纯矩阵版前向（不经过 autograd，用 Matrix<T> 直接算），返回标量 loss
double plain_forward(const Matrix<double>& W1, const Matrix<double>& W2) {
    const Matrix<double> X({{1.0, -0.5}});
    Matrix<double> h = relu(X * W1);     // 1×3
    Matrix<double> o = h * W2;           // 1×1
    return o(0, 0);
}

void verify_mlp_numeric() {
    std::cout << "===== 6) 小型 MLP 梯度 vs 数值有限差分 =====\n";

    const Matrix<double> W1({{0.8, 0.6, 0.4},
                             {0.6, 0.4, 0.2}});   // 2×3
    const Matrix<double> W2({{0.5},
                             {0.3},
                             {0.2}});             // 3×1

    // 先手动算 h = X@W1 = [0.5, 0.4, 0.3]（全 > 0，relu 不动），
    // 因此数值差分不必担心跨过不可导点。
    {
        Matrix<double> h = relu(Matrix<double>({{1.0, -0.5}}) * W1);
        std::cout << "h = " << h << "（期望 [0.5, 0.4, 0.3]，全正）\n";
        const bool h_ok = (std::abs(h(0, 0) - 0.5) < 1e-12)
                       && (std::abs(h(0, 1) - 0.4) < 1e-12)
                       && (std::abs(h(0, 2) - 0.3) < 1e-12);
        if (!h_ok) {
            std::cout << "  ✗ h 的手算核对失败，测试假设不成立\n";
            std::exit(1);
        }
    }

    // autograd 前向 + 反向
    Tensor<double> x(Matrix<double>({{1.0, -0.5}}), false);  // 常数输入
    Tensor<double> w1(W1, true);
    Tensor<double> w2(W2, true);
    Tensor<double> h = relu(matmul(x, w1));
    Tensor<double> loss = sum(matmul(h, w2));
    loss.backward();

    std::cout << "loss = " << loss.item()
              << "（期望 0.43 = 0.25+0.12+0.06）\n";
    if (std::abs(loss.item() - 0.43) > 1e-12) {
        std::cout << "  ✗ loss 手算核对失败\n";
        std::exit(1);
    }

    // 中央差分求数值梯度，与 autograd 对拍
    double max_err_w1 = 0.0, max_err_w2 = 0.0;
    const double hh = 1e-6;

    for (std::size_t i = 0; i < W1.rows(); ++i) {
        for (std::size_t j = 0; j < W1.cols(); ++j) {
            Matrix<double> Wp = W1, Wm = W1;
            Wp(i, j) += hh;
            Wm(i, j) -= hh;
            const double num = (plain_forward(Wp, W2) - plain_forward(Wm, W2)) / (2 * hh);
            const double ana = w1.grad()(i, j);
            const double err = std::abs(num - ana);
            if (err > max_err_w1) max_err_w1 = err;
        }
    }
    for (std::size_t i = 0; i < W2.rows(); ++i) {
        for (std::size_t j = 0; j < W2.cols(); ++j) {
            Matrix<double> Wp = W2, Wm = W2;
            Wp(i, j) += hh;
            Wm(i, j) -= hh;
            const double num = (plain_forward(W1, Wp) - plain_forward(W1, Wm)) / (2 * hh);
            const double ana = w2.grad()(i, j);
            const double err = std::abs(num - ana);
            if (err > max_err_w2) max_err_w2 = err;
        }
    }

    std::cout << "max |数值梯度 - autograd| : W1 = " << max_err_w1
              << ", W2 = " << max_err_w2 << "\n";
    const bool ok = (max_err_w1 < 1e-6) && (max_err_w2 < 1e-6);
    std::cout << (ok ? "  ✓\n" : "  ✗\n");
    if (!ok) std::exit(1);
}

// ----------------------------------------------------------------------------
// 入口
// ----------------------------------------------------------------------------
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::cout << "mini-mlmath —— 教学用迷你矩阵库（自动求导测试）\n\n";

    verify_mul_sum();
    verify_constant();
    verify_fanout();
    verify_sub_neg();
    verify_accumulate_and_zero_grad();
    verify_mlp_numeric();

    std::cout << "\n全部通过 ✓\n";
    std::cout << "API：Tensor<T>(matrix, track) 造张量，a+b/a*b/matmul/relu/sum 建图，\n";
    std::cout << "loss.backward() 反向，leaf.grad() 读梯度（实现见 autograd.h）。\n";
    return 0;
}
