// ============================================================================
//  rope_test.cpp —— 验证旋转位置编码 RoPE
//
//  数据用手算可核对的小矩阵（d_model=4，θ = [1, 0.01]）。骨架期：核心
//  函数抛 std::logic_error("not implemented")，被统一 catch 成 SKIPPED；
//  等你实现完 rope.h，本文件自动变成验收清单，无需改动。
//
//  手算依据（公式见 rope.h 文件头）：
//    d_model=4：theta[0] = 1/10000^0 = 1，theta[1] = 1/10000^(2/4) = 0.01
//    位置 0：不旋转（角度 0），输出 == 输入
//    位置 1，输入 (1,2,3,4)：
//      (0,1) 对旋转 1 rad：
//        new0 = 1·cos1 − 2·sin1 ≈ −1.142640
//        new1 = 1·sin1 + 2·cos1 ≈  1.922076
//      (2,3) 对旋转 0.01 rad：
//        new2 = 3·cos(0.01) − 4·sin(0.01) ≈  2.959851
//        new3 = 3·sin(0.01) + 4·cos(0.01) ≈  4.029800
//
//  验证点（第 4 条是 RoPE 的灵魂性质）：
//    1. 骨架占位：函数抛 not-implemented；
//    2. 位置 0 恒等：第 0 行不旋转；
//    3. 位置 1 的手算数值点（容差 1e-6）；
//    4. 相对位置性质：<R(m)q, R(n)k> == <q, R(n−m)k>（绝对位置在点积里相消）；
//    5. 保范数：旋转是正交变换，每行的 2-范数不变；
//    6. 参数校验：奇数列 / 空（0 行）矩阵必须 CHECK 失败。
//
//  实现与原理见 rope.h 头部注释和 docs/rope.md。
// ============================================================================
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "mini_mlmath/matrix.h"
#include "mini_mlmath/rope.h"

// 手算用的固定尺寸：theta = [1, 0.01]
constexpr std::size_t kRows = 2;
constexpr std::size_t kDim = 4;

/**
 * @brief 验证骨架占位：函数必须抛 not-implemented（骨架期的正确姿势）
 */
void verify_placeholder() {
    std::printf("== 骨架占位：抛 not-implemented ==\n");
    const Matrix<double> X = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    try {
        (void) rope_rotate(X);
        std::printf("  FAIL: should have thrown (not implemented)\n");
    } catch (const std::logic_error &) {
        std::printf("  throws not-implemented: OK\n");
    }
}

/**
 * @brief 验证位置 0 恒等：第 0 行角度为 0，不旋转
 */
void verify_pos0_identity() {
    std::printf("== 位置 0 恒等：第 0 行不旋转 ==\n");
    try {
        const Matrix<double> X = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        const Matrix<double> out = rope_rotate(X);
        bool ok = true;
        for (std::size_t j = 0; j < kDim; ++j) {
            if (std::fabs(out(0, j) - X(0, j)) > 1e-12) ok = false;
        }
        std::printf("  第 0 行原样返回: %s\n", ok ? "OK" : "FAIL");
        if (!ok) std::printf("  FAIL: expected row 0 unchanged\n");
    } catch (const std::logic_error &) {
        std::printf("  SKIPPED (not implemented)\n");
    }
}

/**
 * @brief 验证位置 1 的手算数值点（theta=[1, 0.01]，输入 (1,2,3,4)）
 */
void verify_known_values() {
    std::printf("== 位置 1 手算数值点 ==\n");
    try {
        const Matrix<double> X = {{1, 2, 3, 4}, {1, 2, 3, 4}};
        const Matrix<double> out = rope_rotate(X);
        const double expect[kDim] = {-1.142640, 1.922076, 2.959851, 4.029800};
        bool ok = true;
        for (std::size_t j = 0; j < kDim; ++j) {
            if (std::fabs(out(1, j) - expect[j]) > 1e-6) {
                std::printf("  FAIL: out(1,%zu)=%.9f, expected %.9f\n",
                            j, out(1, j), expect[j]);
                ok = false;
            }
        }
        std::printf("  4 个手算点全部命中: %s\n", ok ? "OK" : "FAIL");
        if (!ok) std::printf("  FAIL\n");
    } catch (const std::logic_error &) {
        std::printf("  SKIPPED (not implemented)\n");
    }
}

// 小工具：两个向量的点积（1×d 矩阵的行）
double dot(const Matrix<double> &a, std::size_t ra,
           const Matrix<double> &b, std::size_t rb) {
    double s = 0.0;
    for (std::size_t j = 0; j < a.cols(); ++j) s += a(ra, j) * b(rb, j);
    return s;
}

// 小工具：一行的 2-范数
double row_norm(const Matrix<double> &a, std::size_t r) {
    double s = 0.0;
    for (std::size_t j = 0; j < a.cols(); ++j) s += a(r, j) * a(r, j);
    return std::sqrt(s);
}

/**
 * @brief 验证 RoPE 灵魂性质：<R(m)q, R(n)k> == <q, R(n-m)k>
 *
 * 构造 6 行全相同的 q（或 k），rope_rotate 后第 m 行就是 R(m)q——
 * 这样不用显式传位置就能取到任意 (m, n) 的旋转结果。
 */
void verify_relative_property() {
    std::printf("== 相对位置性质：<R(m)q, R(n)k> == <q, R(n-m)k> ==\n");
    try {
        const std::vector<double> q = {1, 2, 3, 4};
        const std::vector<double> k = {5, 6, 7, 8};
        Matrix<double> Q(kRows + 4, kDim);   // 6 行，位置 0..5
        Matrix<double> K(kRows + 4, kDim);
        for (std::size_t m = 0; m < Q.rows(); ++m) {
            for (std::size_t j = 0; j < kDim; ++j) {
                Q(m, j) = q[j];
                K(m, j) = k[j];
            }
        }
        const Matrix<double> Qr = rope_rotate(Q);   // 第 m 行 = R(m)·q
        const Matrix<double> Kr = rope_rotate(K);   // 第 n 行 = R(n)·k

        // 取 m=2, n=5：左边 <R(2)q, R(5)k>，右边 <q, R(3)k>（q 即 Qr 第 0 行）
        const double lhs = dot(Qr, 2, Kr, 5);
        const double rhs = dot(Qr, 0, Kr, 3);
        const bool ok = std::fabs(lhs - rhs) < 1e-9;
        std::printf("  <R(2)q,R(5)k>=%.9f vs <q,R(3)k>=%.9f: %s\n",
                    lhs, rhs, ok ? "OK" : "FAIL");
        if (!ok) std::printf("  FAIL: absolute positions did not cancel\n");
    } catch (const std::logic_error &) {
        std::printf("  SKIPPED (not implemented)\n");
    }
}

/**
 * @brief 验证保范数：旋转是正交变换，每行 2-范数不变
 */
void verify_norm_preserved() {
    std::printf("== 保范数：每行 2-范数不变 ==\n");
    try {
        const Matrix<double> X = {{1, 2, 3, 4}, {5, 6, 7, 8}, {0.5, -3, 2, 1}};
        const Matrix<double> out = rope_rotate(X);
        bool ok = true;
        for (std::size_t m = 0; m < X.rows(); ++m) {
            const double n1 = row_norm(X, m);
            const double n2 = row_norm(out, m);
            if (std::fabs(n1 - n2) > 1e-9) {
                std::printf("  FAIL: row %zu norm %.9f -> %.9f\n", m, n1, n2);
                ok = false;
            }
        }
        std::printf("  三行范数全部不变: %s\n", ok ? "OK" : "FAIL");
        if (!ok) std::printf("  FAIL\n");
    } catch (const std::logic_error &) {
        std::printf("  SKIPPED (not implemented)\n");
    }
}

/**
 * @brief 验证参数校验：奇数列 / 0 行矩阵必须 CHECK 失败
 */
void verify_guards() {
    std::printf("== 参数校验 ==\n");
    {
        try {
            const Matrix<double> X = {{1, 2, 3}};   // 3 列（奇数）
            (void) rope_rotate(X);
            std::printf("  FAIL: odd d_model should have thrown\n");
            return;
        } catch (const std::invalid_argument &) {
            std::printf("  odd d_model guard: OK\n");
        }
    }
    {
        try {
            const Matrix<double> X(0, 4);   // 0 行
            (void) rope_rotate(X);
            std::printf("  FAIL: 0-row should have thrown\n");
            return;
        } catch (const std::invalid_argument &) {
            std::printf("  0-row guard: OK\n");
        }
    }
}

/**
 * @brief 测试程序入口
 * @return 0（骨架期功能测试显示 SKIPPED 是预期行为，不视为失败）
 */
int main() {
    std::printf("mini-mlmath —— RoPE 旋转位置编码测试（骨架期）\n\n");
    verify_placeholder();
    verify_pos0_identity();
    verify_known_values();
    verify_relative_property();
    verify_norm_preserved();
    verify_guards();
    std::printf("\nDONE（骨架期：功能测试为 SKIPPED，实现 rope.h 后自动验收）\n");
    return 0;
}
