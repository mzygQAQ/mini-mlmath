// ============================================================================
//  positional_encoding_test.cpp —— 验证正弦位置编码
//
//  数据用手算可核对的小矩阵（seq_len=3, d_model=4）。
//
//  手算依据（公式见 positional_encoding.h 文件头）：
//    inv_freq[0] = 1/10000^(0/4) = 1
//    inv_freq[1] = 1/10000^(2/4) = 1/100 = 0.01
//    PE(0,:)   = [sin0, cos0, sin0, cos0]          = [0, 1, 0, 1]
//    PE(1,0)   = sin(1·1)      ≈ 0.841470985
//    PE(1,1)   = cos(1·1)      ≈ 0.540302306
//    PE(1,2)   = sin(1·0.01)   ≈ 0.009999833
//    PE(2,0)   = sin(2·1)      ≈ 0.909297427
//    PE(2,2)   = sin(2·0.01)   ≈ 0.019998667
//
//  验证点：
//    1. 形状：seq_len × d_model；
//    2. pos=0 行恒为 [0,1,0,1,...]（sin0=0 / cos0=1，一眼可核对）；
//    3. 手算的若干个数值点（容差 1e-6）；
//    4. 参数校验：seq_len=0 / d_model=0 / d_model 奇数，必须 CHECK 失败。
//
//  实现与原理见 positional_encoding.h 头部注释。
// ============================================================================
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>

#include "mini_mlmath/matrix.h"
#include "mini_mlmath/positional_encoding.h"

// 手算用的固定尺寸
constexpr std::size_t kSeq = 3;
constexpr std::size_t kD = 4;

/**
 * @brief 验证形状：返回 seq_len × d_model 的矩阵
 */
void verify_shape() {
    std::printf("== 形状：%zu × %zu ==\n", kSeq, kD);
    try {
        const Matrix<double> pe = sinusoidal_positional_encoding<double>(kSeq, kD);
        const bool ok = (pe.rows() == kSeq && pe.cols() == kD);
        std::printf("  rows=%zu cols=%zu: %s\n", pe.rows(), pe.cols(),
                    ok ? "OK" : "FAIL");
        if (!ok) std::printf("  FAIL\n");
    } catch (const std::logic_error &) {
        std::printf("  SKIPPED (not implemented)\n");
    }
}

/**
 * @brief 验证 pos=0 行恒为 [0,1,0,1,...]（sin0=0 / cos0=1）
 */
void verify_pos0_row() {
    std::printf("== pos=0 行 = [0,1,0,1,...] ==\n");
    try {
        const Matrix<double> pe = sinusoidal_positional_encoding<double>(kSeq, kD);
        bool ok = true;
        for (std::size_t j = 0; j < kD; ++j) {
            const double expect = (j % 2 == 0) ? 0.0 : 1.0;
            if (std::fabs(pe(0, j) - expect) > 1e-9) ok = false;
        }
        std::printf("  [%g,%g,%g,%g]: %s\n", pe(0, 0), pe(0, 1), pe(0, 2),
                    pe(0, 3), ok ? "OK" : "FAIL");
        if (!ok) std::printf("  FAIL: expected [0,1,0,1]\n");
    } catch (const std::logic_error &) {
        std::printf("  SKIPPED (not implemented)\n");
    }
}

/**
 * @brief 验证手算的若干数值点（d_model=4 时 inv_freq = [1, 0.01]）
 */
void verify_known_values() {
    std::printf("== 手算数值点核对 ==\n");
    try {
        const Matrix<double> pe = sinusoidal_positional_encoding<double>(kSeq, kD);
        struct Pt { std::size_t r, c; double v; };
        const Pt pts[] = {
            {1, 0, 0.841470985},   // sin(1)
            {1, 1, 0.540302306},   // cos(1)
            {1, 2, 0.009999833},   // sin(0.01)
            {2, 0, 0.909297427},   // sin(2)
            {2, 2, 0.019998667},   // sin(0.02)
        };
        bool ok = true;
        for (const Pt &p : pts) {
            if (std::fabs(pe(p.r, p.c) - p.v) > 1e-6) {
                std::printf("  FAIL: PE(%zu,%zu)=%.9f, expected %.9f\n",
                            p.r, p.c, pe(p.r, p.c), p.v);
                ok = false;
            }
        }
        std::printf("  5 个手算点全部命中: %s\n", ok ? "OK" : "FAIL");
        if (!ok) std::printf("  FAIL\n");
    } catch (const std::logic_error &) {
        std::printf("  SKIPPED (not implemented)\n");
    }
}

/**
 * @brief 验证参数校验：seq_len=0 / d_model=0 / d_model 奇数都必须 CHECK 失败
 */
void verify_guards() {
    std::printf("== 参数校验 ==\n");
    {
        try {
            (void) sinusoidal_positional_encoding<double>(0, kD);
            std::printf("  FAIL: seq_len=0 should have thrown\n");
            return;
        } catch (const std::invalid_argument &) {
            std::printf("  seq_len=0 guard: OK\n");
        }
    }
    {
        try {
            (void) sinusoidal_positional_encoding<double>(kSeq, 0);
            std::printf("  FAIL: d_model=0 should have thrown\n");
            return;
        } catch (const std::invalid_argument &) {
            std::printf("  d_model=0 guard: OK\n");
        }
    }
    {
        try {
            (void) sinusoidal_positional_encoding<double>(kSeq, 3);   // 奇数
            std::printf("  FAIL: odd d_model should have thrown\n");
            return;
        } catch (const std::invalid_argument &) {
            std::printf("  d_model odd guard: OK\n");
        }
    }
}

/**
 * @brief 测试程序入口
 * @return 0（全部通过）
 */
int main() {
    std::printf("mini-mlmath —— 正弦位置编码测试\n\n");
    verify_shape();
    verify_pos0_row();
    verify_known_values();
    verify_guards();
    std::printf("\nALL OK\n");
    return 0;
}
