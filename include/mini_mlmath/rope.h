// ============================================================================
//  rope.h —— 旋转位置编码 RoPE（Rotary Position Embedding，header-only）
//
//  干的事（对应 LLaMA / Qwen / DeepSeek / Mistral 等现代 LLM 的位置编码
//  方案；论文 RoFormer: Enhanced Transformer with Rotary Position Embedding,
//  Su et al. 2021）：
//  和正弦位置编码（positional_encoding.h，Vaswani 2017）解决同一个问题
//  ——给注意力注入位置信息——但方式从「加」变成「乘」：
//
//      正弦版：输入 = embedding + PE(pos)              ← 加法，绝对位置
//      RoPE ：Q_m ← R(m·θ)·Q_m，K_n ← R(n·θ)·K_n      ← 乘法（旋转）
//
//  核心直觉（为什么叫「旋转」）：把 d 维向量看成 d/2 个二维平面——每对
//  相邻维度 (x_{2i}, x_{2i+1}) 是一个平面上的点（等价于一个复数）。RoPE
//  把第 i 个平面旋转 m·θ_i 角度。Q 旋转 m·θ、K 旋转 n·θ 后做点积，
//  旋转会相消（复数版一行看懂）：
//
//      e^{imθ} · e^{−inθ} = e^{i(m−n)θ}
//
//  即旋转矩阵的正交性 R(m)ᵀR(n) = R(n−m)。于是注意力分数**只依赖内容 +
//  相对距离 (m−n)**，「全局第几个位置」这种绝对信息从不出现在 QKᵀ 里。
//
//  和正弦版的血缘关系：频率 θ_i = 10000^(−2i/d) 与正弦版一字不差——
//  同一束「从快到慢的指针」（多尺度时钟），只是用法从「把指针读数抄在
//  词的脸上（加）」变成「把词本身按指针角度转过去（乘）」。
//
//  为什么现代 LLM 几乎都用它（vs 正弦绝对版）：
//    1. 相对位置：语言本质是「隔几个词」不是「全局第几位」；
//    2. 不碰 embedding / V：位置只在 QK 打分那一刻生效，词义不被污染；
//    3. KV cache 契合：K 在位置 n 旋转一次后缓存，**永远不用再动**——
//       之后任何新 query 来点积，相对距离自动从旋转相消里长出来；
//    4. 长度外推：调 θ 基频即可扩上下文（NTK-aware / YaRN / Llama3
//       的 rope scaling 全是围绕 θ 做文章）。
//
//  数学定义：
//    θ_i = 10000^(−2i/d_model)，i = 0..d/2−1
//    对位置 m 的向量 x，第 i 对维度 (2i, 2i+1) 的旋转：
//      x'_{2i}   = x_{2i}·cos(m·θ_i) − x_{2i+1}·sin(m·θ_i)
//      x'_{2i+1} = x_{2i}·sin(m·θ_i) + x_{2i+1}·cos(m·θ_i)
//    （就是二维旋转矩阵 [cos −sin; sin cos] 作用在列向量 (x_{2i}, x_{2i+1}) 上）
//
//  约定（与 positional_encoding.h 一致的无状态纯函数）：
//    - 无参数、无状态：旋转角由公式确定，不属于要学的权重；
//    - 输入 Matrix<T>（n×d，每行一个向量），**行号即位置**：第 m 行按
//      m·θ_i 旋转（transformer 里 Q 的第 m 行正是位置 m 的 query，天然对齐）；
//    - d 必须是偶数（维度成对旋转），CHECK 掉；
//    - Q 和 K 用同一个函数（RoPE 对 Q/K 对称，V 不旋转！）。
//
//  用法（transformer 里，对照正弦版）：
//    auto q2 = rope_rotate(Q);                  // Q 的第 m 行自动按位置 m 旋转
//    auto k2 = rope_rotate(K);                  // K 同样
//    scores = q2 * k2.transposed() / sqrt(d);  // 点积里只剩相对位置
//    // 注意：V 不旋转！RoPE 只动 Q/K
//
//  留给你实现的（骨架期方法体 throw，不给提示，公式见上方数学定义）：
//    - rope_rotate(...)
//    - 想扩展可加：显式传位置数组（稀疏/偏移场景）、θ 基频可配（外推
//      scaling）、只旋转部分维度（partial rotary，GPT-NeoX 的做法）
// ============================================================================
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "mini_mlmath/check.h"
#include "mini_mlmath/matrix.h"

/**
 * @brief RoPE 旋转：对矩阵每一行按其行号（位置）旋转
 * @param X n×d 的矩阵（Q 或 K），行号即位置（第 m 行 = 位置 m 的向量）
 * @return n×d 旋转后的矩阵：第 m 行的第 i 对维度 (2i, 2i+1) 旋转 m·θ_i
 * @note θ_i = 10000^(-2i/d)，与正弦位置编码同源的频率；只对 Q/K 用，V 不旋转
 */
template <typename T>
Matrix<T> rope_rotate(const Matrix<T> &X) {
    // 约束：T 必须是浮点类型。sin/cos 对整型无定义，和
    // positional_encoding.h 同一个套路：把错误钉在实例化点。
    static_assert(std::is_floating_point_v<T>,
                  "rope_rotate<T> requires floating-point T "
                  "(float / double / long double). sin/cos are undefined "
                  "for integer types.");

    // ---- 1) 参数校验 ----
    CHECK(X.rows() >= 1) << "rope_rotate: X must have at least 1 row";
    CHECK(X.cols() >= 2 && X.cols() % 2 == 0)
        << "rope_rotate: d_model (" << X.cols()
        << ") must be even and >= 2 — rotary pairs dimensions (2i, 2i+1)";

    // ---- 2) 核心旋转（留给你实现，不给提示）----
    //   TODO —— 公式见文件头注释或 docs/rope.md；写完删掉 throw。
    throw std::logic_error("rope_rotate: not implemented yet");
}
