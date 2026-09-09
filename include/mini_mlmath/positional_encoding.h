// ============================================================================
//  positional_encoding.h —— 正弦位置编码 Sinusoidal Positional Encoding
//  （header-only，transformer 的组件之一）
//
//  干的事：生成一张 seq_len × d_model 的「位置表」，每一行对应序列里的一个
//  位置，用 sin / cos 的周期震荡把「位置信息」编码进数值。transformer 里把
//  它加到 token embedding 上，让模型知道「第几个词」。
//
//  为什么需要它？（这是理解位置编码存在意义的关键）
//  注意力本身是一个**与顺序无关**的运算：把输入序列任意打乱，Q·Kᵀ 里
//  每一对 (i,j) 的分数不变——因为 Q/K/V 各自按行独立算，没有任何东西
//  告诉模型「第 i 行和第 j 行谁前谁后」。对语言来说顺序就是一切
//  （「猫追狗」≠「狗追猫」），所以必须**显式地**把位置塞进输入。
//
//  数学定义（Vaswani 2017《Attention Is All You Need》原始版）：
//      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))    偶数维
//      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))    奇数维
//    pos = 位置（0..seq_len-1），i = 维度下标（0..d_model/2-1）。
//  也就是说：把 d_model 维劈成 d_model/2 对 (sin, cos)，每对共享同一个
//  **角频率** 1/10000^(2i/d) —— 维度越低震荡越快、维度越高震荡越慢。
//
//  为什么用 sin / cos？
//    1) 数值有界：任何位置、任何维度都落在 [-1, 1]，不会因位置大而爆炸；
//    2) 相对位置可「算出来」：PE(pos+k) 能写成 PE(pos) 的线性组合
//       （三角恒等式），模型可以学着捕捉「距离 k」这种相对位置信息，
//       而不是只记死每个绝对位置；
//    3) 高频维区分相邻位置、低频维区分长距离，多尺度分工。
//
//  约定（和 softmax.h / activation.h 一致的无状态纯函数）：
//    - 无参数、无状态：位置编码是**常数矩阵**，不属于要学的权重
//      （PyTorch 里是 register_buffer，不是 nn.Parameter）；
//    - 返回 Matrix<T>：seq_len × d_model，每行一个位置的编码向量；
//    - d_model 必须是**偶数**：sin/cos 成对出现（见上方公式），
//      奇数会让最后一维没配对，本实现直接 CHECK 掉。
//
//  用法（transformer 里的位置）：
//    auto pe = sinusoidal_positional_encoding<float>(seq_len, d_model);
//    embedded(i, j) += pe(i, j);   // 加到 token embedding 上，再进注意力
//
//  复杂度：O(seq_len · d_model)，纯填充矩阵，无状态可缓存。
//
//  配套讲解：docs/softmax.md / docs/autograd.md 是它的邻居组件；
//  位置编码单独的原理文档（docs/positional_encoding.md）待写。
//
//  实现（骨架期已填好）：预计算 inv_freq[i] = 1 / 10000^(2i/d)，再逐行填
//  sin/cos（偶数维 sin、奇数维 cos，每两列共享同一频率）。
//    - 想扩展可加：可学习位置编码（学一个参数矩阵）、相对位置编码
//      （Transformer-XL 的 rotary / RoPE 是另一个故事）。
// ============================================================================
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "mini_mlmath/check.h"
#include "mini_mlmath/matrix.h"

// ----------------------------------------------------------------------------
//  sinusoidal_positional_encoding(seq_len, d_model)：生成正弦位置编码矩阵
// ----------------------------------------------------------------------------
/**
 * @brief 生成正弦位置编码矩阵（无参数、无状态）
 * @param seq_len 序列长度（位置个数，>= 1）
 * @param d_model 特征维数（必须是偶数，sin/cos 成对）
 * @return seq_len × d_model 的 Matrix<T>：第 pos 行 = 位置 pos 的编码向量
 * @note 返回值是常数矩阵，transformer 里加到 token embedding 上再进注意力
 */
template <typename T>
Matrix<T> sinusoidal_positional_encoding(std::size_t seq_len,
                                         std::size_t d_model) {
    // 约束：T 必须是浮点类型（float / double / long double）。
    // std::sin / std::cos / std::pow 对整型 T 没有定义，T=int 会在 <cmath>
    // 内部报晦涩错误 —— 用 static_assert 把错误钉在本函数的实例化点，
    // 并给出人话。（和 Perceptron / KMeans 同一个套路）
    static_assert(std::is_floating_point_v<T>,
                  "sinusoidal_positional_encoding<T> requires floating-point "
                  "T (float / double / long double). sin/cos/pow are undefined "
                  "for integer types.");

    // ---- 1) 参数校验 ----
    CHECK(seq_len >= 1)
        << "sinusoidal_positional_encoding: seq_len (" << seq_len
        << ") must be >= 1";
    CHECK(d_model >= 1)
        << "sinusoidal_positional_encoding: d_model (" << d_model
        << ") must be >= 1";
    CHECK(d_model % 2 == 0)
        << "sinusoidal_positional_encoding: d_model (" << d_model
        << ") must be even — sin/cos come in pairs (see file header)";

    //   1) 预计算角频率 inv_freq[i] = 1 / 10000^(2i/d_model)，i = 0..d/2-1
    //      （std::pow 只在初始化算一次，别放进内层循环）；
    std::vector<T> inv_freq(d_model / 2);
    for (std::size_t i = 0; i < d_model / 2; ++i) {
        const T y = T(2 * i) / T(d_model);   // 浮点除法：2i/d_model 不能被整数截断
        inv_freq[i] = T(1) / std::pow(T(10000), y);
    }

    //   2) 建 Matrix<T> pe(seq_len, d_model)，双循环：
    //        for pos: for j in 0..d_model-1:
    //            angle = pos * inv_freq[j/2]
    //            pe(pos, j) = (j 为偶数) ? sin(angle) : cos(angle)
    Matrix<T> pe(seq_len, d_model);
    for (std::size_t pos = 0; pos < seq_len; ++pos) {
        for (std::size_t j = 0; j < d_model; ++j) {
            const T angle = T(pos) * inv_freq[j / 2];
            if (j % 2 == 0) {
                pe(pos, j) = std::sin(angle);
            } else {
                pe(pos, j) = std::cos(angle);
            }
        }
    }

    return pe;
}
