// statistics.h —— 基础描述性统计：mean / median / mode / variance / stddev
// 输入 std::vector<T>；variance / stddev 是样本方差（除以 n-1，n<2 抛异常）；
// 空输入抛异常；mode 并列时取最小值保证确定性。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "mini_mlmath/check.h"

// 算术平均数 mean = (1/n) Σ x_i
template <typename T>
T mean(const std::vector<T> &x) {
    CHECK(!x.empty()) << "statistics::mean: empty input";
    T sum = T(0);
    for (const T &v : x) sum += v;
    return sum / static_cast<T>(x.size());
}

// 中位数：排序后取中间（n 奇）或中间两个平均（n 偶）
template <typename T>
T median(const std::vector<T> &x) {
    CHECK(!x.empty()) << "statistics::median: empty input";
    std::vector<T> sorted = x;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t n = sorted.size();
    if (n % 2 == 1) return sorted[n / 2];
    return (sorted[n / 2 - 1] + sorted[n / 2]) / T(2);
}

// 众数：出现次数最多的元素，并列时取最小（确定性与 scipy.stats.mode 一致）
template <typename T>
T mode(const std::vector<T> &x) {
    CHECK(!x.empty()) << "statistics::mode: empty input";
    std::vector<T> sorted = x;
    std::sort(sorted.begin(), sorted.end());

    T best = sorted[0], current = sorted[0];
    std::size_t best_count = 1, current_count = 1;

    for (std::size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i] == current) {
            ++current_count;
        } else {
            if (current_count > best_count) {
                best = current;
                best_count = current_count;
            }
            current = sorted[i];
            current_count = 1;
        }
    }
    if (current_count > best_count) best = current;
    return best;
}

// 样本方差 = (1/(n-1)) Σ (x_i - mean)²，n<2 未定义故 CHECK 拦下
template <typename T>
T variance(const std::vector<T> &x) {
    CHECK(x.size() >= 2)
            << "statistics::variance: need >= 2 elements (sample variance uses n-1)";
    const T m = mean(x);
    T sum_sq = T(0);
    for (const T &v : x) {
        const T d = v - m;
        sum_sq += d * d;
    }
    return sum_sq / static_cast<T>(x.size() - 1);
}

// 样本标准差 = sqrt(variance)
template <typename T>
T stddev(const std::vector<T> &x) {
    return std::sqrt(variance(x));
}
