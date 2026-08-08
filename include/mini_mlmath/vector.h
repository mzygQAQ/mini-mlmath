// ============================================================================
//  vector.h —— 教学用迷你向量库（Vector<T>）
//
//  和 matrix.h 配套：Matrix 是二维容器，Vector 是一维数学向量。
//  它干的是「两个向量之间求关系」这件事：
//     点积   a·b    ：Σ_i a_i·b_i，衡量两向量「同向程度」
//     模长   |v|    ：sqrt(Σ_i v_i²)，向量的长度（L2 范数）
//     余弦相似度    ：a·b / (|a|·|b|)，只看方向、不看长度
//
//  为什么需要它而不是直接用 std::vector？
//    std::vector 只有「存储」，没有「向量数学」。把这些操作收敛到一个
//    类里，调用方的语义就清楚了：v.dot(w) 就是数学书上的 a·b。
//    Eigen 里的 VectorXd 干的就是这件事 —— 你现在看到的就是它的雏形。
//
//  约定（与 matrix.h 一致）：
//    - 无宏、无表达式模板、无第三方库，纯手写
//    - 内部用 std::vector 连续存储，data() 暴露裸指针
//    - eager 求值：dot/norm/cosine 都是当场算完返回数值
// ============================================================================
#pragma once

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

template <typename T>
class Vector {
public:
    using value_type = T;
    using size_type = std::size_t;

    // ---- 构造 ----
    Vector() = default;                          // 空向量
    explicit Vector(size_type n) : data_(n, T{}) {}   // n 维零向量
    Vector(std::initializer_list<T> init) : data_(init) {}  // {1,2,3} 直接初始化

    // 静态工厂，语义更明确：Vector<double>::fromList({1,2,3})
    static Vector fromList(std::initializer_list<T> init) { return Vector(init); }

    // ---- 访问 ----
    size_type size() const noexcept { return data_.size(); }

    // 带越界检查的方括号下标（教学友好）。数学上 v(i) 等价 v[i]。
    T& operator[](size_type i) {
        if (i >= data_.size())
            throw std::out_of_range("Vector::operator[]: index out of range");
        return data_[i];
    }
    const T& operator[](size_type i) const {
        if (i >= data_.size())
            throw std::out_of_range("Vector::operator[]: index out of range");
        return data_[i];
    }

    T*       data() noexcept       { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    // ------------------------------------------------------------------------
    // 点积 dot：a·b = Σ_i a_i·b_i
    //
    // 几何意义：a·b = |a|·|b|·cosθ（θ 为夹角）
    //   - a、b 同向   → 正数，越大越「同向」
    //   - a、b 垂直   → 恰好 0
    //   - a、b 反向   → 负数
    // 两个向量长度必须相同，否则抛异常。
    // ------------------------------------------------------------------------
    T dot(const Vector& rhs) const {
        if (size() != rhs.size())
            throw std::invalid_argument("Vector::dot: dimension mismatch");
        T s = T(0);
        // 内层一个乘加循环，和 matrix.h 的乘法内层一样，
        // -O3 下会被向量化成一次处理多个分量。
        for (size_type i = 0; i < size(); ++i) s += data_[i] * rhs.data_[i];
        return s;
    }

    // ------------------------------------------------------------------------
    // 平方模长：Σ_i v_i²
    // 单独拎出来是因为很多场合（归一化、方差）只需要「平方和」，
    // 开方既慢又丢精度，能不开就不开。
    // ------------------------------------------------------------------------
    T squared_norm() const {
        T s = T(0);
        for (const T& v : data_) s += v * v;
        return s;
    }

    // ------------------------------------------------------------------------
    // 模长（L2 范数）：|v| = sqrt(Σ_i v_i²)
    // 就是「从原点到向量终点」的直线距离。零向量模长为 0。
    // ------------------------------------------------------------------------
    T norm() const { return std::sqrt(squared_norm()); }

    // ------------------------------------------------------------------------
    // 余弦相似度：cosθ = a·b / (|a|·|b|)，结果 ∈ [-1, 1]
    //
    // 和点积的关系：点积里「长度」和「夹角」混在一起，余弦相似度把长度
    // 除掉，**只看方向**。比如 {1,0} 和 {5,0} 长度差 5 倍，相似度依然是 1。
    // 这正适合文本/embedding/向量数据库检索：文档长度不一，但方向相近
    // 就认为是「相似」。
    //
    // 边界：任一向量模长为 0（零向量没有方向可言），抛异常比悄悄返回 0
    // 更能暴露 bug。
    // ------------------------------------------------------------------------
    T cosine_similarity(const Vector& rhs) const {
        const T na = norm();
        const T nb = rhs.norm();
        if (na == T(0) || nb == T(0))
            throw std::invalid_argument(
                "Vector::cosine_similarity: zero vector has no direction");
        return dot(rhs) / (na * nb);
    }

    // ---- 基础二元运算（eager，返回新向量）----
    Vector operator+(const Vector& rhs) const {
        if (size() != rhs.size())
            throw std::invalid_argument("Vector::operator+: dimension mismatch");
        Vector r(size());
        for (size_type i = 0; i < size(); ++i) r[i] = data_[i] + rhs.data_[i];
        return r;
    }

    Vector operator-(const Vector& rhs) const {
        if (size() != rhs.size())
            throw std::invalid_argument("Vector::operator-: dimension mismatch");
        Vector r(size());
        for (size_type i = 0; i < size(); ++i) r[i] = data_[i] - rhs.data_[i];
        return r;
    }

    // 标量乘法：v * s，把向量每个分量放大 s 倍
    Vector operator*(const T& s) const {
        Vector r(size());
        for (size_type i = 0; i < size(); ++i) r[i] = data_[i] * s;
        return r;
    }

private:
    std::vector<T> data_;   // 连续存储
};

// 标量在左边的写法：s * v
template <typename T>
Vector<T> operator*(const T& s, const Vector<T>& v) { return v * s; }

// ----------------------------------------------------------------------------
// 打印：[1, 2, 3]
// ----------------------------------------------------------------------------
template <typename T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& v) {
    os << '[';
    for (std::size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i + 1 < v.size()) os << ", ";
    }
    os << ']';
    return os;
}
