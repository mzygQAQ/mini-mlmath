// ============================================================================
//  Matrix.h —— mini-mlmath 库本体（header-only）
//
//  这是写给「懂编程、但好奇矩阵库内部到底怎么实现」的人看的教学代码。
//  它的目标不是性能冠军，而是**可读性**：让你看懂 Eigen / OpenBLAS 这类
//  工业级库在背后做了什么，以及它们为什么长得那么复杂。
//
//  和 Eigen 的三大刻意不同（也是本库的定位）：
//
//  ① 无宏。除了 #pragma once 一个预处理指令外，全库不用任何 #define。
//     Eigen 内部充满了宏（EIGEN_MAY_ALIAS、EIGEN_DEVICE_FUNC、行内代码
//     生成等），是它难读的第一道坎；本库全部用 constexpr / inline / 模板
//     来表达同样的东西，让你看到「不用宏也能写矩阵库」。
//
//  ② 无表达式模板。Eigen 里 A * B 返回的不是矩阵，而是一个懒求值的
//     Product<...> 表达式对象，真正的计算要等到赋值给 Matrix 时才发生，
//     好处是可以「融合」多个操作避免中间结果。本库的 A * B 立即算完、
//     立即返回一个真正的 Matrix —— 这叫 eager 求值，一行一个结果，
//     概念上简单得多。
//
//  ③ 无 BLAS 对接。工业级矩阵库底层几乎都会在运行时切换到 BLAS
//     （OpenBLAS / MKL）做 GEMM，或者自己内嵌一套手工汇编的微内核
//     （Eigen 的 GeneralBlockPanelKernel.h 就是）。本库三种乘法全部是
//     自己写的最朴素的 C++ 循环，你在任何一行都能看到「计算本身」。
//
//  存储布局：行主序（row-major）+ std::vector 连续内存。
//  这是所有性能讨论的地基，请先记住索引公式：
//
//      data_[i * cols_ + j]  ==  m(i, j)      （第 i 行第 j 列）
//
//  行主序 = 同一行的元素在内存里紧挨着。这决定了后面 i/k/j 循环顺序
//  和「打包」两个优化为什么成立，请务必带着这个公式看下面的乘法注释。
// ============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

// CHECK 宏是本库唯一的宏例外（见 check.h 头注释）——它要拿到条件原文和
// 调用点文件/行号，本质是编译期文本处理，只能靠宏。文件里 multiply_*
// 用的是裸 throw，两种写法等价；operator* 这里选 CHECK 是因为错误消息
// 会把「条件原文 + 实际值」都打出来，对使用者更友好。
#include "mini_mlmath/check.h"

// 分块大小：所有分块算法共用的块边长。
// 为什么是 64？
//   - 一个 64×64 的 double 块正好是 64*64*8 = 32 KB，正好等于现代 x86
//     CPU（Skylake+ / Zen2+）的 L1 数据缓存大小。整块放得下 L1，内层循环
//     里反复读它都能命中缓存。
//   - 64 也是 OpenBLAS 默认 GEMM 面板尺寸的量级（GEMM_DEFAULT_P 等），
//     真实世界选块大小就是在这种「够大覆盖一次访问、又小到装进缓存」的
//     权衡里调试出来的。
constexpr std::size_t kBlock = 64;

// ============================================================================
//  Matrix<T>：动态大小、行主序矩阵
//
//  为什么用 std::vector 而不是裸指针 + new？
//  -  RAII：析构自动释放，拷贝自动深拷贝，不怕泄漏、不怕双重释放。
//     这是「C++ 新手写矩阵最容易翻车」的地方，工业库其实也一样，
//     Eigen 内部就是用带内存对齐的 allocator 管理一块 buffer。
//  -  data() 直接暴露连续内存，后面乘法里可以直接拿裸指针索引，
//     跟 C 数组一样快（vector::data() 在 -O2 下就是零开销）。
// ============================================================================
template <typename T>
class Matrix {
public:
    using value_type = T;
    using size_type = std::size_t;

    // ---- 构造 ----

    // 默认构造：0×0 空矩阵
    Matrix() = default;

    // 直接指定行列：全 0。注意 data_ 是 rows*cols 个 T{}（对 double 即 0.0）
    Matrix(size_type rows, size_type cols)
        : rows_(rows), cols_(cols), data_(rows * cols, T{}) {}

    // 从嵌套花括号初始化。这是给用户「看得懂」的初始化方式，
    // 等价于 Eigen 的 MatrixXd 初始化写法（Eigen 靠 << 操作符 + comma
    // initializer 语法糖实现，这里用 C++11 自带的 initializer_list）。
    Matrix(std::initializer_list<std::initializer_list<T>> init) {
        rows_ = init.size();
        if (rows_ == 0) return;  // 空列表 -> 0×0，合法
        cols_ = init.begin()->size();
        data_.reserve(rows_ * cols_);
        for (const auto& row : init) {
            if (row.size() != cols_) {
                throw std::invalid_argument(
                    "Matrix: each row must have the same number of elements (no jagged matrices)");
            }
            for (const T& v : row) data_.push_back(v);
        }
    }

    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) noexcept = default;

    static Matrix fromList(std::initializer_list<std::initializer_list<T>> init) {
        return Matrix(init);
    }

    // ---- 形状查询 ----
    size_type rows() const noexcept { return rows_; }
    size_type cols() const noexcept { return cols_; }

    // ---- 元素访问 ----

    // operator()(i, j)：带越界检查的版本，教学期友好。
    // 为什么用圆括号 (i,j) 而不是方括号 [i][j]？
    //   - [i][j] 需要两个下标运算符、无法在第二层拿到行长度做检查；
    //   - 圆括号可以拿到 (行, 列) 两个参数，直接越界检查。
    //   Eigen 也是 (i,j) 语义。注意教学版检查会拖慢内层循环，
    //   所以后面乘法里我们一律走 data() 裸指针（见 multiply_* 注释）。
    T& operator()(size_type i, size_type j) {
        if (i >= rows_ || j >= cols_)
            throw std::out_of_range("Matrix::operator(): index out of range");
        return data_[i * cols_ + j];
    }
    const T& operator()(size_type i, size_type j) const {
        if (i >= rows_ || j >= cols_)
            throw std::out_of_range("Matrix::operator(): index out of range");
        return data_[i * cols_ + j];
    }

    // 直接暴露连续缓冲区。乘法内层循环用它，绕开 operator() 的越界检查，
    // 也顺便让你看到「真实库的 hot path 里其实全是指针运算」。
    T*       data() noexcept       { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    // ---- 原地自操作（返回引用便于链式：m += n *= 2 这类）----

    Matrix& operator+=(const Matrix& rhs) {
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_)
            throw std::invalid_argument("operator+=: shape mismatch");
        // 直接按连续内存逐元素加，跟行列无关，最快。
        for (size_type i = 0; i < data_.size(); ++i) data_[i] += rhs.data_[i];
        return *this;
    }

    Matrix& operator-=(const Matrix& rhs) {
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_)
            throw std::invalid_argument("operator-=: shape mismatch");
        for (size_type i = 0; i < data_.size(); ++i) data_[i] -= rhs.data_[i];
        return *this;
    }

    Matrix& operator*=(const T& scalar) {
        for (T& v : data_) v *= scalar;
        return *this;
    }

    // ---- 转置（eager：立刻返回一个全新的矩阵）----
    // 这里能看出和 Eigen 的第二个不同：Eigen 的 .transpose() 返回一个
    // 「反向索引的懒表达式」，不拷贝数据，真正的搬运发生在它被使用/赋值时，
    // 避免了一次中间拷贝。我们为了讲清楚概念，直接现场搬好返回。
    // 代价是：做 A*B 时若先转置会多一次 O(M*N) 拷贝 —— 真实库用懒表达式
    // 就是为了省掉它。
    Matrix transposed() const {
        Matrix r(cols_, rows_);
        for (size_type i = 0; i < rows_; ++i)
            for (size_type j = 0; j < cols_; ++j)
                r(j, i) = (*this)(i, j);
        return r;
    }

    // ---- 追加全 1 列（bias folding 的增广：x -> [x, 1]）----
    // 机器学习里给设计矩阵右边并一列恒 1，配合「权重最后一个分量 = bias」，
    // 就能把 y = w·x + b 写成纯矩阵乘 y = [w,b]·[x,1]（感知机的 bias
    // folding，见 ml/perceptron.h 注释）。对应 numpy 的
    //   np.hstack([X, np.ones((n, 1))])
    // 返回新矩阵（n × (d+1)），不修改原矩阵（eager，同 transposed()）。
    // 0 行的矩阵没有行可增广，拒绝。
    Matrix with_ones_column() const {
        CHECK(rows() > 0) << "cannot append a ones column to an empty matrix";
        Matrix r(rows_, cols_ + 1);
        for (size_type i = 0; i < rows_; ++i) {
            for (size_type j = 0; j < cols_; ++j) r(i, j) = (*this)(i, j);
            r(i, cols_) = T(1);   // 最后一列恒 1，喂给 bias
        }
        return r;
    }

private:
    size_type rows_ = 0;
    size_type cols_ = 0;
    std::vector<T> data_;   // 连续内存，长度 rows_*cols_，行主序
};

// ============================================================================
//  二元 operator+ / operator-（a + b 不修改操作数的版本）
//  这里只用「就地 += / -=」拼出来，避免重复写循环 —— 顺带演示组合。
//  注意返回值语义：按值返回一个全新矩阵（eager）。写成
//  auto C = A + B;  实际发生了：构造临时、operator+=、移动返回，两步。
// ============================================================================
template <typename T>
Matrix<T> operator+(const Matrix<T>& a, const Matrix<T>& b) {
    Matrix<T> r(a);      // 拷贝 a
    r += b;              // 就地加 b
    return r;            // 移动返回（C++17 起有 guaranteed copy elision）
}

template <typename T>
Matrix<T> operator-(const Matrix<T>& a, const Matrix<T>& b) {
    Matrix<T> r(a);
    r -= b;
    return r;
}

// 标量乘法：scalar * m 和 m * scalar 都给一个，方便两边写
template <typename T>
Matrix<T> operator*(const Matrix<T>& m, const T& scalar) {
    Matrix<T> r(m);
    r *= scalar;
    return r;
}
template <typename T>
Matrix<T> operator*(const T& scalar, const Matrix<T>& m) { return m * scalar; }

// ============================================================================
//  operator<<：打印。行主序，每行一个方括号。
// ============================================================================
template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& m) {
    for (std::size_t i = 0; i < m.rows(); ++i) {
        os << '[';
        for (std::size_t j = 0; j < m.cols(); ++j) {
            os << m(i, j);
            if (j + 1 < m.cols()) os << ", ";
        }
        os << "]\n";
    }
    return os;
}

// ============================================================================
//  三种矩阵乘法：naive / blocked / packed
//  同一个数学运算（C = A*B，A 是 M×K，B 是 K×N，C 是 M×N），
//  三种实现，速度从慢到快，注释按这个顺序层层递进地解释为什么。
//  三者结果在浮点精度内完全一致（只是累加顺序不同），main.cpp 会用
//  approxEqual 验证。
// ============================================================================

// ----------------------------------------------------------------------------
// 版本 1：multiply_naive —— 最朴素的 ikj 三层循环
// ----------------------------------------------------------------------------
// 数学定义：C(i,j) = Σ_k A(i,k) * B(k,j)。按定义直接写有三个候选循环顺序：
//   ijk、ikj、kij... 为什么我们选 ikj？这是行主序下缓存命中的关键，请对照
//   索引公式 data[i*cols_+j] 逐条看：
//
//   - 若内层是 k（ijk 顺序）：此时 j 固定，访问
//       A[i*K + k]  连续 ✓
//       B[k*N + j]  k 每加 1 就跳一整行（stride = N 个 double）✗ 狂刷缓存行
//     B 的列访问对行主序是灾难 —— 每次跨 2048 个 double 取一个值，
//     一条 64B 缓存行只用到 8B，浪费 87.5%。
//
//   - 若内层是 j（ikj 顺序，本版本）：此时 k 固定，访问
//       A[i*K + k]  一个标量，取一次进寄存器复用 ✓
//       B[k*N + j]  j 连续递增 → B 的这一行顺序读 ✓
//       C[i*N + j]  j 连续递增 → C 的这一行顺序写 ✓
//     两个连续、一个标量，完美贴合行主序。这就是缓存友好的内层。
//
//   更直观的说法：ikj 内层做的事是
//       C 的第 i 行 += aik * B 的第 k 行        （把 B 的一行乘个系数累加到 C 一行）
//   即一次 row-axpy（行级 乘-加），内层循环里 B 和 C 都是整行顺序流，
//   硬件预取器能轻松预判，编译器也能安全地向量化成
//       C_row[j..j+3] += aik * B_row[j..j+3]   （AVX2 一次算 4 个 double）
//
//   实测基准（本机 g++ 13.3 -O3 -march=native，double）：N=2048 时 2845 ms；
//   N=512 时 21.9 ms。它是下面两版的对比基准线。
// ============================================================================
template <typename T>
Matrix<T> multiply_naive(const Matrix<T>& A, const Matrix<T>& B) {
    const std::size_t M = A.rows();   // A: M×K
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();   // B: K×N -> C: M×N
    if (B.rows() != K)
        throw std::invalid_argument("multiply_naive: A.cols() != B.rows()");

    Matrix<T> C(M, N);
    const T* a = A.data();
    const T* b = B.data();
    T* c = C.data();

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const T aik = a[i * K + k];   // 标量，内层复用（编译器放寄存器）
            // 内层 j：C 行和 B 行都顺序访问 —— 这是缓存友好的关键，见上面注释
            for (std::size_t j = 0; j < N; ++j) {
                c[i * N + j] += aik * b[k * N + j];
            }
        }
    }
    return C;
}

// ----------------------------------------------------------------------------
// 版本 2：multiply_blocked —— 分块（cache blocking）
// ----------------------------------------------------------------------------
// naive 的问题：对每个 (i, k)，内层要扫一遍 B 的第 k 行。B 的第 k 行会被
// 读 M 次（每个 i 读一次）。N=2048 时一行就有 16KB，B 整个 32MB —— 远大于
// 各级缓存。结果：B 的数据反复从主存读 M 次，吞吐被内存带宽掐死。
//
// 分块思想：把三个矩阵都切成 kBlock×kBlock 的小块，让「正在算的这块 C、
// 它需要的 A 列块、它需要的 B 行块」三块能同时待在缓存里，内层循环反复
// 读的都是刚用过的数据（时间局部性）。
//
// 执行顺序（本版本）：
//   外层按 C 的块 (i0, j0) 遍历
//     中层按 K 方向按块推进 (k0)
//       内层是「块内的小 ikj」：i∈[i0,i0+bs), k∈[k0,k0+bs), j∈[j0,j0+bs)
//   这样 A 的某个列块 [i0.., k0..] 被 j 方向所有块共享，B 的行块被 i 方向共享。
//
// 坏消息（这是真实性能教训 2 的前奏）：分块之后「块」在内存里并不连续！
//   A 的列块 [i0:i0+bs, k0:k0+bs]：行主序下按行存储，块内两行之间隔了
//     整整一行（stride = K 个 double），不是一个连续区间；
//   B 的行块 [k0:k0+bs, j0:j0+bs]：行内连续，但 bs 行之间同样有 stride = N。
//   所以本版本内层 j 循环虽然连续，但读 A 列块时每行都要跨步长跳一次，
//   缓存行用不满、TLB 容易爆 —— 这就是「分块不打包反而更慢」的根源。
//
//   实测（本机 g++ 13.3 -O3 -march=native）：
//     N=512： 22.4 ms，反而不如 naive 的 21.9 ms —— 块太小、全在缓存里，
//             分块没数据可救，只剩块循环 + 余数判断的开销，纯亏（教训 3）。
//     N=2048：1887 ms，比 naive(2845 ms) 快约 1.5 倍 —— 缓存复用开始生效，
//             但没打包，依然比 packed(941 ms) 慢约 2 倍（教训 2）。
// ============================================================================
template <typename T>
Matrix<T> multiply_blocked(const Matrix<T>& A, const Matrix<T>& B,
                           std::size_t bs = kBlock) {
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();
    if (B.rows() != K)
        throw std::invalid_argument("multiply_blocked: A.cols() != B.rows()");

    Matrix<T> C(M, N);
    const T* a = A.data();
    const T* b = B.data();
    T* c = C.data();

    // 外层：C 的块按 (i0, j0) 遍历。iSize/jSize 处理边长不是 64 整数倍的
    // 余数（比如 N=1000，最后一块只有 1000-15*64=40 列）。
    for (std::size_t i0 = 0; i0 < M; i0 += bs) {
        const std::size_t iSize = std::min(bs, M - i0);
        for (std::size_t j0 = 0; j0 < N; j0 += bs) {
            const std::size_t jSize = std::min(bs, N - j0);
            // K 方向推进：把 A 的这个列块和 B 的这个行块喂给内层
            for (std::size_t k0 = 0; k0 < K; k0 += bs) {
                const std::size_t kSize = std::min(bs, K - k0);
                // 块内小 ikj：和 naive 的差别只是把 M/K/N 换成块的尺寸，
                // 以及下标都加上了块原点 (i0, k0, j0)
                for (std::size_t i = 0; i < iSize; ++i) {
                    for (std::size_t k = 0; k < kSize; ++k) {
                        // 注意这里取 A 用的是「列块」，两行之间跨 stride=K ——
                        // 就是不连续的具体表现，见函数头注释
                        const T aik = a[(i0 + i) * K + (k0 + k)];
                        for (std::size_t j = 0; j < jSize; ++j) {
                            c[(i0 + i) * N + (j0 + j)] +=
                                aik * b[(k0 + k) * N + (j0 + j)];
                        }
                    }
                }
            }
        }
    }
    return C;
}

// ----------------------------------------------------------------------------
// 版本 3：multiply_packed —— 分块 + 数据打包（真实 BLAS/GEMM 的做法）
// ----------------------------------------------------------------------------
// 这是「真实世界」的做法，对应 OpenBLAS 的 SGEMM kernel、Eigen 的
// GeneralBlockPanelKernel.h：在分块之上再叠加一步「打包」（packing）。
//
// 打包要解决的正是版本 2 末尾指出的问题：
//   A 的列块 / B 的行块在行主序矩阵里「不是一块连续内存」。
// 解决方式很朴素：算这个块之前，先把它复制到一块连续的小 buffer 里
// （叫 panel），内层微内核只在连续 buffer 上跑，读的全是顺序地址。
//
// 为什么连续内存这么重要？
//   1) 缓存行利用率：64B 缓存行装 8 个 double，连续读每个都用满；
//      跨 stride 读平均每行只用到头几个，其余全部浪费。
//   2) 硬件预取：连续地址预取器（prefetcher）能稳稳预判，跨 stride 经常
//      预判失败。
//   3) TLB：连续大块只需少数页表项，跨 stride 的散布访问会快速消耗 TLB。
//   4) 编译器向量化更干净：连续读写编译器可以放心发宽向量 load/store，
//      无需担心别名/对齐。
//
// 打包的代价：多一次 O(块面积) 的拷贝。但这份拷贝是一次性成本，换来整个
// 块被反复复用的若干轮循环都是连续访问 —— 收益远大于代价。这就是为什么
// 真实 GEMM 全部先 pack 再算。
//
// 本版本结构（和真实 BLAS 的 panel-based GEMM 同构）：
//   for k0 in K:
//       pack B 的横向 panel: B[k0:k0+bs, 0:N]  → bPack（连续）
//       for i0 in M:
//           pack A 的小块:    A[i0:i0+bs, k0:k0+bs] → aPack（连续）
//           for j0 in N:
//               microkernel:  C[i0.., j0..] += aPack * bPack 的 [j0..] 列段
//   bPack 每个 k0 只 pack 一次，被所有 (i0) 复用 —— 这就是「面板」的意义；
//   aPack 每个 (k0,i0) 打包一次，被所有 j0 复用。
//
//   实测（本机 g++ 13.3 -O3 -march=native，double）：
//     N=512：  15.3 ms，只比 naive(21.9 ms) 快约 1.4 倍 —— 块太小，拷贝
//              + 分块的开销吃掉大半收益；
//     N=1024：132.4 ms，naive 202.1 ms —— 约 1.5 倍；
//     N=2048：941 ms，naive 2845 ms —— 约 3.0 倍；比 blocked(1887 ms) 快
//              约 2.0 倍。大矩阵下「连续访问」与「面板复用」的优势全兑现。
//   再对照 -O2 的同一份代码：N=2048 要 4128 ms（教训 1，见 CMakeLists.txt）。
// ============================================================================
template <typename T>
Matrix<T> multiply_packed(const Matrix<T>& A, const Matrix<T>& B,
                          std::size_t bs = kBlock) {
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();
    if (B.rows() != K)
        throw std::invalid_argument("multiply_packed: A.cols() != B.rows()");

    Matrix<T> C(M, N);
    const T* a = A.data();
    const T* b = B.data();
    T* c = C.data();

    // 打包用的连续 workspace（真实 BLAS 在进程启动时申请固定大小的
    // workspace 复用；这里直接重用两个 vector，避免在热循环里反复 malloc）。
    std::vector<T> aPack(bs * bs);   // 一块 A 的小块，行主序：aPack[i*bs+k]
    std::vector<T> bPack(bs * N);    // 一条 B 的横向 panel，行主序：bPack[k*N+j]

    for (std::size_t k0 = 0; k0 < K; k0 += bs) {
        const std::size_t kSize = std::min(bs, K - k0);

        // ---- 打包 B 的横向 panel：B[k0:k0+kSize, 0:N] ----
        // 每行就是 B 原始的一整行（本来就是连续的），整行 memcpy 级拷贝。
        for (std::size_t k = 0; k < kSize; ++k)
            std::copy(b + (k0 + k) * N, b + (k0 + k) * N + N, bPack.data() + k * N);

        for (std::size_t i0 = 0; i0 < M; i0 += bs) {
            const std::size_t iSize = std::min(bs, M - i0);

            // ---- 打包 A 的小块：A[i0:i0+iSize, k0:k0+kSize] ----
            // 这是真正的「搬运」：把 stride=K 的列块，压成 stride=kSize 的连续块
            for (std::size_t i = 0; i < iSize; ++i)
                std::copy(a + (i0 + i) * K + k0,
                          a + (i0 + i) * K + k0 + kSize,
                          aPack.data() + i * kSize);

            for (std::size_t j0 = 0; j0 < N; j0 += bs) {
                const std::size_t jSize = std::min(bs, N - j0);

                // ---- 微内核（microkernel）：块内 ikj，但只读打包好的连续内存 ----
                // 这是 GEMM 真正的 hot loop，真实库里这一小段会被
                // 手写汇编 / intrinsics 抠到极致（寄存器平铺、FMA 等）。
                // 我们保留朴素的 C++，你仍然能看出它和 naive 的唯一区别：
                // 读的是 aPack / bPack 两个连续 buffer，而 naive 读的是
                // 有 stride 的原始矩阵。
                for (std::size_t i = 0; i < iSize; ++i) {
                    for (std::size_t k = 0; k < kSize; ++k) {
                        const T aik = aPack[i * kSize + k];   // 连续 ✓
                        T* cRow  = c + (i0 + i) * N + j0;     // 连续 ✓
                        const T* bRow = bPack.data() + k * N + j0; // 连续 ✓
                        for (std::size_t j = 0; j < jSize; ++j) {
                            cRow[j] += aik * bRow[j];         // 全部顺序访问
                        }
                    }
                }
            }
        }
    }
    return C;
}

// ============================================================================
//  operator* —— 矩阵 × 矩阵（对应 numpy 的 A @ B）
//
//  为什么有了三种 multiply_* 还要再补一个 operator*？
//    三种乘法用命名函数是为了教学：让你显式挑选算法、对比性能。
//    但日常写代码时「A * B 就是矩阵乘」才是直觉 —— 数学记号 A·B、
//    numpy 的 A @ B 都是这个意思。所以补一个便捷入口，
//    默认委托最快的 multiply_packed。
//
//  和「标量乘法 operator*」不冲突：
//    重载决议按参数签名区分 —— (Matrix, Matrix) 走这里，
//    (Matrix, scalar) 走上面那个，编译器自动挑对。
//
//  语义对照（防混淆，重要）：
//    numpy  A @ B   ==  本库 A * B      （真正的矩阵乘法）
//    numpy  A * B   ==  逐元素乘        （本库没有这个运算符！）
//  也就是说：本库的 * 对应 numpy 的 @，千万别和 numpy 的 * 混为一谈。
//
//  教学代价（一句话说清楚）：
//    有了 A * B，调用处就看不到「底层用哪个算法」了。想对比三种实现的
//    性能差异，仍然请显式调用 multiply_naive / multiply_blocked /
//    multiply_packed。
// ============================================================================
template <typename T>
Matrix<T> operator*(const Matrix<T>& a, const Matrix<T>& b) {
    CHECK(a.cols() == b.rows()) << "A.cols() must equal B.rows(), got "
                                << a.cols() << " vs " << b.rows();
    return multiply_packed(a, b);   // 默认走最快的版本
}

// ============================================================================
//  approxEqual：浮点容差比较
//  三种乘法累加顺序不同（naive 按 k 全量累加、packed 按块累加），double
//  的加法不满足结合律，结果在最后几位会有差异 —— 这是**正常的浮点行为**，
//  不是 bug。所以「正确性」用相对容差判断，而不是 bit 级 ==。
// ============================================================================
template <typename T>
bool approxEqual(const Matrix<T>& a, const Matrix<T>& b, T relTol = T(1e-9)) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            const T diff = std::abs(a(i, j) - b(i, j));
            // 相对误差 = diff / max(1, |a|, |b|)，避免除以 0、避免 0.0 附近误判
            const T scale =
                std::max(T(1), std::max(std::abs(a(i, j)), std::abs(b(i, j))));
            if (diff > relTol * scale) return false;
        }
    }
    return true;
}
