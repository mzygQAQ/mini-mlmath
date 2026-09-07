// ============================================================================
//  autograd.h —— 迷你自动求导（reverse-mode AD，header-only）
//
//  这是 mini-mlmath 向「真正深度学习框架」迈进的一步：Matrix<T> 只有
//  「值」，本文给值加上「怎么算出来的」这一层 —— 自动微分。
//
//  和 PyTorch 的对应关系（教学版做了哪些裁剪，先说清楚）：
//    - 张量句柄：Tensor<T> 是「指向共享内核的轻量句柄」，拷贝共享同一份
//      数据和 autograd 状态（类似 PyTorch 里 b = a 后 b 与 a 的关系），
//      这是后面「图引用」能成立的地基。
//    - 反向图：每个由算子产生的张量挂一个 grad_fn（反向节点），节点记住
//      「前向时的输入张量 + 反向需要的中间量」，把整条计算轨迹连成 DAG。
//    - backward()：从 loss 出发做拓扑排序，按「链式法则」把梯度逐节点
//      反着传回去，只往叶子张量的 .grad 里累加。
//    - 本版刻意不做（PyTorch 有、教学版为聚焦 AD 而砍掉）：
//        view 共享存储、in-place 版本计数、广播、dtype promotion、GPU。
//      其中「in-place 版本计数」是 PyTorch 防止用脏图反向的关键，本版靠
//      「先 forward 后立刻 backward、backward 前别改叶子值」的约定来绕开，
//      注释里会反复提醒。
//
//  设计要点（和 matrix.h「无宏 / 无表达式模板」一脉相承）：
//    1) eager 求值：A * B 立即算出值存进结果张量，不搞懒求值表达式。
//    2) 节点多态：GradNode<T> 是抽象基类，Add/Mul/MatMul... 各实现一个
//       apply()（局部梯度公式），backward 只需要「按拓扑序轮流调 apply」。
//    3) 共享 ptr 表达引用：节点持有输入张量的 shared_ptr（句柄），
//       图是「新张量 -> 旧张量」的单向引用，天然无环，shared_ptr 不会泄漏。
//    4) 只有叶子张量真正累积 .grad，中间张量的 grad 在每次 backward 前清零
//       （多次 backward 会往叶子累加 —— 和 PyTorch 的 accumulate 语义一致，
//       每轮训练记得对参数调用 zero_grad()）。
// ============================================================================
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mini_mlmath/activation.h"
#include "mini_mlmath/check.h"
#include "mini_mlmath/matrix.h"

// 前置声明：Tensor 的 Impl 里要放 shared_ptr<GradNode>，但 GradNode 的完整
// 定义依赖 Tensor 的 Impl —— 先声明，定义放在 Tensor 之后。
template <typename T>
class GradNode;

// ============================================================================
//  Tensor<T>：带自动求导的矩阵
//
//  p_ 指向共享内核 Impl（值 + 梯度 + 反向外链）。所有拷贝共享同一内核：
//    auto b = a;      // b 与 a 指向同一个 Impl（值、梯度、图都共享）
//  这意味着 b 参与运算后，a.grad() 也能看到反向结果 —— 教学上把这种
//  「句柄语义」类比成 PyTorch 的 Python 对象绑定即可。
// ============================================================================
template <typename T>
class Tensor {
public:
    // ---- 共享内核：值 / 梯度 / 反向函数 ----
    // 字段公开是为了让 autograd.h 内部（GradNode）直接读写；用户代码请只
    // 通过下面公开方法访问，别直接动 grad_ / grad_fn。
    struct Impl {
        Matrix<T> value;                      // 前向计算出来的值
        Matrix<T> grad;                       // 反向累积的梯度（形状同 value）
        std::shared_ptr<GradNode<T>> grad_fn; // 谁产生了这个张量；null=叶子
        bool track = false;                   // 是否参与自动求导

        explicit Impl(const Matrix<T> &v, bool t)
            : value(v), grad(v.rows(), v.cols()), track(t) {}
    };

    // ---- 构造 ----
    Tensor() = default;

    // 从 Matrix 构造。track=true 表示「我是需要梯度的叶子（参数）」。
    // track=false 表示「我是常数（数据/标签），梯度直接丢弃」。
    explicit Tensor(const Matrix<T> &value, bool track = true)
        : p_(std::make_shared<Impl>(value, track)) {}

    // 从标量构造 1×1 张量（常数）。
    explicit Tensor(const T &scalar, bool track = false)
        : p_(std::make_shared<Impl>(Matrix<T>({{scalar}}), track)) {}

    // 内部：由算子构造结果张量（传进已算好的内核 + 已挂上的反向函数）。
    // 名字带下划线，表示「autograd 机制内部使用，普通用户请勿调用」。
    explicit Tensor(const std::shared_ptr<Impl> &impl)
        : p_(impl) {}

    // 拷贝/移动默认：拷贝 = 共享同一个 Impl（句柄语义，见文件头注释）。
    Tensor(const Tensor &) = default;
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    // ---- 形状与取值 ----
    std::size_t rows() const { return p_->value.rows(); }
    std::size_t cols() const { return p_->value.cols(); }
    std::size_t numel() const { return p_->value.rows() * p_->value.cols(); }

    // 前向值（只读）。想改一个叶子参数的值，用 set_value()（见下）。
    const Matrix<T> &value() const { return p_->value; }

    // 反向后叶子累积的梯度（只读）。
    const Matrix<T> &grad() const { return p_->grad; }

    // 1×1 张量取标量。CHECK 保证只有 1×1 才能取。
    T item() const {
        CHECK(p_->value.rows() == 1 && p_->value.cols() == 1)
            << "Tensor::item() only valid for 1x1 tensors, got "
            << p_->value.rows() << "x" << p_->value.cols();
        return p_->value(0, 0);
    }

    // ---- autograd 状态查询 ----
    bool requires_grad() const { return p_->track; }
    // 叶子 = 不是由任何算子产生（grad_fn 为空）。只有叶子能 set_value。
    bool is_leaf() const { return p_->grad_fn == nullptr; }

    // 把梯度清零（每轮训练前对参数调用，见文件头注释的 accumulate 语义）。
    void zero_grad() {
        p_->grad = Matrix<T>(p_->value.rows(), p_->value.cols());
    }

    // 原地修改叶子参数的值（手动 SGD：w.set_value(w.value() - lr * w.grad())）。
    // 只允许叶子：改中间张量会破坏已建好的图，梯度就错了。
    // 注意：只能在前向之后、同一步的 backward() 之前改，别动「正在反向的图」
    // 需要的叶子值 —— 这是本版没有版本计数器的替代约定（见文件头注释）。
    void set_value(const Matrix<T> &v) {
        CHECK(p_->grad_fn == nullptr)
            << "set_value() only allowed on leaf tensors (grad_fn == nullptr); "
               "changing a non-leaf would corrupt the backward graph";
        p_->value = v;
    }

    // ---- 反向传播 ----
    // 从本张量出发，初始上游梯度 = 全 1（等价于「先对自身做 sum 再反向」，
    // 所以对标量 loss 直接调用即可；对矩阵调用相当于求它对自身逐元素求和
    // 的梯度）。
    void backward();

private:
    std::shared_ptr<Impl> p_;

    // GradNode 是唯一需要直接读/写 Impl 字段的外部类（accumulate 累加梯度、
    // 图遍历读 grad_fn），所以只对它开 friend，其他一律走公开接口。
    friend class GradNode<T>;
};

// ============================================================================
//  GradNode<T>：反向节点（反向图的基本单元）
//
//  每个由算子产生的张量，其 grad_fn 指向一个 GradNode。节点做两件事：
//    1) 记住前向的输入张量（inputs_，用于图遍历 + 局部梯度公式取输入值）；
//    2) apply()：把「自己输出张量累积到的上游梯度」按链式法则改写成
//       每个输入的局部梯度，累加到输入张量的 grad 里。
//
//  图的引用方向（这里要仔细，关系到会不会内存泄漏）：
//    - 张量 Impl  ——shared_ptr——►  产生它的 GradNode（grad_fn）
//    - GradNode  ——shared_ptr——►  它的输入张量（inputs_，比它更早）
//    - GradNode  ——裸指针(不拥有)——►  它的输出张量（out_）
//  为什么 out_ 用裸指针？如果节点也 shared 持有输出，就会和「输出张量的
//  grad_fn 持有节点」构成循环引用：Impl ⇄ GradNode 互相牵制，引用计数永不
//  归零，图永远释放不掉（这是本文件踩过的坑，已修复，见下方 live_count_）。
//  out_ 用非拥有指针是安全的：节点被它的输出张量通过 grad_fn 持有，所以
//  「节点活着 ⇒ 输出活着」，节点访问 out_ 时输出必然还活着。
//  整体：张量→节点→更早的张量，严格单向，无环，图随最后一个用户句柄析构
//  而整体释放。live_count_ 是验证这一点用的（见 tests/autograd_test.cpp）。
// ============================================================================
template <typename T>
class GradNode {
public:
    // 当前存活的节点数（用于测试验证「图释放无泄漏」）。
    static inline std::atomic<std::int64_t> live_count_{0};

    GradNode() { live_count_.fetch_add(1); }
    virtual ~GradNode() { live_count_.fetch_sub(1); }

    // 把「输出张量累积到的梯度」分发到各输入的 grad_（按局部梯度公式）。
    virtual void apply() = 0;

    // 输出张量的梯度清零（backward 每次开始前清理中间节点）。
    void clear_out_grad() {
        out_->grad = Matrix<T>(out_->value.rows(), out_->value.cols());
    }

    // 后序遍历收集整张图（依赖在前、自身在后），供 backward 逆序调度。
    static std::vector<GradNode<T> *> collect_topo(
        std::shared_ptr<GradNode<T>> root) {
        std::unordered_set<GradNode<T> *> seen;
        std::vector<GradNode<T> *> topo;
        collect(root, seen, topo);
        return topo;
    }

protected:
    typename Tensor<T>::Impl *out_ = nullptr; // 输出（非拥有，见文件头注释）
    std::vector<Tensor<T>> inputs_;           // 我的直接输入（拥有，图遍历用）

    GradNode(const std::shared_ptr<typename Tensor<T>::Impl> &out,
             std::vector<Tensor<T>> inputs)
        : out_(out.get()), inputs_(std::move(inputs)) {
        // 参数化构造也要计入 live_count_（析构统一 -1，构造必须对称 +1）
        live_count_.fetch_add(1);
    }

    // 读取输入张量的前向值（apply 的局部梯度公式要用）。
    const Matrix<T> &val(const Tensor<T> &t) const { return t.p_->value; }

    // 读取「我输出张量」已累积到的上游梯度。
    const Matrix<T> &out_grad() const { return out_->grad; }

    // 把局部梯度 g 累加进输入张量的 grad_；常数张量（不追踪）直接跳过。
    void accumulate(const Tensor<T> &in, const Matrix<T> &g) {
        if (!in.requires_grad())
            return;
        in.p_->grad += g;
    }

private:
    static void collect(const std::shared_ptr<GradNode<T>> &n,
                        std::unordered_set<GradNode<T> *> &seen,
                        std::vector<GradNode<T> *> &topo) {
        if (!n || seen.count(n.get()))
            return;
        seen.insert(n.get());
        for (const auto &in : n->inputs_) {
            if (in.p_->grad_fn)
                collect(in.p_->grad_fn, seen, topo);
        }
        topo.push_back(n.get());
    }
};

// ============================================================================
//  反向传播主体
//  调度顺序：先收集拓扑序（依赖在前），再反着跑 —— 越接近 loss 的节点越先
//  apply，保证任何节点的输出梯度在被消费前已经累加完。
// ============================================================================
template <typename T>
void Tensor<T>::backward() {
    CHECK(p_->track) << "backward() on a tensor that does not require grad";
    if (!p_->grad_fn) {
        // 叶子直接 backward：初始梯度全 1（相当于对它做 sum 的梯度）。
        p_->grad = Matrix<T>(p_->value.rows(), p_->value.cols());
        for (std::size_t i = 0; i < p_->grad.rows() * p_->grad.cols(); ++i)
            p_->grad.data()[i] = T(1);
        return;
    }

    auto topo = GradNode<T>::collect_topo(p_->grad_fn);

    // 1) 清零所有中间节点的梯度（叶子不清 —— 多轮累积是期望语义）。
    for (auto *node : topo)
        node->clear_out_grad();

    // 2) 初始上游梯度：全 1。
    for (std::size_t i = 0; i < p_->grad.rows() * p_->grad.cols(); ++i)
        p_->grad.data()[i] = T(1);

    // 3) 拓扑逆序跑 apply()。
    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        (*it)->apply();
}

// ============================================================================
//  逐元素工具（autograd 内部使用）
// ============================================================================

// 逐元素乘（对应 numpy 的 A * B，注意：不是矩阵乘！矩阵乘用 matmul()）
template <typename T>
Matrix<T> ew_mul(const Matrix<T> &a, const Matrix<T> &b) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols())
        << "ew_mul: shape mismatch, got " << a.rows() << "x" << a.cols()
        << " vs " << b.rows() << "x" << b.cols();
    Matrix<T> r(a.rows(), a.cols());
    for (std::size_t i = 0; i < a.rows() * a.cols(); ++i)
        r.data()[i] = a.data()[i] * b.data()[i];
    return r;
}

// 逐元素除
template <typename T>
Matrix<T> ew_div(const Matrix<T> &a, const Matrix<T> &b) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols())
        << "ew_div: shape mismatch";
    Matrix<T> r(a.rows(), a.cols());
    for (std::size_t i = 0; i < a.rows() * a.cols(); ++i)
        r.data()[i] = a.data()[i] / b.data()[i];
    return r;
}

// 逐元素乘标量
template <typename T>
Matrix<T> ew_scale(const Matrix<T> &a, const T &s) {
    Matrix<T> r(a.rows(), a.cols());
    for (std::size_t i = 0; i < a.rows() * a.cols(); ++i)
        r.data()[i] = a.data()[i] * s;
    return r;
}

// 用 v 填满一张和 like 形状相同的矩阵
template <typename T>
Matrix<T> filled(const Matrix<T> &like, const T &v) {
    Matrix<T> r(like.rows(), like.cols());
    for (std::size_t i = 0; i < like.rows() * like.cols(); ++i)
        r.data()[i] = v;
    return r;
}

// ============================================================================
//  具体反向节点
//  每个节点的 apply() 是「链式法则的局部公式」，是自动求导里最该读懂的部分：
//  每个算子先写清楚 dL/d输入 = f(dL/d输出, 前向的输入值)，再照抄成代码。
// ============================================================================

// y = a + b  ==>  da = g, db = g          （加法把上游梯度原样分给两个输入）
template <typename T>
class AddNode : public GradNode<T> {
public:
    AddNode(const std::shared_ptr<typename Tensor<T>::Impl> &out,
            const Tensor<T> &a, const Tensor<T> &b)
        : GradNode<T>(out, {a, b}), a_(a), b_(b) {}
    void apply() override {
        this->accumulate(a_, this->out_grad());
        this->accumulate(b_, this->out_grad());
    }

private:
    Tensor<T> a_, b_;
};

// y = a - b  ==>  da = g, db = -g
template <typename T>
class SubNode : public GradNode<T> {
public:
    SubNode(const std::shared_ptr<typename Tensor<T>::Impl> &out,
            const Tensor<T> &a, const Tensor<T> &b)
        : GradNode<T>(out, {a, b}), a_(a), b_(b) {}
    void apply() override {
        this->accumulate(a_, this->out_grad());
        this->accumulate(b_, ew_scale(this->out_grad(), T(-1)));
    }

private:
    Tensor<T> a_, b_;
};

// y = a ⊙ b  ==>  da = g ⊙ b, db = g ⊙ a   （逐元素乘）
template <typename T>
class MulNode : public GradNode<T> {
public:
    MulNode(const std::shared_ptr<typename Tensor<T>::Impl> &out,
            const Tensor<T> &a, const Tensor<T> &b)
        : GradNode<T>(out, {a, b}), a_(a), b_(b) {}
    void apply() override {
        const auto &g = this->out_grad();
        this->accumulate(a_, ew_mul(g, this->val(b_)));
        this->accumulate(b_, ew_mul(g, this->val(a_)));
    }

private:
    Tensor<T> a_, b_;
};

// y = a / b  ==>  da = g / b, db = -g·a / b²
template <typename T>
class DivNode : public GradNode<T> {
public:
    DivNode(const std::shared_ptr<typename Tensor<T>::Impl> &out,
            const Tensor<T> &a, const Tensor<T> &b)
        : GradNode<T>(out, {a, b}), a_(a), b_(b) {}
    void apply() override {
        const auto &g = this->out_grad();
        const auto &av = this->val(a_);
        const auto &bv = this->val(b_);
        this->accumulate(a_, ew_div(g, bv));
        this->accumulate(b_, ew_scale(ew_div(ew_mul(g, av), ew_mul(bv, bv)), T(-1)));
    }

private:
    Tensor<T> a_, b_;
};

// y = a * s  ==>  da = g * s           （标量乘，s 存在节点里，不用记输入张量）
template <typename T>
class ScalarMulNode : public GradNode<T> {
public:
    ScalarMulNode(const std::shared_ptr<typename Tensor<T>::Impl> &out,
                  const Tensor<T> &a, const T &s)
        : GradNode<T>(out, {a}), a_(a), s_(s) {}
    void apply() override {
        this->accumulate(a_, ew_scale(this->out_grad(), s_));
    }

private:
    Tensor<T> a_;
    T s_;
};

// y = a @ b（矩阵乘，a: M×K, b: K×N）
//   da = g @ bᵀ   （M×N @ N×K = M×K，形状正好等于 a）
//   db = aᵀ @ g   （K×M @ M×N = K×N，形状正好等于 b）
//  这是「把矩阵乘反向看成两次新的矩阵乘」，配合 transposed() 即可。
template <typename T>
class MatMulNode : public GradNode<T> {
public:
    MatMulNode(const std::shared_ptr<typename Tensor<T>::Impl> &out,
               const Tensor<T> &a, const Tensor<T> &b)
        : GradNode<T>(out, {a, b}), a_(a), b_(b) {}
    void apply() override {
        const auto &g = this->out_grad();
        this->accumulate(a_, g * this->val(b_).transposed());
        this->accumulate(b_, this->val(a_).transposed() * g);
    }

private:
    Tensor<T> a_, b_;
};

// y = sum(a)：把 a 所有元素求和成 1×1
//   da = 把上游标量梯度填满成 a 的形状（每个元素都分到同一个值）
template <typename T>
class SumNode : public GradNode<T> {
public:
    SumNode(const std::shared_ptr<typename Tensor<T>::Impl> &out,
            const Tensor<T> &a)
        : GradNode<T>(out, {a}), a_(a) {}
    void apply() override {
        this->accumulate(a_, filled(this->val(a_), this->out_grad()(0, 0)));
    }

private:
    Tensor<T> a_;
};

// y = relu(a)  ==>  da = g ⊙ (a > 0 ? 1 : 0)
// 注意这是「前向读输入值」做掩码：relu'(a) 在 a>0 处是 1，其余是 0。
template <typename T>
class ReluNode : public GradNode<T> {
public:
    ReluNode(const std::shared_ptr<typename Tensor<T>::Impl> &out,
             const Tensor<T> &a)
        : GradNode<T>(out, {a}), a_(a) {}
    void apply() override {
        const auto &g = this->out_grad();
        const auto &av = this->val(a_);
        Matrix<T> local(av.rows(), av.cols());
        for (std::size_t i = 0; i < av.rows() * av.cols(); ++i)
            local.data()[i] = av.data()[i] > T(0) ? g.data()[i] : T(0);
        this->accumulate(a_, local);
    }

private:
    Tensor<T> a_;
};

// ============================================================================
//  算子（前向）+ 建图（反向）
//
//  每个算子三件事：
//    1) 算前向值（eager，直接用 Matrix<T> 现成运算）；
//    2) 判断要不要追踪（任一输入 requires_grad -> 输出也追踪）；
//    3) 要追踪就 new 一个对应节点、把结果张量的 grad_fn 指过去。
//  语义对照（和 matrix.h 的 A*B 不同！）：
//    Tensor 的 * 是逐元素乘（同 PyTorch 的 *）；
//    矩阵乘请用 matmul(a, b)（对应 PyTorch 的 @）。
// ============================================================================

// 逐元素加
template <typename T>
Tensor<T> operator+(const Tensor<T> &a, const Tensor<T> &b) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols())
        << "Tensor::operator+ shape mismatch, got "
        << a.rows() << "x" << a.cols() << " vs " << b.rows() << "x" << b.cols();
    const bool track = a.requires_grad() || b.requires_grad();
    auto impl = std::make_shared<typename Tensor<T>::Impl>(a.value() + b.value(), track);
    if (track)
        impl->grad_fn = std::make_shared<AddNode<T>>(impl, a, b);
    return Tensor<T>(impl);
}

// 逐元素减
template <typename T>
Tensor<T> operator-(const Tensor<T> &a, const Tensor<T> &b) {
    CHECK(a.rows() == b.rows() && a.cols() == b.cols())
        << "Tensor::operator- shape mismatch";
    const bool track = a.requires_grad() || b.requires_grad();
    auto impl = std::make_shared<typename Tensor<T>::Impl>(a.value() - b.value(), track);
    if (track)
        impl->grad_fn = std::make_shared<SubNode<T>>(impl, a, b);
    return Tensor<T>(impl);
}

// 逐元素乘（numpy A*B 语义；不是矩阵乘！矩阵乘用 matmul()）
template <typename T>
Tensor<T> operator*(const Tensor<T> &a, const Tensor<T> &b) {
    const bool track = a.requires_grad() || b.requires_grad();
    auto impl = std::make_shared<typename Tensor<T>::Impl>(ew_mul(a.value(), b.value()), track);
    if (track)
        impl->grad_fn = std::make_shared<MulNode<T>>(impl, a, b);
    return Tensor<T>(impl);
}

// 逐元素除
template <typename T>
Tensor<T> operator/(const Tensor<T> &a, const Tensor<T> &b) {
    const bool track = a.requires_grad() || b.requires_grad();
    auto impl = std::make_shared<typename Tensor<T>::Impl>(ew_div(a.value(), b.value()), track);
    if (track)
        impl->grad_fn = std::make_shared<DivNode<T>>(impl, a, b);
    return Tensor<T>(impl);
}

// 标量乘：s * a 和 a * s
template <typename T>
Tensor<T> operator*(const Tensor<T> &a, const T &s) {
    const bool track = a.requires_grad();
    auto impl = std::make_shared<typename Tensor<T>::Impl>(a.value() * s, track);
    if (track)
        impl->grad_fn = std::make_shared<ScalarMulNode<T>>(impl, a, s);
    return Tensor<T>(impl);
}

template <typename T>
Tensor<T> operator*(const T &s, const Tensor<T> &a) {
    return a * s;
}

// 一元负号：-a == a * (-1)
template <typename T>
Tensor<T> operator-(const Tensor<T> &a) {
    return a * T(-1);
}

// 矩阵乘（PyTorch 的 @）：A: M×K, B: K×N -> M×N
template <typename T>
Tensor<T> matmul(const Tensor<T> &a, const Tensor<T> &b) {
    CHECK(a.cols() == b.rows())
        << "Tensor::matmul: inner dims must match, got "
        << a.cols() << " vs " << b.rows();
    const bool track = a.requires_grad() || b.requires_grad();
    auto impl = std::make_shared<typename Tensor<T>::Impl>(a.value() * b.value(), track);
    if (track)
        impl->grad_fn = std::make_shared<MatMulNode<T>>(impl, a, b);
    return Tensor<T>(impl);
}

// 逐元素 ReLU（PyTorch 的 relu）—— 现代深度学习隐藏层默认激活
template <typename T>
Tensor<T> relu(const Tensor<T> &a) {
    const bool track = a.requires_grad();
    auto impl = std::make_shared<typename Tensor<T>::Impl>(relu(a.value()), track);
    if (track)
        impl->grad_fn = std::make_shared<ReluNode<T>>(impl, a);
    return Tensor<T>(impl);
}

// 全部元素求和 -> 1×1（PyTorch 的 loss 标量化常用写法）
template <typename T>
Tensor<T> sum(const Tensor<T> &a) {
    const T *d = a.value().data();
    T s = T(0);
    for (std::size_t i = 0; i < a.numel(); ++i)
        s += d[i];
    const bool track = a.requires_grad();
    Matrix<T> v(1, 1);
    v(0, 0) = s;
    auto impl = std::make_shared<typename Tensor<T>::Impl>(std::move(v), track);
    if (track)
        impl->grad_fn = std::make_shared<SumNode<T>>(impl, a);
    return Tensor<T>(impl);
}

// 打印前向值（调试用）
template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &t) {
    return os << t.value();
}
