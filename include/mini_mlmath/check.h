// ============================================================================
//  check.h —— glog 风格 CHECK(pred) << "message"（header-only）
//
//  用途：库函数入口处最常做的事是「先校验参数合法性再干活」，
//  比如矩阵乘法要检查 A.cols() == B.rows()、VarianceThreshold 要检查
//  X 至少有一列、transform 要检查列数与 fit 时一致。手写
//      if (!ok) throw std::invalid_argument("...");
//  会重复一堆样板，这里收敛成 glog 的流式写法：
//
//      CHECK(A.cols() == B.rows()) << "A.cols() must equal B.rows(), got "
//                                  << A.cols() << " vs " << B.rows();
//
//  语义：条件为 false 时抛 std::invalid_argument，消息就是 << 拼出来的
//  那段（前面自动带上 "CHECK(条件) failed at 文件:行号" 前缀，方便定位）；
//  条件为 true 时什么都不做。
//
//  为什么这里破例用宏？
//    matrix.h 的铁律「无宏」针对的是**库本体**：不用宏表达矩阵运算，
//    是为了让你看清计算本身（对比 Eigen 满屏的 EIGEN_XXX 宏）。而断言
//    CHECK 是个例外 —— 它要「拼出失败消息并抛异常」，本质是编译期文本
//    处理，必须用宏才能拿到 #cond（条件原文）和 __FILE__/__LINE__。
//    glog / abseil / gtest 全都这么干，教学上也正好拿它当「哪些场景
//    才值得用宏」的反例。
//
//  实现细节（为什么能析构时抛？）：
//    C++ 默认析构函数是 noexcept(true)，析构里 throw 会直接 std::terminate；
//    所以必须显式声明 ~Check() noexcept(false) 才能安全地把「抛异常」放进
//    析构。这是 glog 同款小把戏：CHECK 展开成 if/else，失败分支构造一个
//    临时对象，<< 攒消息，整个表达式结束临时对象析构时再抛。
//
//  展开示意：CHECK(x) << "msg" 等价于
//      if (x) (void)0; else Check("x", __FILE__, __LINE__) << "msg";
//    用 if/else 而非三元表达式，是因为 << 必须流进失败分支才创建的临时对象，
//    三元表达式做不到。完整的 if/else 也天然避开了悬空 else 问题。
// ============================================================================
#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

// 流式检查对象：CHECK 失败分支构造它，析构时抛 std::invalid_argument。
// 析构是唯一抛出点，所以 noexcept(false) 是必须的（见文件头注释）。
class Check {
public:
    // cond 是 #cond 得到的条件原文，file/line 是调用点，全塞进消息前缀
    Check(const char* cond, const char* file, int line)
        : message_("CHECK(" + std::string(cond) + ") failed at "
                   + std::string(file) + ":" + std::to_string(line) + ": ") {}

    // 拼消息：模板 + ostringstream，兼容 const char* / std::string /
    // int / double 等任意能 << 的类型。
    template <typename T>
    Check& operator<<(const T& v) {
        std::ostringstream os;
        os << v;
        message_ += os.str();
        return *this;
    }

    ~Check() noexcept(false) { throw std::invalid_argument(message_); }

private:
    std::string message_;
};

// CHECK(条件) << "失败时的消息"：条件为 false 才抛（glog 同款语义）。
// 注意这是本库唯一的宏，其余地方仍遵守 matrix.h 的「无宏」铁律。
#define CHECK(cond) \
    if (cond) (void)0; \
    else Check(#cond, __FILE__, __LINE__)
