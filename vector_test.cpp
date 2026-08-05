// ============================================================================
//  vector_test.cpp —— mini-mlmath 的向量测试程序
//
//  验证三件事（全部手算可核对）：
//    1. 点积：{1,2,3}·{4,5,6} = 1*4 + 2*5 + 3*6 = 4+10+18 = 32
//    2. 模长：|{3,4}| = sqrt(9+16) = 5（经典的 3-4-5 直角三角形）
//    3. 余弦相似度：只看方向不看长度
//       {1,0}·{1,0} = 1  （完全同向）
//       {1,0}·{0,1} = 0  （垂直）
//       {1,0}·{-1,0} = -1（完全反向）
//       {1,0}·{5,0} = 1  （长度差 5 倍，方向相同，相似度还是 1）
//
//  实现与原理见 vector.h 头部注释。
// ============================================================================
#include "vector.h"

#include <cmath>
#include <iostream>

// Windows 下让控制台按 UTF-8 输出中文，理由同 matrix_test.cpp。
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// ----------------------------------------------------------------------------
// 正确性验证：点积 / 模长 / 余弦相似度
// ----------------------------------------------------------------------------
void verify_vector() {
    std::cout << "===== 正确性验证：Vector =====\n";

    // 1) 点积：{1,2,3}·{4,5,6} = 32
    const Vector<double> a = Vector<double>::fromList({1, 2, 3});
    const Vector<double> b = Vector<double>::fromList({4, 5, 6});
    const double d = a.dot(b);
    std::cout << a << " · " << b << " = " << d
              << (std::abs(d - 32.0) < 1e-12 ? "  ✓\n" : "  ✗\n");

    // 2) 模长：|{3,4}| = 5
    const Vector<double> c = Vector<double>::fromList({3, 4});
    const double n = c.norm();
    std::cout << "|" << c << "| = " << n
              << (std::abs(n - 5.0) < 1e-12 ? "  ✓\n" : "  ✗\n");

    // 3) 余弦相似度四个用例
    const Vector<double> e1 = Vector<double>::fromList({1, 0});
    const Vector<double> e2 = Vector<double>::fromList({0, 1});
    const Vector<double> e1n = Vector<double>::fromList({-1, 0});
    const Vector<double> e1long = Vector<double>::fromList({5, 0});

    std::cout << "cos(" << e1 << ", " << e1 << ")     = "
              << e1.cosine_similarity(e1) << "  (同向=1)\n";
    std::cout << "cos(" << e1 << ", " << e2 << ")     = "
              << e1.cosine_similarity(e2) << "  (垂直=0)\n";
    std::cout << "cos(" << e1 << ", " << e1n << ")    = "
              << e1.cosine_similarity(e1n) << "  (反向=-1)\n";
    std::cout << "cos(" << e1 << ", " << e1long << ") = "
              << e1.cosine_similarity(e1long)
              << "  (长度无关，仍是1)\n";

    const double cs = e1.cosine_similarity(e1long);
    std::cout << "✓ 余弦相似度范围正确"
              << (std::abs(cs - 1.0) < 1e-12 ? "（长度不影响）\n" : "✗\n");
}

// ----------------------------------------------------------------------------
// 入口
// ----------------------------------------------------------------------------
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::cout << "mini-mlmath —— 教学用迷你矩阵库（向量测试）\n\n";

    verify_vector();

    std::cout << "\n向量 API：v.dot(w) 点积，v.norm() 模长，\n";
    std::cout << "v.cosine_similarity(w) 余弦相似度（实现见 vector.h）。\n";
    return 0;
}
