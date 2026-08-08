# mini-mlmath

教学用迷你矩阵库：纯手写、零第三方依赖，用来研究「矩阵库到底怎么实现的」。

对比 Eigen 刻意保留三大简化：
- **无宏**（只有 `#pragma once`）
- **无表达式模板**（`A*B` 立即求值，不返回懒求值表达式）
- **无 BLAS 对接**（乘法就是自己写的三层循环）

## 目录结构（标准 C++ 布局，按模块分文件夹）

```
mini-mlmath/
├── CMakeLists.txt            # 顶层：INTERFACE 库 + 挂载 tests
├── include/
│   └── mini_mlmath/          # 库本体，全部 header-only
│       ├── matrix.h          #   Matrix<T>：动态大小、行主序 + 三种乘法
│       ├── vector.h          #   Vector<T>：点积 / 模长 / 余弦相似度
│       ├── softmax.h         #   数值稳定的 softmax（一维 + 按行矩阵版）
│       └── feature_selection/    # 特征选择模块
│           └── variance_threshold.h  # 基于方差的特征选择（VarianceThreshold）
└── tests/                    # 测试程序，每个模块一个
    ├── CMakeLists.txt        #   每个 *_test.cpp 一个可执行 + 编译优化选项
    ├── matrix_test.cpp       #   矩阵乘法正确性验证 + 三版性能对比
    ├── softmax_test.cpp      #   softmax 数值稳定性 / 归一化验证
    └── vector_test.cpp       #   点积 / 模长 / 余弦相似度验证
```

头文件引用统一写 `<mini_mlmath/xxx.h>`，`include/` 是头文件搜索根。

## 三种乘法（越往后越接近真实 BLAS）

1. `multiply_naive`：ikj 三层循环，讲清为什么 `j` 在最内层（行主序连续访问）
2. `multiply_blocked`：分块 `kBlock=64`，讲清缓存分块思想
3. `multiply_packed`：分块 + 数据打包（对应 OpenBLAS / Eigen `GeneralBlockPanelKernel.h` 的做法）

## 三个真实性能教训（写在注释里的实测数据）

1. GCC 的 `-O2` 不自动向量化，必须 `-O3`；实测 N=2048 时 `-O2` 下三版全部约 4.1s 打平
2. 分块不打包反而更慢：行主序的「块」在 k 方向有 stride，真实 GEMM 必须先 pack 进连续 buffer
3. 分块优势只在矩阵超过缓存后兑现：N>=1024 后 blocked/packed 才拉开差距，N=2048 时 packed 比 naive 快约 3 倍

## 编译

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/tests/matrix_test
./build/tests/softmax_test
```

Windows 直接用 VS2022「打开本地文件夹」指向本目录（切 Release、选 x64）。

> 注意：编译选项带 `-march=native`，只对当前机器有效，拷到别的机器需删掉或改为具体型号（如 `-march=haswell`）。
