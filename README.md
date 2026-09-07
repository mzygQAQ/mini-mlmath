# mini-mlmath

教学用迷你机器学习数学库：纯手写、零第三方依赖，从矩阵、向量一路写到 softmax、
激活函数、特征选择、感知机、线性回归、KNN、自动求导，用来研究「数值计算和机器
学习到底是怎么实现的」。

对比 Eigen / numpy / sklearn 刻意保留三大简化：
- **无宏**（唯一例外是 `check.h` 里的 `CHECK` 断言——它要拿条件原文和调用点
  文件/行号，只能靠宏，见该文件注释）
- **无表达式模板**（`A*B` 立即求值，不返回懒求值表达式）
- **无 BLAS / 现成数值库对接**（矩阵乘法就是自己写的三层循环，模型就是手写的类）

## 目录结构（标准 C++ 布局，按模块分文件夹）

```
mini-mlmath/
├── CMakeLists.txt            # 顶层：INTERFACE 库 + 挂载 tests
├── include/
│   └── mini_mlmath/          # 库本体，全部 header-only
│       ├── matrix.h          #   Matrix<T>：动态大小、行主序 + 三种乘法
│       ├── vector.h          #   Vector<T>：点积 / 模长 / 余弦相似度
│       ├── softmax.h           #   数值稳定的 softmax → [讲解](docs/softmax.md)
│       ├── activation.h        #   激活函数：sigmoid + ReLU → [讲解](docs/activation.md)
│       ├── positional_encoding.h  #   正弦位置编码（骨架，待实现）：给注意力注入位置信息
│       ├── autograd.h          #   迷你自动求导：Tensor + 记录式反向图 → [讲解](docs/autograd.md)
│       ├── statistics.h        #   基础描述性统计：mean / median / mode / variance / stddev
│       ├── random.h          #   类似 numpy.random：均匀/正态随机标量与矩阵
│       ├── check.h           #   glog 风格 CHECK(pred) << "msg" 断言
│       ├── feature_selection/    # 特征选择模块（有监督/无监督）
│       │   ├── variance_threshold.h   # 方差法：方差低于阈值的列删掉（无监督）
│       │   └── correlation_selector.h # 相关系数法：|r| 低于阈值的列删掉（有监督）
│       └── ml/                   # 机器学习模型模块
│           ├── perceptron.h         #   感知机 → [原理](docs/perceptron.md) / [逻辑门](docs/logic_gates.md)
│           ├── linear_regression.h  #   线性回归 → [原理](docs/linear_regression.md)
│           ├── knn.h               #   K 近邻 → [原理](docs/knn.md)
│           └── kmeans.h            #   K 均值聚类（骨架，待实现）→ [原理](docs/kmeans.md)
├── docs/                      # 文档：原理讲解 + 配图
│   ├── math_functions.md      #   数学地基：幂/指数/对数/三角 + 导数速查
│   ├── chain_rule.md          #   链式法则：零基础入门（反向传播的地基）
│   ├── perceptron.md          #   感知机原理（结构 / 学习规则 / 收敛定理）
│   ├── logic_gates.md         #   逻辑门：AND / OR / NAND 权重推导 + XOR 不可分
│   ├── activation.md          #   激活函数：sigmoid / ReLU vs 阶跃、ReLU 为什么成为现代默认
│   ├── linear_regression.md   #   线性回归：模型、闭式解 / 梯度下降、R²、bias folding
│   ├── knn.md                 #   KNN：常见 metric、搜索策略（暴力/KDTree/BallTree）
│   ├── kmeans.md              #   KMeans：Lloyd 迭代、初始化策略、k 怎么选
│   ├── softmax.md             #   softmax：减 max 数值稳定性、attention 用法、温度
│   ├── positional_encoding.md #   正弦位置编码：为什么需要、多尺度频率、PE 矩阵（骨架期）
│   ├── autograd.md            #   自动求导：反向图 + 梯度怎么算出来
│   └── images/                #   配图（手写 SVG，零依赖）
├── tests/                    # 测试程序，每个模块一个
    ├── CMakeLists.txt        #   每个 *_test.cpp 一个可执行 + 编译优化选项
    ├── matrix_test.cpp       #   矩阵乘法正确性验证 + 三版性能对比
    ├── softmax_test.cpp      #   softmax 数值稳定性 / 归一化验证 → [讲解](docs/softmax.md)
    ├── vector_test.cpp       #   点积 / 模长 / 余弦相似度验证
    ├── random_check.cpp      #   随机数分布抽样 smoke test
    ├── logic_gate.cpp        #   感知机学逻辑门 → [讲解](docs/logic_gates.md) / [原理](docs/perceptron.md)
    ├── perceptron_test.cpp   #   端到端测 Perceptron 类
    ├── knn_test.cpp          #   KNN：多数投票 + metric / strategy 扩展点
    ├── autograd_test.cpp     #   自动求导：手算链式法则 + MLP 数值梯度对拍
    └── kmeans_test.cpp       #   KMeans：聚类 + 初始化策略扩展点（骨架期）
```

头文件引用统一写 `<mini_mlmath/xxx.h>`，`include/` 是头文件搜索根。

## 文档

以下按**推荐的学习顺序**排列：先数学地基与微积分工具，再走「感知机 → 神经网络组件」与「线性模型 → 距离模型」两条线，最后落到自动求导。

- [常见数学函数与图像](docs/math_functions.md) —— 数学地基：幂/指数/对数/三角函数图像 + 定义域/值域/导数速查表
- [链式法则](docs/chain_rule.md) —— 微积分工具：零基础入门，导数直觉、路径相加、反向传播（autograd 的地基）
- [感知机原理](docs/perceptron.md) —— 入门第一课：结构、学习规则、收敛定理、bias folding
- [逻辑门与决策边界](docs/logic_gates.md) —— 感知机的应用：AND / OR / NAND 的权重推导（不等式组法）、XOR 为什么线性不可分
- [激活函数](docs/activation.md) —— sigmoid vs 阶跃、ReLU 为什么成为现代默认、为什么多层必须可导、数值稳定性
- [线性回归](docs/linear_regression.md) —— 线性模型：正规方程 / 梯度下降、R²、bias folding、和感知机的对照
- [KNN](docs/knn.md) —— 距离模型：常见 metric（L1/L2/L∞/余弦/汉明…）、搜索策略（暴力/KDTree/BallTree）、k 与投票
- [KMeans](docs/kmeans.md) —— 无监督：Lloyd 迭代、初始化策略（随机 / k-means++）、n_init 多起点、k 怎么选（骨架期，核心算法待实现）
- [Softmax](docs/softmax.md) —— 减 max 救命符、按行归一化（attention 用法）、温度 T、KV cache 简化
- [正弦位置编码](docs/positional_encoding.md) —— 注意力为什么是「集合运算」、多尺度频率、PE 矩阵（骨架期，待实现）
- [自动求导](docs/autograd.md) —— 压轴：Tensor + 反向图，梯度怎么在图上算出来（reverse-mode AD）

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
./build/tests/matrix_test       # 矩阵乘法：正确性 + 三版性能对比
./build/tests/softmax_test      # softmax：数值稳定性 + 按行归一化
./build/tests/vector_test       # 向量：点积 / 模长 / 余弦相似度
./build/tests/random_check      # 随机数分布抽样 smoke test
./build/tests/logic_gate        # 感知机学逻辑门（AND/OR 可学，XOR 线性不可分）
./build/tests/perceptron_test   # 感知机：端到端测试
./build/tests/knn_test          # KNN：多数投票 + metric / strategy 扩展点
./build/tests/autograd_test     # 自动求导：手算链式法则 + MLP 数值梯度对拍
./build/tests/kmeans_test       # KMeans：聚类 + 初始化策略（骨架期，SKIPPED 为预期）
./build/tests/positional_encoding_test   # 正弦位置编码（骨架期，SKIPPED 为预期）
```

Windows 直接用 VS2022「打开本地文件夹」指向本目录（切 Release、选 x64）。

> 注意：编译选项带 `-march=native`，只对当前机器有效，拷到别的机器需删掉或改为具体型号（如 `-march=haswell`）。
