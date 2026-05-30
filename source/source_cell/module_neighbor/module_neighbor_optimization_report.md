# ABACUS module_neighbor 近邻搜索优化报告

## 1. 优化目标

本次优化面向 ABACUS 旧版 `source/source_cell/module_neighbor` 近邻原子搜索模块，目标是按项目书 4.2 的并行划分策略改进现有网格搜索：

- 将原先对所有网格的全空间遍历改为只遍历当前原子所在网格附近的邻域网格。
- 在旧主算法 `Grid::Construct_Adjacent()` 中启用 OpenMP 线程级并行。
- 提供 MPI 三维空间域分解候选实现，将进程划分为 `Px x Py x Pz` 个空间子域。
- 在 MPI 候选实现中加入幽灵原子交换，避免子域边界处漏掉跨进程邻居。

## 2. 原始问题

旧算法中 `Construct_Adjacent_near_box()` 虽然计算了当前原子所在 box，但实际搜索时仍遍历全部 box：

```cpp
for (int box_i_x_adj = 0; box_i_x_adj < glayerX + glayerX_minus; box_i_x_adj++)
```

这会导致每个原子都扫描整个扩展空间的所有网格，复杂度接近“原子数 x 网格总数”。当周期镜像层数和原子规模增大时，大量距离判断是不必要的。

## 3. 已完成修改

### 3.1 邻域网格搜索

新增 `Grid::Construct_Adjacent_near_box_local()`，搜索流程为：

1. 通过 `getBox()` 得到当前原子所在 box。
2. 根据 `ceil(sradius / box_edge_length)` 计算需要检查的邻域 box 半径。
3. 只遍历 `[center - span, center + span]` 范围内的局部 box。
4. 保留原有距离判断 `dr <= sradius2`，保证不会误加入超出截断半径的原子。

该优化已经并入旧主算法：

```cpp
Grid::Construct_Adjacent()
```

默认主路径现在调用局部邻域搜索，而不是全空间网格遍历。

### 3.2 OpenMP 并行

`Grid::Construct_Adjacent()` 现在先构造原子任务列表，然后使用：

```cpp
#pragma omp parallel for schedule(static)
```

并行处理每个原子的邻居搜索。每个线程写入不同原子的 `all_adj_info[type][natom]`，避免多个线程同时修改同一个邻居表。

`Construct_Adjacent_omp()` 保留为兼容接口，目前转调 `Construct_Adjacent()`。

### 3.3 MPI 三维空间域分解

`GridParallel` 中新增三维域分解逻辑：

- 自动枚举 MPI 进程数的三因子分解。
- 选择子域形状更接近立方体的 `Px x Py x Pz`。
- 每个 rank 根据自身三维坐标负责一个空间子域。
- 本 rank 只为自己子域内的 owned atoms 构建邻居表。

### 3.4 幽灵原子交换

MPI 候选实现中加入 ghost atom 交换：

- owned atom：当前 rank 子域内真正负责计算邻居表的原子。
- ghost atom：相邻 rank 边界附近原子的只读副本。

每个 rank 从 owned atoms 中挑出靠近子域边界、距离小于搜索层厚度的原子，通过 `MPI_Isend` / `MPI_Irecv` 与 26 个相邻子域交换。随后用：

```text
owned atoms + ghost atoms
```

重建本地搜索网格。搜索时只遍历 owned atoms，但候选邻居可以来自 owned atoms 或 ghost atoms，从而避免跨进程边界漏邻居。

### 3.5 构建集成

`sltk_grid_parallel.cpp` 已在 `ENABLE_MPI` 时加入 `module_neighbor` 构建：

```cmake
if(ENABLE_MPI)
  target_sources(neighbor PRIVATE sltk_grid_parallel.cpp)
endif()
```

串行或非 MPI 构建不会编译该文件，避免引入 MPI 依赖。

## 4. 与项目书 4.2 的对应关系

| 项目书要求 | 当前完成情况 |
| --- | --- |
| MPI 域分解沿三维空间进行 | 已在 `GridParallel` 中实现 |
| 自动确定 `Px x Py x Pz` | 已实现，优先选择接近立方体的分解 |
| 每个 MPI 进程负责一个子域 | 已实现 owned atom 子域归属 |
| OpenMP 在线程内并行搜索 | 已并入 `Grid::Construct_Adjacent()` |
| 每个线程独立搜索负责原子的近邻列表 | 已实现 |
| 幽灵原子交换 | 已在 `GridParallel` 中实现 26 邻域交换 |
| 最小化通信面 | 已按边界层筛选发送原子，仍可继续优化通信打包 |

## 5. 当前限制

- `GridParallel` 已进入 MPI 构建，但尚未替换 ABACUS 所有调用路径；默认旧主路径主要使用邻域搜索和 OpenMP。
- ghost atom 交换目前采用 26 邻域直接交换，代码清晰但通信打包仍有优化空间。

## 6. 材料算例测试：近邻网格串行版与 OpenMP 并行版对比

### 6.1 测试方法

本次测试复用旧算法测试报告中的四个材料算例，但性能对比不再使用旧的全空间遍历算法作为基线。原因是旧算法与当前算法同时存在“搜索策略差异”和“并行差异”，直接比较会把近邻网格搜索带来的算法级加速也计入并行加速，口径不公平。

因此，本节采用更公平的对比方式：

- 串行基线：使用当前近邻网格搜索算法，设置 `OMP_NUM_THREADS=1`。
- 并行版本：使用相同近邻网格搜索算法，设置 `OMP_NUM_THREADS=2`、`OMP_NUM_THREADS=4` 和 `OMP_NUM_THREADS=8`。

也就是说，本节只评估 OpenMP 并行带来的收益，而不把“全空间遍历改为近邻网格遍历”的算法级收益混入并行加速比。

测试材料如下：

| 体系 | 原子数 | 场景 | 结构类型 |
| --- | ---: | --- | --- |
| Al fcc | 1,000 | PW | 金属 |
| Si diamond | 2,000 | LCAO | 半导体 |
| NaCl | 3,000 | PW | 离子晶体 |
| TiO2 rutile | 4,200 | LCAO | 复杂氧化物 |

测试程序为：

```text
source/source_cell/module_neighbor/test/sltk_material_runtime_runner.cpp
```

测试口径与旧报告一致：

- `build_ms`：统计 `Grid_Driver::init()` 构造网格与近邻表的耗时。
- `avg_neighbors`：统计每个原子的平均近邻数。
- `memory_mb`：估算 `atoms_in_box` 与 `all_adj_info` 的核心数据结构内存。
- `traverse_ms`：统计通过 `Find_atom()` 遍历查询全部原子的耗时。

旧算法报告中的平均近邻数仅作为正确性参考，不作为性能基线。

### 6.2 正确性结果

当前近邻网格算法在不同线程数下平均近邻数保持一致，并与旧算法报告中的平均近邻数一致：

| 体系 | 旧算法平均近邻数 | 近邻网格 1 线程 | 近邻网格 8 线程 | 结果 |
| --- | ---: | ---: | ---: | --- |
| Al fcc | 12.0 | 12.0 | 12.0 | 一致 |
| Si diamond | 4.0 | 4.0 | 4.0 | 一致 |
| NaCl | 6.0 | 6.0 | 6.0 | 一致 |
| TiO2 rutile | 4.0 | 4.0 | 4.0 | 一致 |

这说明局部邻域网格搜索没有漏掉旧算法能找到的近邻原子，并且 OpenMP 并行没有改变邻居表结果。

### 6.3 性能对比

| 体系 | 1 线程 build_ms | 2 线程 build_ms | 2 线程加速比 | 4 线程 build_ms | 4 线程加速比 | 8 线程 build_ms | 8 线程加速比 | 内存 MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Al fcc | 5.163 | 5.327 | 0.97x | 5.753 | 0.90x | 2.707 | 1.91x | 1.74739 |
| Si diamond | 12.040 | 9.526 | 1.26x | 7.467 | 1.61x | 4.649 | 2.59x | 3.68440 |
| NaCl | 13.087 | 8.077 | 1.62x | 10.614 | 1.23x | 5.771 | 2.27x | 5.07750 |
| TiO2 rutile | 20.255 | 11.235 | 1.80x | 14.896 | 1.36x | 9.779 | 2.07x | 7.54429 |

从结果可以看到，OpenMP 并行收益与体系规模和线程数有关。2 线程和 4 线程结果存在一定波动，说明这些小中型算例会受到线程调度、缓存访问和系统负载影响；8 线程在本轮测试中对四个材料都取得了正向加速，构建阶段加速比约为 1.91x 到 2.59x。

### 6.4 测试输出

近邻网格串行基线输出，`OMP_NUM_THREADS=1`：

```text
[sltk] Al fcc / 1000 atoms / PW / metal: atoms=1000, avg_neighbors=12, build_ms=5.163, memory_mb=1.74739, traverse_ms=0.117, total_ms=5.28
[sltk] Si diamond / 2000 atoms / LCAO / semiconductor: atoms=2000, avg_neighbors=4, build_ms=12.04, memory_mb=3.6844, traverse_ms=0.172, total_ms=12.212
[sltk] NaCl / 3000 atoms / PW / ionic crystal: atoms=3000, avg_neighbors=6, build_ms=13.087, memory_mb=5.0775, traverse_ms=0.128, total_ms=13.215
[sltk] TiO2 rutile / 4200 atoms / LCAO / complex oxide: atoms=4200, avg_neighbors=4, build_ms=20.255, memory_mb=7.54429, traverse_ms=0.319, total_ms=20.574
```

OpenMP 2 线程输出，`OMP_NUM_THREADS=2`：

```text
[sltk] Al fcc / 1000 atoms / PW / metal: atoms=1000, avg_neighbors=12, build_ms=5.327, memory_mb=1.74739, traverse_ms=0.129, total_ms=5.456
[sltk] Si diamond / 2000 atoms / LCAO / semiconductor: atoms=2000, avg_neighbors=4, build_ms=9.526, memory_mb=3.6844, traverse_ms=0.073, total_ms=9.599
[sltk] NaCl / 3000 atoms / PW / ionic crystal: atoms=3000, avg_neighbors=6, build_ms=8.077, memory_mb=5.0775, traverse_ms=0.172, total_ms=8.249
[sltk] TiO2 rutile / 4200 atoms / LCAO / complex oxide: atoms=4200, avg_neighbors=4, build_ms=11.235, memory_mb=7.54429, traverse_ms=0.15, total_ms=11.385
```

OpenMP 4 线程输出，`OMP_NUM_THREADS=4`：

```text
[sltk] Al fcc / 1000 atoms / PW / metal: atoms=1000, avg_neighbors=12, build_ms=5.753, memory_mb=1.74739, traverse_ms=0.158, total_ms=5.911
[sltk] Si diamond / 2000 atoms / LCAO / semiconductor: atoms=2000, avg_neighbors=4, build_ms=7.467, memory_mb=3.6844, traverse_ms=0.158, total_ms=7.625
[sltk] NaCl / 3000 atoms / PW / ionic crystal: atoms=3000, avg_neighbors=6, build_ms=10.614, memory_mb=5.0775, traverse_ms=0.146, total_ms=10.76
[sltk] TiO2 rutile / 4200 atoms / LCAO / complex oxide: atoms=4200, avg_neighbors=4, build_ms=14.896, memory_mb=7.54429, traverse_ms=0.28, total_ms=15.176
```

OpenMP 8 线程输出，`OMP_NUM_THREADS=8`：

```text
[sltk] Al fcc / 1000 atoms / PW / metal: atoms=1000, avg_neighbors=12, build_ms=2.707, memory_mb=1.74739, traverse_ms=0.056, total_ms=2.763
[sltk] Si diamond / 2000 atoms / LCAO / semiconductor: atoms=2000, avg_neighbors=4, build_ms=4.649, memory_mb=3.6844, traverse_ms=0.06, total_ms=4.709
[sltk] NaCl / 3000 atoms / PW / ionic crystal: atoms=3000, avg_neighbors=6, build_ms=5.771, memory_mb=5.0775, traverse_ms=0.111, total_ms=5.882
[sltk] TiO2 rutile / 4200 atoms / LCAO / complex oxide: atoms=4200, avg_neighbors=4, build_ms=9.779, memory_mb=7.54429, traverse_ms=0.124, total_ms=9.903
```

### 6.5 旧全空间遍历算法参考对比

旧报告中的原始算法使用全空间网格遍历，每个原子会扫描所有候选 box。该结果可以用来说明“全空间遍历改为近邻网格遍历”的算法级收益，但不能作为 OpenMP 并行加速比。

| 体系 | 旧全空间遍历 build_ms | 近邻网格 1 线程 build_ms | 算法级加速比 |
| --- | ---: | ---: | ---: |
| Al fcc | 140.929 | 5.163 | 27.30x |
| Si diamond | 573.257 | 12.040 | 47.61x |
| NaCl | 1343.78 | 13.087 | 102.68x |
| TiO2 rutile | 2554.68 | 20.255 | 126.12x |

这组结果表明，主要的数量级提升来自搜索策略本身：用几何空间网格和邻域 box 搜索替代全空间候选 box 遍历。并行收益应以上一节中“近邻网格 1 线程 vs 近邻网格多线程”的结果为准。

### 6.6 结果分析

本节只分析并行收益，不再把旧全空间遍历算法作为性能基线。测试表明：

1. 近邻网格串行版和 OpenMP 并行版的平均近邻数完全一致，说明并行化没有改变计算结果。
2. OpenMP 并行在较大算例上有收益，但小算例会受到线程开销影响。
3. 在本机测试中，8 线程总体最快，但并行效率没有线性增长，说明当前算例规模仍偏小，线程开销和内存访问开销较明显。
4. 旧全空间遍历算法与近邻网格 1 线程算法的差距可作为算法级优化收益参考，但不应称为并行加速比。

## 7. 后续建议

1. 增加单元测试：比较原全空间搜索和局部邻域搜索的邻居表是否一致。
2. 增加 MPI 测试：验证 `GridParallel` 在 1、2、4、8 rank 下邻居数一致。
3. 扩展材料性能测试规模，例如 8,000、16,000、32,000 原子，观察大体系并行效率。
4. 若要完全替换生产路径，再设计 `Grid_Driver` 或 `AtomArrange` 层面的 MPI 并行入口开关。
