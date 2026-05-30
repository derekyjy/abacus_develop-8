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
- 尚未完成系统性能测试，需要用材料 benchmark 验证正确性、加速比和并行效率。

## 6. 后续建议

1. 增加单元测试：比较原全空间搜索和局部邻域搜索的邻居表是否一致。
2. 增加 MPI 测试：验证 `GridParallel` 在 1、2、4、8 rank 下邻居数一致。
3. 用 Al、Si、NaCl、TiO2 四类材料做性能测试，记录串行时间、OpenMP 时间、MPI+OpenMP 时间和加速比。
4. 若要完全替换生产路径，再设计 `Grid_Driver` 或 `AtomArrange` 层面的并行入口开关。
