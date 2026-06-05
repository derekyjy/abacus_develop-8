# ABACUS module_neighbor 近邻搜索优化报告

## 1. 优化目标

本次优化面向 ABACUS 旧版 `source/source_cell/module_neighbor` 近邻原子搜索模块。项目书里提到的几个方向基本可以归到三类：先把搜索范围缩小，再把任务并行化，最后处理不同原子附近候选数量不均带来的负载差异。

本阶段实际完成了以下内容：

- 保留旧接口 `Grid` / `Grid_Driver`，不改变上层调用方式。
- 将原来的全空间 box 遍历改成局部邻域 box 搜索。
- 在 `Grid::Construct_Adjacent()` 中启用 OpenMP 线程级并行。
- 保留 `GridParallel` 中的 MPI 三维空间域划分和 ghost atom 交换候选实现。
- 新增自适应 box 剪枝：根据候选 box 内真实原子的包围盒判断这个 box 是否可能含有半径内近邻。
- 将 OpenMP 调度从静态分块改为 `guided` 调度，用于线程级动态负载均衡。

## 2. 原始问题

旧实现里 `Construct_Adjacent_near_box()` 虽然计算了当前原子所在 box，但真正搜索时仍然遍历扩胞后的全部 box：

```cpp
for (int box_i_x_adj = 0; box_i_x_adj < glayerX + glayerX_minus; box_i_x_adj++)
{
    for (int box_i_y_adj = 0; box_i_y_adj < glayerY + glayerY_minus; box_i_y_adj++)
    {
        for (int box_i_z_adj = 0; box_i_z_adj < glayerZ + glayerZ_minus; box_i_z_adj++)
        {
            ...
        }
    }
}
```

这样做的问题是很直接的：每个原子都要扫一遍扩展空间中的所有 box，大量 box 要么为空，要么距离当前原子很远。体系规模变大、周期镜像层数增加后，距离判断次数会迅速增加。

另外，即使有 OpenMP 并行，不同原子的实际候选邻居数量也不完全一样。特别是在复杂氧化物、非均匀结构或边界附近，线程之间容易出现有的线程已经结束、有的线程还在处理大任务的情况。

## 3. 已完成的优化

### 3.1 局部邻域 box 搜索

主路径现在不再遍历全部 box，而是通过 `getBox()` 找到当前原子所在 box，再根据搜索半径计算需要检查的邻域跨度：

```cpp
const int search_span = std::max(1, static_cast<int>(std::ceil(sradius / box_edge_length)));
const int x_begin = std::max(0, box_i_x - search_span);
const int x_end = std::min(box_nx - 1, box_i_x + search_span);
```

随后只遍历 `[center - span, center + span]` 范围内的局部 box。最后仍然保留精确距离判断：

```cpp
if (dr != 0.0 && dr <= this->sradius2)
{
    all_adj_info[fatom1.type][fatom1.natom].push_back(fatom2);
}
```

也就是说，这一步只是减少候选集合，不改变近邻判定标准。

### 3.2 OpenMP 并行

`Grid::Construct_Adjacent()` 先把原胞内所有原子整理成任务列表：

```cpp
std::vector<std::pair<int, int>> atom_pairs;
for (int i_type = 0; i_type < ucell.ntype; i_type++)
{
    for (int j_atom = 0; j_atom < ucell.atoms[i_type].na; j_atom++)
    {
        atom_pairs.push_back({i_type, j_atom});
    }
}
```

每个任务只写入一个原子的邻接表 `all_adj_info[type][natom]`，线程之间不会同时写同一个 vector，因此可以直接并行处理。

### 3.3 自适应 box 剪枝

这次新增的优化重点是自适应搜索。构建 `atoms_in_box` 时，同时记录每个 box 内真实原子的坐标包围盒：

```cpp
struct BoxBounds
{
    double x_min = std::numeric_limits<double>::max();
    double y_min = std::numeric_limits<double>::max();
    double z_min = std::numeric_limits<double>::max();
    double x_max = std::numeric_limits<double>::lowest();
    double y_max = std::numeric_limits<double>::lowest();
    double z_max = std::numeric_limits<double>::lowest();
    bool empty = true;
};
```

放入原子时同步更新：

```cpp
this->atoms_in_box[box_i_x][box_i_y][box_i_z].push_back(atom);
this->box_bounds[box_i_x][box_i_y][box_i_z].add_atom(atom);
```

搜索某个原子时，先计算该原子到候选 box 包围盒的最小可能距离。如果这个最小距离已经大于搜索半径，那么该 box 内所有原子都不可能成为近邻，可以直接跳过：

```cpp
if (!this->box_may_contain_neighbor(fatom, this->box_bounds[box_i_x_adj][box_i_y_adj][box_i_z_adj]))
{
    continue;
}
```

核心判断如下：

```cpp
const double min_distance2 = axis_distance2(fatom.x, bounds.x_min, bounds.x_max)
                           + axis_distance2(fatom.y, bounds.y_min, bounds.y_max)
                           + axis_distance2(fatom.z, bounds.z_min, bounds.z_max);
return min_distance2 <= this->sradius2;
```

这一步的意义是：不是机械地按照固定 `search_span` 扫所有邻域 box，而是根据每个 box 里实际原子的分布进一步缩小搜索范围。它仍然是精确剪枝，不会漏掉半径内近邻。

为了做 A/B 测试，代码里保留了一个编译期关闭开关：

```cpp
#ifndef SLTK_DISABLE_ADAPTIVE_BOX_PRUNING
...
#endif
```

正常编译时该优化默认开启。

### 3.4 动态负载均衡

原来的 OpenMP 调度是静态分块：

```cpp
#pragma omp parallel for schedule(static)
```

这对每个原子工作量差不多的体系没问题，但近邻搜索里每个原子要扫的候选 box 和候选原子数量不一定一样。当前改成：

```cpp
#pragma omp parallel for schedule(guided, 16)
```

`guided` 调度前期给线程较大的任务块，后期任务块逐渐变小。这样既能减少动态调度的频繁取任务开销，又能在尾部让空闲线程继续领取剩余任务。测试中它比 `dynamic,16` 更稳，也比原来的 `static` 更适合较大体系。

### 3.5 MPI 空间域划分和 ghost atom 交换

`GridParallel` 中已有 MPI 候选实现，主要包含：

- 自动枚举 MPI 进程数的三因子分解，选择较接近立方体的 `Px x Py x Pz`。
- 每个 rank 按三维坐标负责一个空间子域。
- 子域边界附近的 owned atoms 会作为 ghost atoms 发给相邻 rank。
- 本 rank 用 `owned atoms + ghost atoms` 重建局部搜索网格，避免跨进程边界漏邻居。

ghost atom 交换采用 26 邻域直接交换，通信路径清楚，后续仍可以继续优化打包和减少重复发送。

## 4. 测试方式

完整 ABACUS 顶层 CMake 会继续查找 FFTW3、BLAS、LAPACK、ScaLAPACK 等依赖。当前机器上 FFTW3 已通过 Conda 安装，CMake 可以找到：

```text
D:/Real_Softwares/anaconda/Library/lib/fftw3.lib
```

但顶层配置后续会卡在项目自带 `FindBLAS.cmake` / `FindLAPACK.cmake` 的递归查找上。为了只验证近邻搜索核心，本次使用已有独立 runner：

```text
source/source_cell/module_neighbor/test/sltk_material_runtime_runner.cpp
```

测试体系仍采用四类材料：

| 体系 | 原子数 | 结构类型 |
| --- | ---: | --- |
| Al fcc | 1000 | 金属 |
| Si diamond | 2000 | 半导体 |
| NaCl | 3000 | 离子晶体 |
| TiO2 rutile | 4200 | 复杂氧化物 |

测试关注两件事：

1. `avg_neighbors` 是否一致，确保优化没有改变邻居结果。
2. `build_ms` 是否下降，衡量网格和邻接表构建耗时。

## 5. 测试结果

### 5.1 单线程：自适应剪枝 A/B

单线程编译不启用 OpenMP。关闭自适应剪枝时使用：

```text
-DSLTK_DISABLE_ADAPTIVE_BOX_PRUNING
```

| 体系 | baseline build_ms | adaptive build_ms | 加速比 |
| --- | ---: | ---: | ---: |
| Al 1000 | 10.4496 | 2.3172 | 4.51x |
| Si 2000 | 38.5710 | 6.5390 | 5.90x |
| NaCl 3000 | 88.4995 | 13.9946 | 6.32x |
| TiO2 4200 | 175.0070 | 24.0591 | 7.27x |

四个体系的 `avg_neighbors` 完全一致，说明自适应剪枝没有改变近邻搜索结果。

### 5.2 4 线程：自适应剪枝 + 静态调度

编译加入：

```text
-fopenmp
```

运行时设置：

```powershell
$env:OMP_NUM_THREADS='4'
```

| 体系 | baseline build_ms | adaptive build_ms | 加速比 |
| --- | ---: | ---: | ---: |
| Al 1000 | 3.4369 | 1.6454 | 2.09x |
| Si 2000 | 10.0478 | 3.0583 | 3.29x |
| NaCl 3000 | 22.8761 | 5.7071 | 4.01x |
| TiO2 4200 | 46.3795 | 9.9352 | 4.67x |

### 5.3 4 线程：guided 调度动态负载均衡

将 OpenMP 调度改为 `schedule(guided, 16)` 后，顺序运行得到：

| 体系 | baseline build_ms | adaptive build_ms | 加速比 |
| --- | ---: | ---: | ---: |
| Al 1000 | 2.6884 | 1.5770 | 1.70x |
| Si 2000 | 8.2800 | 2.6953 | 3.07x |
| NaCl 3000 | 19.2133 | 5.2650 | 3.65x |
| TiO2 4200 | 40.6054 | 8.2521 | 4.92x |

和 static 版本相比，guided 调度在四个体系上都更快。这个结果说明线程之间确实存在一定工作量差异，动态领取任务能够减少尾部等待。

## 6. 当前代码状态

| 项目书方向 | 当前状态 |
| --- | --- |
| 空间划分减少距离计算 | 已完成局部邻域搜索 |
| OpenMP 线程并行 | 已并入 `Grid::Construct_Adjacent()` |
| MPI 三维域分解 | 已在 `GridParallel` 中实现候选路径 |
| ghost atom 交换 | 已在 `GridParallel` 中实现 26 邻域交换 |
| 自适应搜索策略 | 已完成 per-box 包围盒剪枝 |
| 动态负载均衡 | 已将 OpenMP 调度改为 `guided,16` |

## 7. 结论

这次优化的核心不是改变物理判定，也不是放宽搜索半径，而是把不必要的候选检查提前过滤掉。最终是否加入邻接表仍由 `dr <= sradius2` 决定，因此结果保持一致。

从测试看，自适应剪枝在单线程下已经有明显收益；4 线程下继续有效。动态负载均衡对较大体系更明显，TiO2 体系在 4 线程下从 static adaptive 的约 `9.94 ms` 降到 guided adaptive 的约 `8.25 ms`。这说明当前近邻搜索已经不只是“能并行”，而是开始处理线程间任务不均的问题。

后续如果继续推进，建议重点放在两处：

1. 给 `GridParallel` 增加正式 MPI benchmark，验证多 rank 下邻居数和串行结果一致。
2. 把 runner 的内存统计补全到 `box_bounds`，当前表格沿用旧统计口径，主要用于时间对比。
