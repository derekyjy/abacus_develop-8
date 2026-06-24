# ABACUS module_neighbor MPI并行化与动态负载均衡实现总结

## 1. 概述

本文档总结了在 `module_neighbor`（旧近邻搜索模块）上实现的 MPI 并行化接入与三层动态负载均衡优化。

### 修改范围

| 文件 | 改动类型 | 行数变化 |
|------|----------|----------|
| `source/source_cell/module_neighbor/sltk_grid_parallel.h` | 修改 | +4 |
| `source/source_cell/module_neighbor/sltk_grid_parallel.cpp` | 修改 | +303 / -99 |
| `source/source_cell/module_neighbor/sltk_atom_arrange.cpp` | 修改 | +28 |
| `source/source_cell/module_neighbor/test/CMakeLists.txt` | 修改 | +23 |

---

## 2. 架构：三层动态负载均衡

```
                    ┌─────────────────────────────────┐
                    │   Construct_Adjacent_parallel()   │
                    └─────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
  ┌───────────────┐      ┌───────────────────┐      ┌──────────────────┐
  │ MPI rank 间    │      │  rank 内线程间     │      │  每原子成本预估   │
  │ 负载均衡       │      │  负载均衡          │      │                  │
  ├───────────────┤      ├───────────────────┤      ├──────────────────┤
  │ 按累计原子数    │      │ workload降序排序   │      │ 统计邻域box内     │
  │ 分配x-domain   │      │ + static,1交错    │      │ 候选原子总数      │
  ├───────────────┤      ├───────────────────┤      ├──────────────────┤
  │ rank_domain_   │      │ LocalAtomTask      │      │ estimate_atom_   │
  │ bounds_        │      │ struct + std::sort │      │ workload()       │
  │ balanced()     │      │ + schedule(static,1)│    │                  │
  └───────────────┘      └───────────────────┘      └──────────────────┘
```

### 2.1 第一层：MPI rank 间负载均衡

**问题**：原有 `rank_domain_bounds` 按等量 box 数分配域，稠密区 rank 过载，稀疏区 rank 空闲。

**方案**：`rank_domain_bounds_balanced` — 按累计原子数分配 x 方向域边界。

```
原有（等量box）:
  rank 0: boxes [0, 6)  → 27,500 原子
  rank 1: boxes [6, 12) → 22,500 原子
  → 负载比 1.22x

新增（按原子数）:
  rank 0: boxes [0, 5)  → 24,800 原子
  rank 1: boxes [5, 8)  → 25,200 原子
  → 负载比 1.02x
```

实现位于 `sltk_grid_parallel.cpp:365-435`。

### 2.2 第二层：rank 内线程间负载均衡

**问题**：原有 `schedule(static)` 将连续块分配给各线程，耗时原子（在稠密区）集中在少数线程，造成尾部等待。

**方案**：
1. `LocalAtomTask` 结构体包含 `{i_type, j_atom, workload}`
2. 按 `workload` 降序排序（重原子在前，轻原子在后）
3. `#pragma omp for schedule(static, 1)` — 交错分配，每线程交替获取重/轻原子

```
排序后: [H0, H1, H2, H3, H4, H5, ..., L0, L1, L2, ...]
static,1:
  Thread 0: H0, H4, H8, ..., L0, L4, ...    ← 混合重/轻
  Thread 1: H1, H5, H9, ..., L1, L5, ...
  ...
```

实现位于 `sltk_grid_parallel.cpp:538-598`。

### 2.3 第三层：每原子成本预估

`estimate_atom_workload(const FAtom& atom)` — 统计当前原子 search_span 范围内所有 box 中的候选原子总数，作为计算成本的近似估计。

```cpp
int GridParallel::estimate_atom_workload(const FAtom& atom) const
{
    // 定位原子所在box
    getBox(bx, by, bz, atom.x, atom.y, atom.z);
    // 确定search_span范围
    const int search_span = ceil(sradius / box_edge_length);
    // 累加邻域box内原子总数
    for (ix in [bx - span, bx + span])
        for (iy in [by - span, by + span])
            for (iz in [bz - span, bz + span])
                count += atoms_in_box[ix][iy][iz].size();
    return count;
}
```

实现位于 `sltk_grid_parallel.cpp:437-471`。

---

## 3. MPI 广播改造

### 3.1 问题

原有 `Construct_Adjacent_parallel` 通过 `MPI_Gather`/`MPI_Gatherv` 将结果收集到 rank 0，其他 rank 无完整近邻表。

### 3.2 修改

将全部 `MPI_Gather`/`MPI_Gatherv` 改为 `MPI_Allgather`/`MPI_Allgatherv`，使所有 rank 都能获得完整的近邻搜索结果。

| 修改 | 原调用 | 新调用 |
|------|--------|--------|
| entry 总数 | `MPI_Reduce` (→rank 0) | `MPI_Allreduce` (→all) |
| 每 rank entry 计数 | `MPI_Gather` (→rank 0) | `MPI_Allgather` (→all) |
| entry 数据 | `MPI_Gatherv` (→rank 0) | `MPI_Allgatherv` (→all) |
| 每 rank 原子计数 | `MPI_Gather` (→rank 0) | `MPI_Allgather` (→all) |
| 原子数据 | `MPI_Gatherv` (→rank 0) | `MPI_Allgatherv` (→all) |

同时新增 `atoms_in_box` 保存/恢复机制：并行搜索阶段修改局部 grid 后，Allgather 完成时恢复完整 grid 供所有 rank 反序列化。

---

## 4. 生产路径接入

### 4.1 调用链

```
esolver_lrtd_lcao.cpp
  → atom_arrange::search()
    → grid_d.init()                          // 设置全部标量成员
    → [MPI可用时] GridParallel 构建
      → GridParallel::init()                 // 准备网格
      → Construct_Adjacent_parallel()        // MPI分布式近邻搜索
      → move all_adj_info/atoms_in_box → grid_d  // 传递结果
    → grid_d.Find_atom()                     // 下游正常查询
```

### 4.2 关键设计

- `grid_d.init()` 先执行以设置全部 Grid 标量成员（`glayerX/Y/Z`、`box_nx/ny/nz`、`sradius` 等）
- MPI 路径的 `Construct_Adjacent_parallel` 仅覆盖数组成员（`all_adj_info`、`atoms_in_box`、`box_bounds`）
- 通过 `std::move` 转移向量所有权，`FAtom*` 指针跟随 `atoms_in_box` 一起移动，保持有效性
- `#ifdef __MPI` + `MPI_Initialized()` 双重保护，单进程时自动回退到串行路径

实现位于 `sltk_atom_arrange.cpp:95-117`。

---

## 5. Benchmark 验证结果

### 5.1 测试场景

- 50,000 原子，10% 在稠密簇（0.05 box），90% 均匀分布
- search_radius = 0.35，grid 12×12×12 boxes
- 4 OpenMP 线程/rank，所有 rank 沿 x 轴 1D 分解

### 5.2 结果

| ranks | 指标 | 等量Box（无均衡） | 按原子数（动态均衡） | 提升 |
|-------|------|-------------------|---------------------|------|
| 4 | 耗时 | 49.6ms | 45.6ms | **8.2%** |
|   | 原子/rank (min–max) | 9,617–16,746 | 9,617–15,854 | |
|   | 不平衡比 | 1.7x | 1.6x | |
| 8 | 耗时 | 52.3ms | 45.7ms | **12.6%** |
|   | 原子/rank (min–max) | 1,639–12,711 | 3,883–8,828 | |
|   | 不平衡比 | **7.8x** | **2.3x** | |
| 16 | 耗时 | 54.5ms | 53.7ms | 1.5% |
|   | 原子/rank (min–max) | 0–8,828 | 1,639–8,828 | |

### 5.3 结论

- **8 rank 为最优均衡点**：动态均衡将负载比从 7.8x 降到 2.3x，带来 12.6% 端到端加速
- **16 rank 避免空域崩溃**：等量box分配导致部分rank分到0个原子，动态均衡保证每个rank至少有原子可处理
- **4 rank 场景均衡有意义但有限**：1.7x→1.6x，8.2%提升
- **近邻总数保持正确**：所有场景下结果一致

---

## 6. 使用方式

### 编译

MPI 路径通过 `__MPI` 宏条件编译，CMake 配置 `-DENABLE_MPI=ON` 时自动启用。

### 运行

无需额外配置。`atom_arrange::search()` 在检测到 MPI 已初始化时自动使用 `GridParallel` 路径：

```cpp
// 用户代码无需修改，调用方式不变
atom_arrange::search(pbc_flag, ofs, grid_d, ucell, radius, test_atom_in);
```

单进程运行时（`mpiexec -n 1`），`MPI_Initialized` 返回 0，自动回退到原有串行 `Grid::Construct_Adjacent` 路径。

### Benchmark

```bash
# 独立 benchmark（需 MS-MPI）
cd source/source_cell/module_neighbor/benchmark
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
mpiexec -n 4 ./Release/benchmark_parallel.exe
```

### 单元测试

```bash
# 包含 MPI 时编译 sltk_material_runtime_runner
cmake -DBUILD_TESTING=ON -DENABLE_MPI=ON ..
cmake --build . --config Release
mpiexec -n 4 ./tests/sltk_material_runtime_runner.exe
```

---

## 7. 技术要点

| 要点 | 说明 |
|------|------|
| **三层负载均衡** | MPI rank → OpenMP 线程 → 单原子成本预估，逐级细化 |
| **零拷贝数据转移** | `std::move` 转移 `all_adj_info`/`atoms_in_box`，`FAtom*` 指针保持有效 |
| **向后兼容** | `#ifdef __MPI` + `MPI_Initialized` 双重检查，串行路径不受影响 |
| **正确性保证** | 所有 rank 收到完整 Allgather 结果，`Find_atom()` 查询结果一致 |
| **grid 状态恢复** | 并行搜索修改局部 grid 后通过 swap 恢复完整状态 |
