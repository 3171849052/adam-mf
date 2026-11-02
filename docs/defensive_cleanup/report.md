本次已完成小范围工程精简；未启动任何正式训练。导师入口仍是 `bash scripts/setup_env.sh` → `./start.sh`。

**最终 GPU 配置（按用户后续要求）**

```yaml
runtime:
  gpu_ids: [0, 1, 2, 3]
  cv_workers_per_gpu: 1
  nlp_workers_per_gpu: 2
  max_parallel_tasks: 4
  min_free_memory_mb: {cv: 10000, nlp: 5000}
```

每卡可容纳 1 个 CV 或最多 2 个 NLP 进程，两类任务不混卡。两个容量字段现在有独立调度作用，保留在配置中；删除 `max_tasks_per_gpu`。全局上限仍为 4，因此不代表默认同时运行 8 个 NLP 进程；需要四卡各两个时，可显式把 `max_parallel_tasks` 配为 8。GPU UUID、显存门槛、缺卡拒绝和检查点原卡绑定均保留。父进程在同卡任务存续期间持有同一个 GPU 排他锁，并将其传给各 Worker；其他运行不能抢占该卡。

**修改前的问题与逐文件修改清单**

| 文件 | 实际修改 |
|---|---|
| `src/adam_mf/audit.py` | 移除 `tests/` 对训练代码身份的影响；增加仅在一次 preflight 中有效的 SHA-256/内容验证缓存，退出后丢弃。缓存以路径、设备、inode、大小、mtime、ctime 为键，最近 1 秒写过的文件不复用哈希，避免快速同大小写入的时间戳粒度问题。 |
| `src/adam_mf/runtime/state.py` | 完整身份可消费已经完整验证的资产哈希；新增 Worker 轻量验证：父进程传入的身份摘要、快照配置、任务归属、物理 UUID、必要资产存在性。完整恢复校验保持严格比较。 |
| `src/adam_mf/runtime/worker.py` | 使用父进程已验证的身份，不再读取整份数据、模型和源码计算哈希。训练引擎仍检查 CUDA 实际 UUID。 |
| `src/adam_mf/runtime/launcher.py` | 完整身份统一由协调进程 preflight 检查一次，返回并沿用；不在 launch、preflight 后和 dry-run 分支重复检查。新建运行复用代码哈希结果。测试源码哈希单独写入 `audit/test_hashes.json`，不阻断训练恢复。普通错误只捕获 Exception；KeyboardInterrupt/SystemExit 标记暂停后重新抛出；PipelineStopped 仍返回 75，锁在 finally 释放。 |
| `src/adam_mf/runtime/scheduler.py` | 容量来自配置；去掉未参与判定的 usage 参数，但保留显存遥测。支持同卡 NLP 双进程及共享继承锁；启动失败日志和调度资源通过 finally 释放。身份摘要经环境传给 Worker。 |
| `src/adam_mf/config.py` | 保留并校验两类每卡进程数为正整数，删除冗余总每卡容量字段，允许配置全局并发超过 GPU 数。 |
| `config/experiment.yaml` | 仅将 NLP 每卡进程数 1→2，并删除 max_tasks_per_gpu；其余值不变。 |
| `src/adam_mf/optimizers/finite.py`（新增） | 单设备张量的有限值标量在设备上合并，每个检查阶段只同步一个布尔值。 |
| `src/adam_mf/optimizers/adam.py` | 合并更新前梯度和更新后状态检查；更新后也覆盖一阶矩。计算、参数更新、噪声和范数归约顺序不变。 |
| `src/adam_mf/optimizers/momentum.py` | 合并梯度和参数检查，并在更新后检查一阶矩。公式与状态更新顺序不变。 |
| `src/adam_mf/optimizers/stpex.py` | 移除 LogicalBatch 与 Optimizer 重复的梯度检查；所有参数的更新前有限值标量一次同步，仍先验证全部 pending 更新，再提交任何状态或参数。更新后参数/一阶矩/二阶矩合并检查。中间公式与归约保持原样。 |
| `src/adam_mf/training/engine.py` | BaseException 改为 Exception；StopRequested 原路径保留，SIGINT/SIGTERM 仍在完整逻辑步保存检查点后退出，系统退出异常直接传播，hooks 仍在 finally 释放。 |
| `src/adam_mf/assets.py` | pinned 原始文件通过 matches 校验后直接使用已确认的 SHA-256，取消紧接着的第二次读取；下载阶段只处理原始资产，缓存验证交给预处理阶段。 |
| `src/adam_mf/data/preprocessing.py` | 同一验证会话按文件哈希及预期内容身份复用预处理内容校验；不再把所有 RuntimeError 当缓存损坏。仅已识别的 ZIP/EOF 读取错误转换为无效缓存，其他 RuntimeError 传播。预处理协议不变。 |
| 根目录 `check.sh`、`download_assets.sh`、`prepare_data.sh`、`verify_assets.sh` | 改为转发到 scripts 中对应实现，参数原样传递。 |
| `scripts/check.sh`、`scripts/download_assets.sh`、`scripts/prepare_data.sh`、`scripts/verify_assets.sh` | 保存实际解释器选择及 CLI 调用；与根目录不再反向转发。 |
| `README.md` | 更新配置示例及每卡容量说明。 |
| `tests/test_runtime.py` | 更新新调度接口，检查同卡 NLP 容量、禁止混卡与显存门槛。 |
| `tests/test_engineering.py` | 混合 160 任务模拟按任务族容量验收；更新 Worker 校验入口测试。 |
| `tests/test_cleanup.py`（新增） | 增加哈希复用/变更失效、预处理内容复用、Worker 身份边界、恢复完整校验、STP-EX 原子拒绝、NLP 共享锁、配置容量、系统退出与锁释放测试。 |
| `docs/defensive_cleanup/*`（新增） | 修改前配置快照、本报告、原始测量 JSON、三个验证脚本。 |

`training/logical_batch.py` 在本仓库不存在；对应职责在三个优化器模块中的 LogicalBatch。所有梯度范数的 FP64 转换、stack 与归约保留，未为了微小分配收益改变浮点顺序。裁剪、隐私预算、模型、数据采样与噪声模块没有修改。

**验证结果**

- 使用现有 `adam-mf` Conda 环境，不安装或升级依赖。
- `ADAM_MF_TEST_CUDA=1 PYTHONPATH=src python -m pytest tests -q`：**114 passed，39 warnings，114.51 秒**。最后仅修改调度日志 finally 和测试哈希审计后，相关三组回归：**26 passed，6.56 秒**。
- 修改前保存到 `/tmp/adam-mf-before` 的源码运行恢复测试：9 passed。修改前后八种算法合成引擎检查点比较：**136 个张量/数组逐位相同**；包含模型、优化器、Python/NumPy/Torch RNG、sampler、噪声状态。日志中的梯度范数、裁剪比例、噪声边际标准差、隐私值和指标一致；仅忽略墙钟/训练/步耗时字段。原始逐样本梯度和噪声张量没有额外导出；其公式未改，现有裁剪、噪声与 CUDA 测试通过。
- 八种方法均通过真实引擎的合成数据“第 3 步保存→中断→恢复→完成”测试；模型、指标、顺序及诊断与不中断运行完全一致；错误身份拒绝恢复。
- 配置快照 `experiment.before.yaml` 与当前配置比较：除 NLP 每卡 1→2、删除 max_tasks_per_gpu 外，全部相同；算法、超参数、随机种子、physical/logical batch、训练步数及隐私参数均未改。
- 任务生成：8 methods × 2 tasks × 10 seeds；CV 80、NLP 80、合计 **160**。没有启动正式训练。
- 真实四张 RTX 3080 Ti：GPU 0/1/2/3 均完成独立子进程 CUDA 运算及实际 UUID 对照。调度测试使用模拟 Worker 验证全局上限、补位、失败停止、同卡 NLP 双进程和锁释放；没有进行真实双 NLP 模型的并发训练压力测试。
- 完整启动资产链路 `fetch_assets → prepare_data → identity` 实际验证：35 个资产/清单哈希；现有不可变资产每份完整 SHA-256 **读取一次**。随后独立 `./verify_assets.sh` 重新完整验证 35 个哈希通过。新写入文件与被重新发布的清单保守重读，不承诺这些文件也只读一次。
- Shell 入口语法检查及 `git diff --check` 通过。

**性能测量的范围与结果**

以下为 GPU 0 上的合成逻辑批次微基准：16 个 Linear(64,64) 层、32 个参数张量、固定梯度、CV 原有累积次数，5 步预热后测 30 步中位数。未运行真实数据前向/反向，也未生成噪声；带噪声方法名称仅表示对应优化器/LogicalBatch 路径。不能把这些数字解释成正式 CV/NLP 单步耗时。

| 方法 | 修改前 ms | 修改后 ms | profiler aten::item 前→后 | 峰值分配 bytes 前→后 |
|---|---:|---:|---:|---:|
| adam-no-noise | 10.154 | 12.347 | 161→67 | 1892352→1892352 |
| momentum-no-noise | 9.670 | 12.225 | 65→3 | 1432576→1432576 |
| adam-iid | 10.195 | 12.395 | 161→67 | 1892352→1892352 |
| momentum-iid | 9.651 | 7.887 | 65→3 | 1432576→1432576 |
| adam-mf-sgd-b2lt | 10.218 | 8.034 | 161→67 | 1892352→1892352 |
| adam-mf-sgd-stp-b2lt | 12.397 | 9.713 | 193→99 | 2433024→2433024 |
| stpex-mf-momentum-bias-b2lt | 24.253 | 22.318 | 265→11 | 3086336→3165696 |
| momentum-mf-momentum-b2lt | 7.184 | 7.933 | 65→3 | 1432576→1432576 |

`aten::item` 是调用数代理，包含优化器内部及 CPU 诊断调用，不能直接等同于 CUDA 同步事件总数。有限值检查本身由逐参数同步改为每阶段一次；Adam 自身步计数等内部同步未改。STP-EX 因暂存检查标量，合成峰值分配增加 79,360 bytes。测试存在运行时间波动，多项方法变慢，不能声称所有方法或正式训练整体提速。

完整身份检查：1.058 秒；Worker 轻量身份检查 5 次为 0.059、0.058、0.063、0.056、0.060 秒。包含新 Python 进程、身份校验和训练模块导入，但止于模型加载之前的启动微基准，修改前 3 次中位数 **9.437 秒**，修改后 **11.707 秒**。这项实测未显示整体启动加速；仅能确认完整身份计算和磁盘读取被移出 Worker。完整模型/数据加载后的 Worker 就绪耗时未测量。

**边界与尚未验证项**

1. 未执行正式 160 次训练或真实 CV/NLP 完整模型单步性能测试，不能给出正式训练加速比例或真实双 NLP 进程的显存结论。
2. 本次会修改训练代码身份及运行配置。旧版本检查点仍会被严格拒绝，不提供绕过身份的迁移；本版本内部的恢复轨迹通过测试。
3. 哈希缓存仅在单次 preflight 生效，不跨进程/运行持久化；显式 verify 和独立实验都重新完整读取。文件时间戳/大小/inode 改变就失效，近期写入禁用哈希复用。正式运行要求输入资产在启动校验后保持不变，Worker 不再做全数据哈希扫描。
4. 合并有限值检查适用于正式 Worker 的单设备模型，未增加跨设备优化器框架。独立调用原生 Adam/Momentum 不增加额外包装；正常训练由 LogicalBatch 负责阶段检查，STP-EX Optimizer 自身负责提交前安全性。
5. 39 条测试 warning 未当作错误屏蔽；包括现有梯度 hook、pin_memory 等提示。首次新增哈希回归发现快速同大小写入可能共享时间戳，修复后全套通过。

原始测量见 `measurements.json`。`benchmark_logical_step.py` 可在不同 PYTHONPATH 源码快照下运行；其余两个脚本保留本次 `/tmp/adam-mf-*` 输入位置，用于重现本次会话。基线 Git commit 记录在 JSON 中；源码快照是修改前当前工作树，配置快照永久保存在本目录。
