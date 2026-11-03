# adam-mf

| 配置名称                        | 方法                                            |
| ------------------------------- | ----------------------------------------------- |
| `adam-iid`                    | Adam + IID 高斯噪声                             |
| `momentum-iid`                | Momentum + IID 高斯噪声                         |
| `adam-mf-sgd-b2lt`            | Adam + SGD workload B²LT                       |
| `adam-mf-sgd-stp-b2lt`        | Adam + STP 缩放裁剪 + SGD workload B²LT        |
| `stpex-mf-momentum-bias-b2lt` | STP-EX + bias-corrected Momentum workload B²LT |
| `momentum-mf-momentum-b2lt`   | Momentum + Momentum workload B²LT              |

`config/experiment.yaml` 是实验参数的配置。

## 系统要求

- Linux x86_64、Bash、Conda、Python 3.11。
- 默认使用物理 GPU **0、1、2、3**，四张都必须存在。
- PyTorch `2.13.0+cu126`、Torchvision `0.28.0+cu126`；安装依赖见 `requirements.lock`。
- NVIDIA Driver 必须支持所安装 PyTorch 的 CUDA 12.6 runtime。安装后逐卡真实 CUDA 运算检查决定能否运行；不要求单独安装同版本 CUDA Toolkit。当前验收机器为 Driver 570.133.20、四张 12 GiB RTX 3080 Ti。
- 下载阶段需要访问 Toronto、Hugging Face 和软件包索引；训练进程只加载本地文件并设置离线变量。
- 为环境、安装缓存、数据、模型和实验检查点预留足够磁盘空间。120 个检查点可能远大于原始资产，运行前请用 `df -h .` 查看磁盘。

## Conda 环境安装

执行：

```bash
git clone https://github.com/3171849052/adam-mf.git
cd adam-mf
bash scripts/setup_env.sh
```

脚本创建独立的 `adam-mf` 环境，从 `environment.yml` 引用的 `requirements.lock` 安装固定依赖，安装本项目，并检查依赖、PyTorch、CUDA、四张 GPU，生成 `logs/setup_environment_report.json`。全新环境安装需要下载 CUDA/PyTorch 大包，请等待日志输出。

## GPU 设置

```yaml
runtime:
  gpu_ids: [0, 1, 2, 3]
  cv_workers_per_gpu: 1
  nlp_workers_per_gpu: 2
  max_parallel_tasks: 8
  min_free_memory_mb: {cv: 10000, nlp: 5000}
  poll_seconds: 3
  resume: true
  stop_on_error: true
  checkpoint_every_steps: 1
```

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv
watch -n 3 nvidia-smi
```

可修改 `gpu_ids` 中的物理编号并相应限制全局并发；指定的所有卡都必须存在。默认最多 8 个进程，每卡最多 1 个 CV 进程或 2 个 NLP 进程（由 `cv_workers_per_gpu` / `nlp_workers_per_gpu` 控制），两类任务不混用同一张卡，CV 和 NLP 共享队列及 GPU 资源锁。卡空闲后自动补位，空闲显存不足就等待，不改变 batch size；`stop_on_error: true` 时失败会停止派发并通知其他 worker 在完整逻辑步保存检查点。

每个 worker 设置 `CUDA_VISIBLE_DEVICES=<物理编号>` 和 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，内部只使用 `cuda:0`，并核对 CUDA 实际 UUID。例如 `CUDA_VISIBLE_DEVICES=2` 时，`cuda:0` 必须对应物理 GPU 2。任务日志、`gpu_mapping.json`、状态和逐种子结果保留物理编号、UUID、型号、逻辑设备；GPU 遥测持续记录显存。

## 数据集、模型和矩阵

```bash
./download_assets.sh
./prepare_data.sh
./verify_assets.sh
```

也可直接执行 `check.sh` 或 `start.sh`：发现缺失文件时自动下载和准备。已有完整资产时 `check.sh` 只检查；损坏文件必须通过下载入口明确修复。`verify_assets.sh` 始终只做离线验证。

| 资产           | 固定来源 / 版本                                                                                                                                                                                                           |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CIFAR-100      | [Toronto 官方 Python 归档](https://www.cs.toronto.edu/~kriz/cifar.html)，2010-02-20；备用镜像必须具有相同 SHA-256                                                                                                          |
| SST-2          | [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue/tree/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/sst2)，commit `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`                                                       |
| ViT-Tiny       | [timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/tree/7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106)，commit `7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106` |
| BERT-Tiny      | [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny/tree/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837)，commit `6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837`                                                       |
| B²LT 冻结矩阵 | 仓库自带 `resources/matrices-v1.tar.gz`；原 exp16a 验收字节，CV T=250、NLP T=310                                                                                                                                        |

## 一键启动与预检查

环境和资产准备好后：

```bash
./check.sh
./start.sh
```

`start.sh` 本身会自动执行：环境/Conda解释器验证 → 四 GPU CUDA 运算检查 → 下载和校验原始资产 → 预处理 → 全部资产 SHA-256 校验 → 含 CUDA 的算法测试 → 生成 120 个任务 → 四 GPU 统一调度 → 汇总与最终审计。

启动立即返回，准备过程也在后台。只有预检查全部通过才启动训练。无需在每次 SSH 会话激活环境。

只验证工程、不运行正式训练：

```bash
./start.sh --dry-run --wait
```

该命令使用真实四 GPU 做 CUDA 检查、运行测试、生成 120 项计划和模拟调度记录，但启动的正式训练任务数为零。模拟调度单独执行：

```bash
conda run --no-capture-output -n adam-mf python -m adam_mf.cli simulate
```

模拟不证明真实硬件通过；真实四卡结果写入环境报告。当前验收详情见 [docs/engineering_validation.md](docs/engineering_validation.md)。

## 后台运行、状态和恢复

```bash
./status.sh
tail -f logs/pipeline.log
./stop.sh
```

协调器通过新会话、脱离 stdin 和日志重定向运行，退出 SSH 后继续执行。运行锁会阻止重复启动；PID 加 `/proc` 进程身份避免向复用 PID 误发信号。worker 继承运行锁和 GPU 独占锁，协调器异常退出时仍阻止重复启动。

`stop.sh` 请求安全停止，使用 `status.sh` 等待所有任务退出。重新执行 `./start.sh` 自动恢复上次未完成的正式运行，已完成任务逐个验证后跳过。检查点只在完整 logical step 提交，保存模型、优化器、epoch、采样/加载器、Python/NumPy/CPU/CUDA RNG、噪声生成器与 B²LT 状态、指标；恢复不会再次更新已提交逻辑步。

恢复前必须匹配配置、代码、依赖环境、数据、模型、矩阵、Driver 和 GPU UUID；有检查点的任务固定回原物理卡，避免迁移到不同硬件；发现变化就停止，不静默修复旧实验。确认需要另开新实验时使用 `./start.sh --new-run`，不要删除身份检查。`--wait` 可让终端等待退出码，默认全程后台。

## 实验配置

`config/experiment.yaml` 包含十个正式种子 `20261111`–`20261120`、固定划分种子 `20261008`、六方法各任务的学习率/裁剪阈值/STP eps_scale、Adam 公共参数、隐私预算、batch、轮数和步数。CV physical batch=250，NLP=1000，logical batch=1000；均为五轮，CV=250 步、NLP=310 步。公共 Adam 参数为 beta1=0.9、beta2=0.999、eps=1e-8；单次 DP 目标 ε=8、δ=1e-5。各方法的 learning_rate、clip 和 STP eps_scale 均按任务冻结，具体值只以 YAML 为准。

改造前完整配置保存在 `docs/baseline/experiment.before.yaml`。字段差异验收命令：

```bash
conda run --no-capture-output -n adam-mf python scripts/check_config.py
```

允许移除 `adam-no-noise`、`momentum-no-noise` 及运行管理字段变化，其余实验参数必须保持一致；检查报告在 `docs/config_preservation.json`。依赖定义中 `pyproject.toml` 声明项目直接依赖，`requirements.lock` 固定相容的完整依赖闭包，`environment.yml` 引用该锁文件，不再另列不同版本。

## 结果解释

每次运行写入 `outputs/runs/<run_id>/`：

```text
config_snapshot.yaml   run_manifest.json   status.json
task_plan.json         checkpoints/       per_seed/{cv,nlp}/
formal_per_seed.csv    summary.csv        paired_differences.csv
final_report.md        figures/           logs/
audit/environment_report.json            audit/asset_hashes.json
audit/config_hash.json                   audit/code_hash.json
audit/gpu_report.json                    audit/final_audit.json
```

- `formal_per_seed.csv`：逐任务准确率、交叉熵、真实隐私预算、训练时间、峰值显存及物理 GPU 信息。正式完整结果应为 CV 60、NLP 60，共 120 条。
- `summary.csv`：每任务每方法十种子准确率均值、**样本标准差（ddof=1）**、95% Student-t 置信区间，以及交叉熵和资源统计。
- `paired_differences.csv`：相同种子方法之间的右减左差异与区间；配对前核对初始化、分类头、顺序和增强轨迹，未做多重比较校正。
- 六种 DP 方法记录各自实际 ε 和 δ。单次 ε 不能直接作为多个种子/方法发布的组合预算。
- `final_report.md` 和图形仅使用本次新训练结果。只有完整 120 项、每组恰好十个种子、资产/结果/随机轨迹审计全部通过，状态才是 `completed`。
