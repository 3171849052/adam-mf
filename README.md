# adam-mf：四 GPU 固定参数十种子实验

adam-mf 比较普通优化器、独立高斯噪声和 B²LT 相关噪声在全参数微调中的表现。正式任务为 CIFAR-100 / ViT-Tiny 和 SST-2 / BERT-Tiny，各使用同一组十个随机种子：**8 种方法 × 2 个任务 × 10 个种子 = 160 次训练**。

八种方法为：

|配置名称|方法|
|---|---|
|`adam-no-noise`|Adam，无裁剪、无噪声|
|`momentum-no-noise`|Momentum，无裁剪、无噪声|
|`adam-iid`|Adam + IID 高斯噪声|
|`momentum-iid`|Momentum + IID 高斯噪声|
|`adam-mf-sgd-b2lt`|Adam + SGD workload B²LT|
|`adam-mf-sgd-stp-b2lt`|Adam + STP 缩放裁剪 + SGD workload B²LT|
|`stpex-mf-momentum-bias-b2lt`|STP-EX + bias-corrected Momentum workload B²LT|
|`momentum-mf-momentum-b2lt`|Momentum + Momentum workload B²LT|

`config/experiment.yaml` 是实验参数的唯一权威配置。不需要重新搜索参数；本项目不提供搜索或自动选参。历史 exp15c / exp16e 准确率不会导入新结果。

## 系统要求

- Linux x86_64、Bash、Conda、Python 3.11。
- 默认使用物理 GPU **0、1、2、3**，四张都必须存在；不自动降级或使用其他 GPU。
- PyTorch `2.13.0+cu126`、Torchvision `0.28.0+cu126`；安装依赖见 `requirements.lock`。其他核心依赖维持原版本。
- NVIDIA Driver 必须支持所安装 PyTorch 的 CUDA 12.6 runtime。安装后逐卡真实 CUDA 运算检查决定能否运行；不要求单独安装同版本 CUDA Toolkit。当前验收机器为 Driver 570.133.20、四张 12 GiB RTX 3080 Ti。
- 下载阶段需要访问 Toronto、Hugging Face 和软件包索引；训练进程只加载本地文件并设置离线变量。
- 为环境、安装缓存、数据、模型和实验检查点预留足够磁盘空间。160 个检查点可能远大于原始资产，运行前请用 `df -h .` 查看磁盘。

## Conda 环境安装

导师在新服务器执行：

```bash
git clone https://github.com/3171849052/adam-mf.git
cd adam-mf
bash scripts/setup_env.sh
```

脚本创建独立的 `adam-mf` 环境，从 `environment.yml` 引用的 `requirements.lock` 安装固定依赖，安装本项目，并检查依赖、PyTorch、CUDA、四张 GPU，生成 `logs/setup_environment_report.json`。已存在的环境只检查版本并安装项目，不覆盖、升级依赖；版本不符会明确失败。全新环境安装需要下载 CUDA/PyTorch 大包，请等待日志输出。

开发者继续使用自己的 `curve`，不重命名、不修改依赖：

```bash
conda activate curve
export ADAM_MF_PYTHON="$(which python)"
./check.sh
```

所有入口共享解释器选择规则：设置 `ADAM_MF_PYTHON` 时使用该绝对路径；否则只使用 Conda 的 `adam-mf`。找不到时提示先运行安装脚本，不自动选 curve、base 或系统 Python。切回导师环境可执行 `unset ADAM_MF_PYTHON`。已有开发环境中其他项目的 `pip check` 冲突会记录在报告中，本项目核心依赖版本不符则直接停止。

## GPU 设置

```yaml
runtime:
  gpu_ids: [0, 1, 2, 3]
  cv_workers_per_gpu: 1
  nlp_workers_per_gpu: 1
  max_parallel_tasks: 4
  max_tasks_per_gpu: 1
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

可修改 `gpu_ids` 中的物理编号并相应限制全局并发；指定的所有卡都必须存在。默认最多四个进程，每卡同时最多一个任务，CV 和 NLP 共享队列及 GPU 资源锁。卡空闲后自动补位，空闲显存不足就等待，不改变 batch size；`stop_on_error: true` 时失败会停止派发并通知其他 worker 在完整逻辑步保存检查点。

每个 worker 设置 `CUDA_VISIBLE_DEVICES=<物理编号>` 和 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，内部只使用 `cuda:0`，并核对 CUDA 实际 UUID。例如 `CUDA_VISIBLE_DEVICES=2` 时，`cuda:0` 必须对应物理 GPU 2。任务日志、`gpu_mapping.json`、状态和逐种子结果保留物理编号、UUID、型号、逻辑设备；GPU 遥测持续记录显存。

## 数据集、模型和矩阵

```bash
./download_assets.sh
./prepare_data.sh
./verify_assets.sh
```

也可直接执行 `check.sh` 或 `start.sh`：发现缺失文件时自动下载和准备。已有完整资产时 `check.sh` 只检查；损坏文件必须通过下载入口明确修复。`verify_assets.sh` 始终只做离线验证。

|资产|固定来源 / 版本|
|---|---|
|CIFAR-100|[Toronto 官方 Python 归档](https://www.cs.toronto.edu/~kriz/cifar.html)，2010-02-20；备用镜像必须具有相同 SHA-256|
|SST-2|[nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue/tree/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/sst2)，commit `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`|
|ViT-Tiny|[timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/tree/7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106)，commit `7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106`|
|BERT-Tiny|[prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny/tree/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837)，commit `6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837`|
|B²LT 冻结矩阵|仓库自带 `resources/matrices-v1.tar.gz`；原 exp16a 验收字节，CV T=250、NLP T=310|

矩阵无已核实的公开下载地址，因此直接随仓库提供约 759 KB 的固定分发包，不依赖兄弟仓库，不重新优化。BERT 原本本地生成的 tokenizer 文件也随代码保存。详细结构、分类头初始化和来源说明见 [docs/assets.md](docs/assets.md)。

```text
data/cifar100/cifar-100-python/{train,test,meta}
data/sst2/{train,validation}.parquet
data/preprocessed/                 # 固定划分和长度128的token缓存
data/manifest.json
model/{vit-tiny,bert-tiny}/
model/manifest.json
assets/matrices/                   # 冻结npz、metadata及manifest.json
```

下载支持 Range 续传、失败重试、大小和 SHA-256 校验、完整文件跳过；临时文件为 `.part`，通过校验才原子发布。解包逐个校验指定普通文件，不直接解压任意路径。`prepare_data.sh` 从原始数据独立重建固定划分和 tokenizer 缓存，并与原实验逐张量 SHA-256、dtype、shape 对照。CV 正式训练仍为 50000 张、评价 10000 张；NLP 固定训练 62000 条、官方验证 872 条。不得改成别的 SST-2 分割。

可信版本与 SHA-256 在 `config/assets.lock.json`，生成的 manifest 记录实际资产字节。`verify_assets.sh` 离线核对两者；修改生成清单不能绕过校验。下载失败、模型缺失、矩阵或数据哈希错误都会阻止正式训练。

## 一键启动与预检查

环境和资产准备好后：

```bash
./check.sh
./start.sh
```

`start.sh` 本身会自动执行：环境/Conda解释器验证 → 四 GPU CUDA 运算检查 → 下载和校验原始资产 → 预处理 → 全部资产 SHA-256 校验 → 含 CUDA 的算法测试 → 生成 160 个任务 → 四 GPU 统一调度 → 汇总与最终审计。

启动立即返回，准备过程也在后台。只有预检查全部通过才启动训练。无需在每次 SSH 会话激活环境。

只验证工程、不运行正式训练：

```bash
./start.sh --dry-run --wait
```

该命令使用真实四 GPU 做 CUDA 检查、运行测试、生成 160 项计划和模拟调度记录，但启动的正式训练任务数为零。模拟调度单独执行：

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

`config/experiment.yaml` 包含十个正式种子 `20261111`–`20261120`、固定划分种子 `20261008`、八方法各任务的学习率/裁剪阈值/STP eps_scale、Adam 公共参数、隐私预算、batch、轮数和步数。CV physical batch=250，NLP=1000，logical batch=1000；均为五轮，CV=250 步、NLP=310 步。公共 Adam 参数为 beta1=0.9、beta2=0.999、eps=1e-8；单次 DP 目标 ε=8、δ=1e-5。各方法的 learning_rate、clip 和 STP eps_scale 均按任务冻结，具体值只以 YAML 为准。不得为了显存或测试修改这些实验参数。

改造前完整配置保存在 `docs/baseline/experiment.before.yaml`。字段差异验收命令：

```bash
conda run --no-capture-output -n adam-mf python scripts/check_config.py
```

只允许运行管理字段变化，检查报告在 `docs/config_preservation.json`。依赖定义中 `pyproject.toml` 声明项目直接依赖，`requirements.lock` 固定相容的完整依赖闭包，`environment.yml` 引用该锁文件，不再另列不同版本。

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

- `formal_per_seed.csv`：逐任务准确率、交叉熵、真实隐私预算、训练时间、峰值显存及物理 GPU 信息。正式完整结果应为 CV 80、NLP 80，共 160 条。
- `summary.csv`：每任务每方法十种子准确率均值、**样本标准差（ddof=1）**、95% Student-t 置信区间，以及交叉熵和资源统计。
- `paired_differences.csv`：相同种子方法之间的右减左差异与区间；配对前核对初始化、分类头、顺序和增强轨迹，未做多重比较校正。
- 六种 DP 方法记录各自实际 ε 和 δ；无噪声方法留空，表示“不适用”，绝不能写 ε=0。单次 ε 不能直接作为多个种子/方法发布的组合预算。
- `final_report.md` 和图形仅使用本次新训练结果。只有完整 160 项、每组恰好十个种子、资产/结果/随机轨迹审计全部通过，状态才是 `completed`。

## 常见问题

|问题|处理|
|---|---|
|Conda 创建失败|查看安装输出，检查索引网络、磁盘和 `requirements.lock`；不要擅自升级核心依赖。已存在但不完整的环境会报版本缺失，请检查后由操作者修复。|
|CUDA 不可用|核对 NVIDIA Driver 与 CUDA 12.6 PyTorch 包，查看 `logs/setup_environment_report.json` 和 `nvidia-smi`。|
|GPU 数量不足|补齐配置中的物理 GPU，或明确修改运行资源配置；程序不会自动降级。|
|下载中断|重复执行 `./download_assets.sh`，有效 `.part` 会续传；正确完整文件跳过。|
|模型哈希错误|新运行可重新下载；恢复旧运行会拒绝资产变化，请先调查文件来源。不要修改可信哈希来绕过校验。|
|磁盘空间不足|用 `df -h .` 检查；清理不再需要的缓存/旧运行，保留需要恢复的完整检查点和资产。|
|训练 OOM|查看任务日志，关闭其他 GPU 负载或提高运行管理中的显存门槛；不能缩 batch 或改算法参数。|
|恢复失败|查看身份不一致错误，对照 config/code/assets/environment/GPU UUID；恢复原身份或明确另开新运行。|
|四张 GPU 未全部使用|检查剩余任务数、各卡空闲显存和 GPU 资源锁；显存不足会等待。查看 `logs/gpu_telemetry.jsonl`（位于对应运行目录）。|
|重复启动被拒绝|使用 `./status.sh` 确认旧协调器和 worker 是否仍运行；不要删除正在使用的锁文件。|
