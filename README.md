# adam-mf

## 八种算法

| 配置中的方法 ID             | 算法                         | 裁剪 / 噪声                               |
| --------------------------- | ---------------------------- | ----------------------------------------- |
| adam-no-noise               | Adam-No-Noise                | 无裁剪、无噪声、无 DP 包装                |
| momentum-no-noise           | Momentum-No-Noise            | 无裁剪、无噪声、无 DP 包装                |
| adam-iid                    | Adam-IID                     | 原空间 Ghost clipping / IID               |
| momentum-iid                | Momentum-IID                 | 原空间 Ghost clipping / IID               |
| adam-mf-sgd-b2lt            | Adam + SGD-B²LT             | 原空间 Ghost clipping / SGD workload      |
| adam-mf-sgd-stp-b2lt        | Adam + STP + SGD-B²LT       | 缩放空间裁剪加噪，撤销缩放，再做标准 Adam |
| stpex-mf-momentum-bias-b2lt | STP-EX + Momentum-Bias-B²LT | 缩放空间一阶矩更新；无末尾逆缩放          |
| momentum-mf-momentum-b2lt   | Momentum + Momentum-B²LT    | 原空间 Ghost clipping / Momentum workload |

Momentum 为 `m=0.9*m+(1-0.9)*g; p+=-lr*m`，没有一阶偏差修正。Adam 有两阶偏差修正。STP 缩放只使用上一已完成步的二阶矩。`eps_scale` 是算法内部缩放参数，与隐私 ε 不同。

## 任务与总配置

正式配置为 [config/experiment.yaml](config/experiment.yaml)。公共训练参数、两任务启用开关、八方法启用开关、各任务独立 LR/C/eps_scale、种子、隐私、路径与 GPU 调度均在此。未知字段、缺失参数、非法方法名、非有限值、无法对应冻结矩阵的训练长度会立即报错。

- CV：完整 CIFAR-100 train 50000，官方 test 10000；物理 batch 250，逻辑 batch 1000，5轮、250步。
- NLP：固定分层划分的 SST-2 62000 条训练样本，官方 validation 872；物理/逻辑 batch 1000，5轮、310步；长度128。
- 全参数 FP32；beta1=.9、beta2=.999、Adam eps=1e-8、weight_decay=0；无调度器。
- 划分 seed=20261008；正式 seeds=20261111–20261120；各 epoch 复用相同样本顺序，增强随机状态逐轮推进。
- 方法参数来自原实验冻结结果

默认 GPU 自动检测，CV 每卡1进程、NLP 每卡最多2进程。

## 环境安装

Python 3.11；Linux、NVIDIA 驱动、CUDA PyTorch、`nvidia-smi`。本机已使用现有 `curve` 环境验证，入口会优先选择当前满足依赖的 Python，也可显式指定解释器：

```bash
conda env create -f environment.yml
conda activate adam-mf
pip install -e . --no-deps
# 可选：export ADAM_MF_PYTHON=/absolute/path/to/python
```

environment.yml 固定本次实测依赖（PyTorch 2.13.0+cu126、Opacus 1.6.0、timm 1.0.29、Transformers 5.15.0 等）。

## 启动、状态、停止和恢复

```bash
./start.sh                 # 后台运行完整正式实验
./status.sh                # run_id、PID、完成数、运行任务、GPU、逻辑步与日志
 tail -f logs/pipeline.log
./stop.sh                  # 请求在逻辑步边界保存后退出
./start.sh                 # 恢复最近未完成的正式运行
```

启动后关闭终端不影响协调器；唯一 run_id、PID 身份、继承文件锁防止重复启动。日志为 logs/pipeline.log 和各 run 的 logs/。不要删除正在使用的 pipeline.lock。已完成任务必须通过身份及产物哈希校验才复用。所有任务完成后再次启动会创建新实验；需要主动新建可使用 `./start.sh --new-run`。

启动顺序：检查环境/GPU、资产哈希、配置，运行算法/数据/恢复等单元测试，生成正式任务队列，再调度 GPU。单元测试仅使用小型合成训练输入和独立资产验证，不额外训练正式模型。

## 隐私预算

六种 DP 方法每次完整训练 ε=8、δ=1e-5，add/remove zero-out 邻接、固定五轮参与、原始 GDP 矩阵校准、不使用采样放大。
