# 第三轮工程验收（2026-11-02）

本轮只修改部署、资产管理、GPU 调度、审计和测试，未启动任何正式训练。当前配置来源是修改前的 YAML，未采用旧报告中的参数替换当前参数。

## 参数保全

原配置备份：`docs/baseline/experiment.before.yaml`，SHA-256：`d25df31beebcb022b7c74ae88c2e971dbacd828be6433abac7028a61c62dc9f1`。

修改后的配置 SHA-256：`4de4623c6e7f6756a86f0e6c2ed6bc082c2577f3adf98a61ebfe7554e3d909bf`。字段比较仅允许 runtime / paths 变化；实际 YAML 差异只有 GPU 编号、NLP 并发、全局并发和新增每卡上限。全部原实验参数的 JSON SHA-256 保持为 `1ed31e464c6cfbe1102489efcf684a407e710ed1063899bf24e2160f4ccbd564`。详见 `config_preservation.json`。

优化器、裁剪、噪声、隐私校准和模型初始化实现没有改动。训练引擎增加 GPU 审计及恢复身份约束，未改动训练更新顺序。旧测试中的过期冻结值改为核对本轮开始时的 YAML。27 个原仓库重点文件与已有来源哈希一致，16 个矩阵文件与 exp16a 原验收哈希一致。

## 已完成的验证

|项目|实测结果|
|---|---|
|curve 显式解释器|`ADAM_MF_PYTHON=/home/longt29/miniconda3/envs/curve/bin/python bash scripts/acceptance.sh` 通过；依赖未修改|
|完整测试|curve 和 adam-mf 均为 104 passed；第三方 warning 各39条，无失败|
|四卡硬件|物理 0、1、2、3 均为 12288 MiB RTX 3080 Ti；Driver 570.133.20；PyTorch CUDA runtime 12.6|
|逐卡算法|四卡分别执行 59 项优化器、clipping、噪声、隐私和预训练模型数值测试，全部通过|
|UUID 映射|逐卡独立 CUDA 进程核对 UUID 与 nvidia-smi；逻辑 cuda:0 对应指定物理卡|
|调度|真实 scheduler 使用合成 worker 检查空卡补位、全局最多4、每卡最多1、失败停调；内核 flock 继承测试通过|
|缺卡|模拟缺少 GPU 3，明确报错，不降级；额外 GPU 9 不进入允许集合|
|160 项计划|80 CV + 80 NLP，16 个方法/任务组合，每组恰好十种子；模拟派发首先覆盖 GPU 0–3|
|独立资产|从空目录下载固定 SST-2/BERT/ViT，解包已下载且 SHA 校验的 CIFAR 归档和仓库矩阵包；35 项资产/清单哈希通过|
|独立预处理|从原始 parquet/pickle 重建分割和全部 token 张量，内容 SHA-256、dtype、shape 与原实验完全一致；使用新目录完整测试通过|
|下载|真实远端下载通过；本地 HTTP 夹具验证 Range 续传、503 重试、服务器忽略 Range、完整文件跳过、错误 SHA/失败拒绝发布|
|幂等与离线|重复 download + prepare 后所有资产及 manifest 哈希不变；离线预训练模型加载与完整测试通过|
|恢复|八种方法真实引擎的合成输入中途恢复与连续运行逐步一致；CPU/CUDA 噪声状态恢复通过；GPU UUID 绑定、配置/代码/资产变更拒绝通过|
|后台生命周期|新会话后台运行、继承锁、防重复启动、PID身份和安全停止实测；stop 后 paused；孤立 worker 停止会核对 PID token|
|统计与审计|160 条合成记录验证结果汇总、样本SD、Student-t区间、配对及完整性；不完整计划不能标 completed|
|源码提交|根目录 ignore 规则不再屏蔽 src/adam_mf/data；矩阵包使用普通 Git，无 LFS 依赖|

最终默认 adam-mf dry-run：`outputs/runs/dry-20261102-191853-656405-1339031/`，状态 `dry_run_complete`、planned=160、completed=0、training_started=false。最终代码在 curve 和 adam-mf 均为 104 passed；逐卡测试各 59 passed。完整机器报告见 [engineering_validation.json](engineering_validation.json)，curve 日志为 `logs/curve_final_tests.log`。

## 独立 Conda 安装

安装锁包含 78 个固定版本，包含 PyTorch 请求的 CUDA Toolkit extras；环境检查额外验证依赖闭包，避免在已有环境中漏锁运行库。验证安装使用官方 wheel 下载及本机已有 pip wheel 缓存，所有数值库保持原版本。没有复制或重命名 curve 环境。

干净创建与安装 **通过**：标准 `scripts/setup_env.sh` 在离线 Conda/pip 缓存下成功创建独立环境，安装78项固定依赖及项目，退出码0；官方索引与CUDA缺失包已实际联网获取并校验。没有克隆 curve。Python 为 `/home/longt29/miniconda3/envs/adam-mf/bin/python`，Conda环境名为 `adam-mf`，`pip check` 为0。

已有环境重复运行安装脚本也退出0；Conda history SHA-256 与依赖 freeze 前后完全一致，未重建或升级依赖。默认解释器选择、项目导入、四卡CUDA及完整104项测试均通过。最终默认环境验收移除了 `LD_LIBRARY_PATH`，不依赖本机系统 CUDA Toolkit 路径。环境报告为 `outputs/runs/dry-20261102-191853-656405-1339031/audit/environment_report.json`。

当前环境、资产、四卡和工程预检查满足正式启动条件。可执行 `./start.sh` 开始正式任务；本轮没有执行该正式启动命令。

## 资产来源与部署

数据与模型来源、commit、逐文件大小和 SHA-256 见 `config/assets.lock.json` 和 [assets.md](assets.md)。矩阵通过仓库自带 `resources/matrices-v1.tar.gz` 固定分发，759 KiB，SHA-256 `6e84b92685583296299d335749c502a15de2b566d685d9ad83ca778cf8ef724b`。没有虚构公开地址，也没有重新优化矩阵。

统一命令为根目录和 scripts 下的 download_assets / prepare_data / verify_assets；start 会在后台完成环境、资产、预处理、CUDA/算法预检查，再进入四卡统一队列。check 遇到缺失文件会自动准备，verify 始终离线。README 已整体重写，包含研究介绍、系统要求、两环境、GPU、来源、启动、后台、配置、结果和 FAQ。

## 已知边界

- 按要求未运行正式160次训练，正式结果为0；准确率、完整训练时间和正式训练批次的峰值显存需正式执行后产生。
- curve 中原有其他项目 dp-muon-correlated 的 Optax 版本冲突已记录，本轮未改动它；本项目78项锁定依赖检查通过。
- 代码变更、配置或资产变更会使旧运行的恢复身份失配；本轮开发时的中间 dry-run 失败记录不能冒充最终验收。旧验证文件已标记为历史轮次。
- 修改保留在当前工作区，未提交或推送 GitHub。正式启动命令留给用户执行。
