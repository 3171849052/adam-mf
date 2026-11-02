> 此文件记录上轮验收，含已失效的部署与 Git LFS 描述。当前工程验收以 [engineering_validation.md](engineering_validation.md) 为准；参数保全以 [config_preservation.json](config_preservation.json) 为准。

# 交付验收记录

2026-11-02（Asia/Shanghai），在本机现有 curve / Python 3.11.15、RTX 3080 Ti 环境执行。完整机器可读记录见 validation.json；参数核对见 parameter_audit.json；源文件哈希及迁移表见 source_audit.json / protocol_notes.md。

## 实际仓库结构

```text
adam-mf/
├── README.md, pyproject.toml, environment.yml, .gitignore, .gitattributes
├── start.sh, status.sh, stop.sh
├── config/experiment.yaml
├── data/{cifar-100-python,sst2,preprocessed,manifest.json}
├── model/{vit-tiny,bert-tiny,manifest.json}
├── assets/matrices/                    # 8组正式矩阵及元数据、manifest
├── src/adam_mf/
│   ├── cli.py, config.py, paths.py, registry.py, audit.py
│   ├── data/{loaders,preprocessing}.py
│   ├── models/{vit_tiny,bert_tiny}.py
│   ├── optimizers/{adam,momentum,stpex,builder}.py
│   ├── privacy/{clipping,noise_b2lt,accountant,calibration}.py
│   ├── training/{engine,checkpoint,evaluate}.py
│   ├── runtime/{launcher,scheduler,worker,state}.py
│   └── reporting/{statistics,report}.py
├── scripts/{copy_assets.py,verify_assets.py,python.sh}
├── tests/                             # 算法、资产、配置、恢复、调度、报告
├── docs/                              # 来源、参数及验收审计
├── outputs/runs/                      # 本次只有不训练的 dry-run
└── logs/
```

workload 构造和矩阵验证集中在 privacy/calibration.py；Ghost clipping 集中在 privacy/clipping.py；IID/B²LT 共用同一个原始在线噪声实现。数据加载与模型分别复用两系列相同实现，并保留 CV 评价 batch 的历史差异。

## 已执行的检查

|检查|实际结果|
|---|---|
|数据实际复制|9文件，400,549,965字节，含 CIFAR-100、SST-2 原始文件及必需预处理输入|
|模型实际复制|6文件，41,583,307字节，ViT safetensors + BERT全部5文件|
|矩阵实际复制|16文件，21,099,490字节；T250/310 的8组 npz/json；源验收哈希一致|
|SHA-256|31个资产文件+3个manifest全部通过；没有软链接|
|预处理全量重建|SST-2 train/validation 每个 token tensor 与复制输入完全一致；CV/NLP 每个划分索引一致|
|冻结参数|16个任务/方法组合均与冻结文件和历史正式训练输入一致，未导入历史准确率|
|完整单元测试|79 passed，95.47秒；33条第三方 warning，无失败|
|GPU算法测试|CPU/CUDA 参数化测试55 passed，13.77秒，含真实预训练模型上的独立合成输入|
|噪声恢复补测|13 passed，12.77秒，含 CPU/CUDA B²LT 旧 innovation 回放的位级一致性|
|任务生成|160个唯一ID；8方法、2任务、每组10种子|
|启动|`./start.sh --dry-run --wait` 成功，后台、日志、PID与状态可用|
|重复启动|真实运行期间第二次 start 被继承文件锁拒绝|
|停止|另起不训练的 dry-run 执行 stop.sh，协调器退出、状态 paused，无正式训练|
|报告事务|合成注入报告错误后状态 failed；恢复后才设 completed，不会误开启新的160次训练|
|原仓库只读|29个重点源文件哈希全部未变；原仓库 git status --porcelain 无输出|

完整测试包含八种优化器公式、Adam偏差修正、Momentum定义、无噪声无裁剪梯度和、缩放/逆缩放、STP-EX、重复 token / padding-aware Ghost clipping、B²LT 全时域稠密矩阵参考、GDP与前缀预算、GPU显存准入、Student-t统计及四张图。八种方法均在独立小型合成输入上比较了连续运行和中途恢复后的逐步梯度、损失、模型参数、噪声状态和数据增强摘要。报告系统用 pytest 临时目录的合成记录验证160行与不完整状态，未写入正式结果目录。

第三方 warning 来自 Opacus/PyTorch backward hook、NumPy 2.4 对旧 CIFAR pickle 的弃用提示，以及合成 CPU 测试的 pin_memory 提示。独立数值参考比较均通过，没有通过屏蔽失败或降低协议要求来获得成功。

成功的最终启动目录：`outputs/runs/dry-20261102-162149-874761-1100796/`；其中 task_plan.json=160 项、status=dry_run_complete、training_started=false。停止测试目录：`outputs/runs/dry-20261102-162414-330554-1113501/`，status=paused。logs/current.json 指向最近的停止测试；没有 last_formal.json，所以首次直接 start.sh 将新建正式实验。

## 交付边界

未执行160次正式训练，当前正式训练记录为0，符合用户禁止擅自启动完整训练的要求。完整训练的最终准确率、160条正式记录和最终研究报告需要用户执行 start.sh 后实际产生；已有报告实现和测试，没有用历史结果代替。

本机未安装 Git LFS。资产全部保留在仓库且未被忽略，.gitattributes 已配置大文件规则；GitHub 分发前需要安装 LFS，或直接打包含资产的归档。仓库已 git init，尚未提交或推送。

离线环境和GPU准备完成后，在仓库根目录执行：

```bash
./start.sh
./status.sh
tail -f logs/pipeline.log
./stop.sh
```
