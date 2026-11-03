# 源码迁移与协议核对

算法来源为开发阶段核对过的原仓库 exp15c 和 exp16e；实验参数权威为 config/experiment.yaml；逐文件 SHA-256 见 source_audit.json。本仓库训练、恢复和报告均不导入原仓库，不读取历史准确率。

|新方法 ID|优化器来源|批处理 / 噪声 / 裁剪来源|
|---|---|---|
|adam-iid|exp16e/optimizer.py 标准 Adam|exp16e/clipping.py、noise.py、privacy.py、train.py|
|momentum-iid|exp16e/momentum_optimizer.py（与 exp15c 实现相同）|同上|
|adam-mf-sgd-b2lt|exp16e/optimizer.py|同上，SGD workload 的 B²LT|
|adam-mf-sgd-stp-b2lt|exp16e/optimizer.py AdamLogicalBatch|scaled clipping；冻结上一已完成步 vhat；缩放、加噪、撤销缩放|
|stpex-mf-momentum-bias-b2lt|exp16e/stpex_optimizer.py|scaled clipping；Momentum-Bias workload 的 B²LT；无末尾逆缩放|
|momentum-mf-momentum-b2lt|exp16e/momentum_optimizer.py|原空间 clipping；Momentum workload 的 B²LT|

所有 DP 方法沿用 exp16e/privacy.py 的 GDP 校准和 fixed_epoch_sensitivity：参与列的 Gram 子矩阵取绝对值求和上界，不进行采样放大。噪声先以梯度和单位生成，每个 logical step 只抽取一次，再除 logical batch size。B²LT 在线 Q-buffer 用私有独立 Generator 及历史 Generator 状态重放旧 innovation，保持原 FP32 运算次序；不替换为稠密矩阵乘法。稠密乘法只用于独立单元测试。

Momentum：m=beta1*m+(1-beta1)*g，p+=-lr*m，无偏差修正、无 Nesterov。Adam 使用 torch.optim.Adam 的一、二阶偏差修正。STP-EX 使用 q（已在缩放空间裁剪加噪的平均梯度）更新 u；p-=lr*u/(1-beta1^t)；v=beta2*v+(1-beta2)*(d_prev*q)^2；d_prev=sqrt(vhat_prev)+eps_scale。Adam-STP 与 STP-EX 不可互换。

参数核对：六种 DP 方法核对 exp16e/results/final_freeze.json，并与 formal_per_seed.csv 核对。已移除 adam-no-noise 和 momentum-no-noise，其余方法的冻结参数保持不变；历史审核文件的旧配置哈希不代表本轮参数，当前保全证据见 config_preservation.json。

## 发现的历史依赖和差异

- exp16e/matrices.py 实际从 exp16a/results/matrices 载入，并对 phase1_acceptance.json 的哈希验收。本仓库仅复制 T=250/310、identity/IID、SGD/B²LT、Momentum/B²LT、Momentum-Bias/B²LT 的 8 组已验收 npz/json。元数据中的历史设计迭代轨迹仅是冻结矩阵来源证据；没有迁入任何矩阵优化代码。
- exp16e 数据资产实际在 runtime/assets，逐张量与 exp9 历史输入比较后生成。独立重建这些输入，并提供 verify-preprocessing 从本地原始 parquet、CIFAR 文件和 tokenizer 重新计算并逐张量/逐索引验证；不依赖 exp9。
- CV 正式训练使用完整 50000 条训练数据，历史 CV 45000/5000 划分只保留为预处理审计。NLP 的 62000 条训练索引来自固定分层划分，再以 split_seed+1 做初始排列。各 trial 在其上以 Generator(seed).randperm 再排列；同一 epoch 顺序逐轮复用。
- CV 增强：RandomResizedCrop(224, bicubic)、RandomHorizontalFlip、ToTensor、预训练 mean/std；每轮新建 DataLoader iterator，两名 spawn worker，Generator(seed) 跨 epoch 延续。NLP 使用 BertTokenizer（非 fast 替换），长度128、max_length padding、truncation；BERT dropout=.1/eager attention，seed+100000+logical_step。
- ViT 原缓存只有 safetensors 及 HF revision 链接，没有独立 architecture config 文件。下载固定 commit 对应的相同权重字节，另外保留同版本配置；继续用相同 timm 模型定义及原 cv_model.py 的严格映射，保留随机100类分类头。BERT 下载固定 commit 的权重、配置、词表，并复用随代码分发的本地生成 tokenizer 文件。没有下载替代版本。
- exp15c 正式 CV 评价 batch=100；exp16e 正式 CV 评价 batch=physical_batch（250）；NLP 均=100。保留此差异，避免改变浮点归约次序。训练参数完全一致。
- 原始 train.py 只保存末尾 checkpoint，原 README 明确仅任务边界恢复。本仓库扩展为完整逻辑步检查点。中途恢复时从 epoch 起点重放并丢弃已处理 microbatch，复原 worker 增强 RNG；随后恢复主线程 CPU/CUDA/Python/NumPy RNG、优化器、噪声 Generator 与 B²LT buffers。不会重新执行已完成 logical step 的优化器更新。独立合成数据测试比较中断/连续两条轨迹的逐步梯度、损失、参数与增强摘要。
- 原始协议只对单次训练声明 ε=8；十次种子或六方法的组合没有相同预算保证。
- 原历史报告使用过官方评价集。本项目不是新盲测，不复制其准确率、不自动修改任何参数。

依赖版本固定为实际验证的版本，见 requirements.lock；对其他 GPU 架构/依赖版本不能承诺跨平台位级一致。恢复严格校验代码、依赖版本、配置和资产，环境发生变化会拒绝续训。
