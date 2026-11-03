# B²LT 矩阵计算

`./generate_matrices.sh` 从固定解析初值独立求解，不读取旧矩阵、历史 theta/omega 或旧仓库。正常下载/启动准备也会自动计算缺失矩阵。`--rebuild` 显式重新计算已有文件；完整有效结果默认复用，损坏或配方变化时报错。求解只使用 CPU，不启动模型训练。

来源为 `../dp-adam-correlated/exp16a/matrices.py` 的 `optimize` 和 `audit_arrays`；仅保留本项目需要的 B²LT 分支。exp16e 本身不求解，它使用 exp16a 已计算的矩阵。

- A：SGD / Momentum / Momentum-Bias 累积更新 workload，beta1=0.9。
- D：把独立高斯 innovation 转换成相关梯度噪声的下三角 Toeplitz 矩阵。
- C=D⁻¹：策略矩阵；B=A D，满足 B C=A。
- Q=5，P=T/5；T=250/310，P=50/62，五轮固定参与。
- d[0]=1；1≤j<P 时 d[j]=Σ omega[q] theta[q]^(j−1)，其余为零。
- 初值 theta[q]=(q+1)/6，omega[q]=−(1−theta[q])/10。
- 目标为 sensitivity(C)² × ||A D||F²/T；敏感度采用原 fixed_epoch_sensitivity 的绝对 Gram 上界，不采用采样放大。
- FP64，PyTorch 两线程，L-BFGS memory=20，最多1000次迭代，每步最多60次可行 Armijo 回溯；保留原 barrier 权重和停止阈值，从初值和已接受迭代中按不含 barrier 的目标选最优。

`assets/matrices/*.{npz,json}` 保存矩阵及求解记录，`generation_audit.json` 汇总数学验收。NPZ 中 `covariance_clip1 = sigma² D Dᵀ` 是裁剪阈值1、梯度和单位、ε=8、δ=1e−5 的噪声协方差；逻辑批次平均梯度的协方差还要除 batch_size²，实际 clip 为 c 时再乘 c²。它不是采样噪声。训练仍使用原在线 Q-buffer 和种子生成实际噪声，未改为稠密矩阵乘法。

验收覆盖 C D=D C=I、B C=A、独立敏感度界、协方差白化、B²LT 可行性、Toeplitz 递推和保存文件哈希。到达迭代上限不等于数学验收失败；与原实验一致，保留明确的 success/message 和已接受最佳迭代，不冒充收敛。

生成代码与数值环境固定后复用结果；不同硬件/数学库可能出现浮点差异，不能仅凭同一求解公式承诺历史准确率逐位一致。已有训练检查点不得迁移到不同矩阵身份。实验学习率、裁剪阈值、eps_scale 和种子本次保持不变。

## 本机重算验收（2026-11-03）

8 组正式矩阵均从规定初值重新计算并保存，求解总计约 14.8 分钟。所有 A、C、D、d、covariance_clip1 数组与原 exp16a 逐元素完全相同（最大绝对差为0）。旧数组仅用于计算完成后的对照，没有作为求解输入。逐文件哈希、源码哈希、耗时及优化器停止状态见 [matrix_regeneration_validation.json](matrix_regeneration_validation.json)。

独立源码目录（无旧仓库、归档或矩阵缓存）的小规模生成通过；真实尺寸生成后的重复命令只校验复用。完整资产准备和离线验证通过35项哈希。未启动正式训练，也未改动实验超参数。

回归测试分两批执行：包含 CUDA 的非数据测试114项通过，数据测试3项通过，合计117项通过；四卡环境检查通过，正式计划仍为120项。
