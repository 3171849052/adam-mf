# 独立资产溯源与分发

`config/assets.lock.json` 是下载与离线验证的可信清单，固定下载文件的 SHA-256 和字节数，并声明生成资产。`data/manifest.json`、`model/manifest.json`、`assets/matrices/manifest.json` 是生成的审计清单，不能通过修改它们绕过可信清单。

## 数据

- CIFAR-100：Toronto 官方 Python pickle 归档（2010-02-20），169001437 字节，SHA-256 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`。正式训练使用全部 50000 张训练图，评价使用官方 10000 张测试图。`train/test/meta` 分别与原实验逐字节校验。保留历史 45000/5000 索引作为协议核对资产；不用于搜索，也不缩减正式 CV 训练集。
- SST-2：Hugging Face `nyu-mll/glue` commit `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` 下 `sst2` parquet。原始训练集 67349、官方验证集 872；与旧仓库两个文件 SHA-256 相同。固定 stratified 划分保留 62000 条训练数据，采用原种子 20261008 和后续 seed+1 排序；正式评价仍为官方 872 条验证数据。
- CIFAR 备用源是固定 commit `nakroy/cifar100-python@201a32345d2c6b970e1a36c582930c83e09c96d2`，只接受同一 SHA-256 的官方原始归档，不使用重编码图像或其他 CIFAR 变体。

## 模型

- ViT：`timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k@7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106`。权重 `model.safetensors` 与旧 cache 一致，另下载同 commit 的配置。保留原 timm backbone 张量映射、100 类随机头初始化和预处理 metadata。
- BERT：`prajjwal1/bert-tiny@6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837`。`config.json`、`pytorch_model.bin`、`vocab.txt` 与旧 cache 一致。2 层、hidden size 128、2 heads、词表 30522；uncased tokenizer，长度 128。
- 原 cache 的 `tokenizer.json` / `tokenizer_config.json` 是本地生成版本，不能冒充模型仓库提供的文件，因此随代码保存在 `config/tokenizer/`，复制时按固定 SHA-256 校验。保留原 BERT pooler 和随机二分类头初始化；不下载替代分类器。
- 预处理缓存采用原始文件推导的逐张量内容 SHA-256（包括 dtype、shape）验证；PyTorch ZIP 容器字节可能因生成位置而不同，因此生成清单另外记录实际缓存文件 SHA-256。恢复仍检查完整文件哈希，不能替换缓存。

## 计算矩阵

矩阵不从压缩包读取。`./generate_matrices.sh` 迁入 exp16a 的数据无关 L-BFGS 求解；exp16e 原本加载 exp16a 的结果，本项目现在独立重算。详见 [matrix_generation.md](matrix_generation.md)。

每个 T=250/310 保存 identity/IID 和 SGD、Momentum、Momentum-Bias/B²LT，共 8 组 NPZ/JSON。NPZ 保存 A、C、D、d、covariance_clip1；JSON 保存初值、选中参数、迭代轨迹、收敛状态、数值审计、生成代码/环境身份及 NPZ 哈希。生成后冻结供十种子共享，训练时才采样独立高斯 innovation。

可信清单对数据和模型固定下载字节，对计算矩阵规定所需文件集合；矩阵使用生成配方、NPZ 哈希、可行性和数学恒等式验收，清单记录本次实际哈希。恢复身份继续绑定实际资产，不能以重新求解后的矩阵继续旧检查点。

参考：[Toronto CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)、[GLUE 固定版本](https://huggingface.co/datasets/nyu-mll/glue/tree/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/sst2)、[ViT 固定版本](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/tree/7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106)、[BERT 固定版本](https://huggingface.co/prajjwal1/bert-tiny/tree/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837)。
