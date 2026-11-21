# openT2L

开源 Text-to-LoRA（导入名 `t2l`）。

[English](README.md)

给冻结的模型看几个新任务的示例，一次前向计算即可得到该任务的 LoRA 适配器，使用时无需微调。

一个小型超网络读取 k 个示例（输入及其正确输出）的嵌入，为冻结的模型生成 LoRA 权重。适配器以标准 PEFT 格式保存，任何能加载 PEFT LoRA 的工具都可以使用：transformers、vLLM，或经转换后的 llama.cpp。

- 适用于**任何**被适配层为 `nn.Linear` 的 PyTorch 模型：纯文本大模型、视觉语言模型、混合注意力模型。
- 核心依赖只有 `torch` 和 `safetensors`。
- 超网络保存为 safetensors + JSON，不使用 pickle。JSON 中记录了示例的嵌入方式。

## 安装

```bash
pip install opent2l          # 核心
pip install "opent2l[hf]"    # + transformers、peft（运行示例、加载适配器）
```

尚未发布到 PyPI。在此之前可运行 `pip install "opent2l[hf] @ git+https://github.com/ASH1998/openT2L"`，或在克隆的仓库中运行 `pip install -e ".[hf]"`。

## 用法

```python
import torch
from t2l import (find_targets, HyperLoRA, Episode, train_hypernet,
                 save_hypernet, load_hypernet, example_embedding, emit_adapter)

targets = find_targets(model, ["q_proj", "v_proj"])          # 按名称选层，形状从模型读取
gen = HyperLoRA(targets, emb_dim=model.config.hidden_size, rank=8, alpha=16)

# 训练：每个 episode = 同一任务的 k 个支持示例嵌入 [k, D] + 1 个查询示例
train_hypernet(gen, model, batches, loss_fn=lambda m, q: m(**q).loss, steps=6000, lr=1e-4)
save_hypernet(gen, "hypernet/", base_model="org/model", embedding={"pooling": "answer", "layer": -1})

# 使用：新任务的几个示例 -> PEFT 适配器目录
gen, cfg = load_hypernet("hypernet/")
emit_adapter(gen, torch.stack([embed(b) for b in examples]), "adapter/")
```

完整示例见 [examples/quickstart_text.py](examples/quickstart_text.py)。加 `--tiny` 参数时，它在随机初始化的小型 Llama 上离线运行，可用作冒烟测试。

## 原理

- **超网络**（Text-to-LoRA "M" 结构）：对每个目标模块，用 [任务嵌入，层位置嵌入，模块类型嵌入] 查询同一个 MLP，再由该权重形状对应的输出头产生 A 和 B。
- **初始状态**：每个适配器 = 学到的共享部分 + 任务部分，两者的 B 都从零开始，因此训练从"不改变模型"开始。
- **任务嵌入**：默认取某一隐藏层在答案 token 上的平均，再对 k 个示例求平均。可以替换为任何返回定长向量的函数，但训练和使用时必须用同一个。
- **训练**：只训练超网络，基座模型保持冻结。

## 预训练超网络

| 超网络 | 基座模型 | 任务类型 | 许可 |
| --- | --- | --- | --- |
| [ASHu2/openT2L-MiniCPM-V-4.7-1B-docs](https://huggingface.co/ASHu2/openT2L-MiniCPM-V-4.7-1B-docs) | openbmb/MiniCPM-V-4.7-1B | 文档 → JSON 信息抽取（表单、票据、表格、发票；中英文） | CC-BY-NC-4.0 |

```python
from huggingface_hub import snapshot_download
from t2l import load_hypernet
gen, cfg = load_hypernet(snapshot_download("ASHu2/openT2L-MiniCPM-V-4.7-1B-docs"), device="cuda")
print(cfg["embedding"])   # 该超网络要求的示例嵌入方式
```

模型卡中有完整的 MiniCPM-V 示例嵌入代码。

规格：
- 7220 万参数；
- 为语言模型的 48 个模块生成 rank-8 LoRA：6 个全注意力层的 q/v，以及 18 个线性注意力层的输入、输出投影；
- 在 333 个文档任务上训练，每个任务使用 k ∈ {1, 2, 4, 8} 页示例。

生成的每个 LoRA 都是普通的 PEFT 适配器，fp16 下约 3.6 MB（180 万参数）。部分训练数据集仅限非商业使用，因此该权重采用非商业许可。

## 结果

本项目为 MiniCPM Spark Program 开发，基座模型为 MiniCPM-V-4.7-1B，任务为文档信息抽取。在 78 个留出的文档任务上，由 4 页示例生成的适配器把字段级 F1 从 0.728 提升到 0.881（+0.154，95% 置信区间 +0.116 至 +0.195），78 个任务中有 71 个提升。该结果只与基座模型做了比较。协议、数据与局限性见 [openBMB-spark-t2l](https://github.com/ASH1998/openBMB-spark-t2l) 仓库。

## 许可

代码：Apache-2.0。预训练超网络权重遵循其训练数据的许可，见上表。
