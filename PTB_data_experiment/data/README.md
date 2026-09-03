# Data

这个目录包含 PTB mechanism study 使用的预处理语料文件：

- `ptb_v5000.pt`：5000-vocab PTB bigram 数据，供 `targets.py` 构造各个 \(P(y|x)\) setting。

已有结果、图片和 setting metadata 可以直接阅读；如需重跑实验，代码会优先读取这个文件。
