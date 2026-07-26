# Kappa-VAD

Non-Additive Local-Future Evidence Interaction for Training-Free Video Anomaly Detection.

## 核心方法

将冻结MLLM从异常分类器重构为交互测量器。核心异常量：
```
κ = v(B+L+F) - v(B+L) - v(B+F) + v(B)
```

## 目录结构

```
final/
├── main.py              # 入口：KappaVAD类 + 命令行接口
├── internvl_ll.py       # InternVL2-8B底层封装（视觉编码、LLM前向、4D mask）
├── tpva.py              # TPVA：渐进未来证据可见性 + HTAG+TPVA联合前向
├── pipeline.py          # SAGEPipeline：统一配置 + 模型加载
├── data_pipeline.py     # 视频读取 + 帧预处理（decord + InternVL）
├── htag/
│   ├── nodes.py         # 证据节点构造（G/L/A/R模态，活跃模态控制）
│   ├── graph.py         # 紧凑图构造 + 4D attention mask + 前向传播
│   └── token_select.py  # SST加权 + 空间覆盖token选择
└── scpage/
    ├── phase_encode.py  # ViT-B/16四相位编码
    ├── instability.py   # 相位不稳定图计算 + MAD归一化
    ├── crop_utils.py    # 框搜索 + padding掩码 + 坐标映射 + 裁剪
    └── __init__.py      # SCPageExtractor：完整SC-Page管线
```

## 依赖

- InternVL2-8B: `/sdb/data_public/llms/llm/InternVL2-8B`
- ViT-B/16: `/sdb/data_public/llms/vit-base-patch16-224-in21k`
- UCF-Crime: `/sdb/data_public/llms/videos/UCFcrime/videos`
- UCF GT: `../CoReVAD/src/ucf/gt_ucf.npy`

## 使用

```bash
# 单视频
python main.py --video Abuse028_x264 --gpu 0 --stride 4

# 限制窗口数（快速测试）
python main.py --video Abuse028_x264 --gpu 1 --stride 12 --max_windows 50
```

## 输出

- 每窗口的vB, vBL, vBF1, vBLF1值
- κ¹ = vBLF1 - vBL - vBF1 + vB
- vB AUC, κ¹ AUC, combined AUC
# FoveaVAD
