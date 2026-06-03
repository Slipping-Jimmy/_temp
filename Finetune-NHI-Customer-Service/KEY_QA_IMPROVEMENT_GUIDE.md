# Key QA 记忆改进方案

## 问题诊断

根据 `training_metrics.jsonl` 分析：

1. **验证集不包含 key_qa** 
   - `LOAD_BEST_MODEL_AT_END = False` 表示验证数据只来自 general QA
   - 无法监控 key_qa 的实际学习效果
   - 模型无法针对 key_qa 进行优化

2. **key_qa 数据曝光不足**
   - key_qa 数据混在训练集中，相对于 general 数据只看一遍
   - 对于需要强制记忆的 key_qa，一次曝光不够

3. **学习率偏高**
   - LR=8e-6 对于需要精准记忆的任务可能过大
   - 容易导致梯度波动，影响 key_qa 的收敛

4. **Batch size 偏小**
   - PBS=1 意味着每次梯度更新都是单样本
   - 缺乏足够的梯度累积稳定性

## 改进方案 v2

### 1. 增加 key_qa 训练机会（重复采样）

```python
KEY_QA_REPEAT_FACTOR = 3  # key_qa 在训练中重复 3 次
```

**效果**：
- key_qa 被模型看到 3 倍多的次数
- 增加记忆强度而不增加总训练时间过多
- 可根据需要调整（试试 2-5 的范围）

### 2. 改进验证策略

```python
LOAD_BEST_MODEL_AT_END = True  # 改为 True，加载最优模型
# 验证集现在包含 10% 的 key_qa 数据
val_dataset = concatenate_datasets([val_dataset, key_qa_split["test"]])
```

**效果**：
- 可以单独追踪 key_qa 的 eval_loss
- 基于 key_qa 表现选择最优模型检查点
- 更准确地反映模型对 key_qa 的掌握程度

### 3. 调整训练超参

```python
# Batch size 调整（提高梯度稳定性）
PER_DEVICE_TRAIN_BATCH_SIZE = 2   # 1 → 2
GRADIENT_ACCUMULATION_STEPS = 4   # 8 → 4（有效 batch size 保持）

# 学习率降低（更平稳的优化）
LEARNING_RATE = 5e-6              # 8e-6 → 5e-6

# 增加预热比例（更好的初期学习）
WARMUP_RATIO = 0.15               # 0.1 → 0.15

# 增加训练周期（更多 key_qa 曝光）
NUM_TRAIN_EPOCHS = 3              # 2 → 3
```

## 预期改进

| 指标 | 原配置 | 改进后 | 预期效果 |
|-----|-------|-------|--------|
| key_qa 曝光次数 | 1x | 3x | **更强记忆** |
| eval_loss 可见性 | 无 key_qa | 包含 key_qa | **能监控效果** |
| 学习率 | 8e-6 | 5e-6 | **更稳定收敛** |
| 训练周期 | 2 | 3 | **更多学习机会** |
| Batch size | 1 | 2 | **更稳定梯度** |

## 运行命令

```bash
# 运行改进版本
python sft_trainer_unsloth_gemma3_qlora_v2.py

# 对比两个版本的训练结果
python -c "
import json
v1_file = 'checkpoints/wingeneai-gemma-3-27b-clean-sft-general-keyqa-paraphrase-20260601_111933/training_metrics.jsonl'
v2_file = 'checkpoints/wingeneai-gemma-3-27b-clean-sft-general-keyqa-v2-*/training_metrics.jsonl'
# 查看最后的 eval_loss，特别是 key_qa 部分
"
```

## 进一步优化建议

### 如果 key_qa 仍需改进，可尝试：

1. **增加重复因子**
   ```python
   KEY_QA_REPEAT_FACTOR = 5  # 试试更高的值
   ```

2. **降低学习率**
   ```python
   LEARNING_RATE = 2e-6  # 进一步降低
   ```

3. **增加更多验证 key_qa**
   ```python
   # 修改 build_train_val_datasets() 中的 VAL_RATIO
   # 从 0.1 改为 0.2，增加验证集中 key_qa 的比例
   ```

4. **用两阶段训练**（高级）
   - 第一阶段：general + key_qa 混合（3 epoch）
   - 第二阶段：仅 key_qa（1 epoch），LR 更低
   
5. **使用数据权重**（如果框架支持）
   ```python
   # 给 key_qa 样本更高的权重
   # 这需要修改 process() 函数添加权重列
   ```

## 监控指标

训练中关注这些指标：

```json
{
  "event": "log",
  "loss": "训练损失（应持续下降）",
  "eval_loss": "验证损失（现在包含 key_qa）",
  "learning_rate": "学习率变化"
}
```

特别注意最后一个 eval_loss 值，应该低于原配置的 0.6197502。

## 快速对比

| 配置项 | 原始版本 | v2 版本 |
|-------|---------|--------|
| 文件 | `sft_trainer_unsloth_gemma3_qlora.py` | `sft_trainer_unsloth_gemma3_qlora_v2.py` |
| Key_QA 曝光 | 1倍 | 3倍 |
| 验证中有 Key_QA | ❌ | ✅ |
| 学习率 | 8e-6 | 5e-6 |
| Epochs | 2 | 3 |
| 可调参数 | 固定 | `KEY_QA_REPEAT_FACTOR` |
