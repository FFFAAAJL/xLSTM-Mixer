# xLSTM-Mixer Windows 运行指南

## 环境准备

### 1. 激活您的 conda 环境

```bash
conda activate pyt39
```

### 2. 检查并安装依赖

您的 `environment.yml` 已经包含大部分依赖，但需要检查以下几点：

#### 检查 xlstm 版本
- `requirements.txt` 要求 `xlstm==1.0.3`
- 您的环境中有 `xlstm==2.0.0`
- **建议**：先尝试运行，如果出现兼容性问题，再降级：
  ```bash
  pip install xlstm==1.0.3
  ```

#### 安装项目本身
```bash
cd xLSTM-Mixer
pip install -e .
```

这会安装 `xlstm_mixer` 包并注册 CLI 命令。

### 3. 准备数据集

#### 下载数据集
从 [Google Drive](https://drive.google.com/drive/folders/1B6BP6fA6j29azC-BJyDLDtfNzxT8cK2Y?usp=sharing) 下载数据集。

#### 设置数据集路径
您的数据集已经放在：`E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets`

在运行命令时，使用 `--data.root_path` 参数指定这个路径（见下方示例）。

## 运行模型

### 快速测试（推荐先运行）

由于您在 Windows 上，`.sh` 脚本无法直接运行。您可以使用以下 Python 命令进行快速测试：

```bash
python -m xlstm_mixer fit+test ^
    --data ForecastingTSLibDataModule ^
    --data.dataset_name Weather ^
    --data.root_path E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets ^
    --data.seq_len 96 ^
    --data.pred_len 96 ^
    --data.label_len 0 ^
    --data.batch_size 32 ^
    --data.num_workers 0 ^
    --model LongTermForecastingExp ^
    --model.criterion torch.nn.L1Loss ^
    --model.architecture xLSTMMixer ^
    --model.architecture.num_mem_tokens 2 ^
    --model.architecture.xlstm_num_heads 8 ^
    --model.architecture.xlstm_num_blocks 1 ^
    --model.architecture.xlstm_embedding_dim 256 ^
    --model.architecture.xlstm_conv1d_kernel_size 0 ^
    --model.architecture.xlstm_dropout 0.25 ^
    --optimizer.lr 0.001 ^
    --lr_scheduler.constant_gamma_epochs 2 ^
    --lr_scheduler.gamma 0.98 ^
    --lr_scheduler.cosine_epochs 15 ^
    --lr_scheduler.warmup_epochs 5 ^
    --trainer.logger.name Weather_test ^
    --trainer.logger.project xlstm-mixer ^
    --trainer.max_epochs 5 ^
    --seed_everything 2021 ^
    --trainer.fast_dev_run true
```

**注意**：
- `^` 是 Windows CMD 的换行符（在 PowerShell 中使用 `` ` ``）
- `--data.num_workers 0` 在 Windows 上通常更稳定
- `--trainer.fast_dev_run true` 用于快速测试，只运行一个批次

### 查看帮助信息

```bash
# 查看主帮助
python -m xlstm_mixer --help

# 查看 fit 命令的详细参数
python -m xlstm_mixer fit --help
```

### 完整训练示例

```bash
python -m xlstm_mixer fit+test ^
    --data ForecastingTSLibDataModule ^
    --data.dataset_name Weather ^
    --data.root_path E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets ^
    --data.seq_len 768 ^
    --data.pred_len 96 ^
    --data.label_len 0 ^
    --data.batch_size 64 ^
    --data.num_workers 0 ^
    --model LongTermForecastingExp ^
    --model.criterion torch.nn.L1Loss ^
    --model.architecture xLSTMMixer ^
    --model.architecture.num_mem_tokens 2 ^
    --model.architecture.xlstm_num_heads 8 ^
    --model.architecture.xlstm_num_blocks 1 ^
    --model.architecture.xlstm_embedding_dim 256 ^
    --model.architecture.xlstm_conv1d_kernel_size 0 ^
    --model.architecture.xlstm_dropout 0.25 ^
    --optimizer.lr 0.001 ^
    --lr_scheduler.constant_gamma_epochs 2 ^
    --lr_scheduler.gamma 0.98 ^
    --lr_scheduler.cosine_epochs 15 ^
    --lr_scheduler.warmup_epochs 5 ^
    --trainer.logger.name Weather_xlstm-mixer_96_2021 ^
    --trainer.logger.project xlstm-mixer ^
    --trainer.max_epochs 40 ^
    --seed_everything 2021
```

## 常见问题

### 1. 如果遇到 xlstm 版本问题

```bash
pip uninstall xlstm
pip install xlstm==1.0.3
```

### 2. 如果遇到路径问题

确保数据集路径正确，Windows 路径格式：
- 正确：`E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets`
- 或使用正斜杠：`E:/LSTM/xLSTM-Mixer-main/xLSTM-Mixer/common-ts/datasets/tslib_datasets`

### 3. 如果 num_workers > 0 报错

在 Windows 上，将 `--data.num_workers` 设置为 `0`。

### 4. 如果 CUDA 相关错误

检查您的 CUDA 版本是否与 PyTorch 匹配：
- 您的环境：`torch==2.6.0+cu126`（CUDA 12.6）
- 确保 GPU 驱动支持 CUDA 12.6

## 使用 Git Bash 运行 .sh 脚本（可选）

如果您安装了 Git Bash，可以修改脚本中的路径，然后运行：

```bash
# 在 Git Bash 中
bash ./scripts/long_term_forecasting/weather.sh --dev
```

但需要先修改脚本中的路径配置。

## 下一步

1. 先运行快速测试确保环境正常
2. 下载完整数据集
3. 运行完整的训练实验
4. 查看训练日志和结果

