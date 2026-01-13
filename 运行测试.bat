@echo off
REM xLSTM-Mixer 快速测试脚本
REM 数据集路径: E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets

echo ========================================
echo xLSTM-Mixer 快速测试
echo ========================================
echo.

cd /d %~dp0

REM 检查是否在正确的目录
if not exist "xlstm_mixer" (
    echo 错误: 请在 xLSTM-Mixer 项目根目录下运行此脚本
    pause
    exit /b 1
)

REM 检查数据集是否存在
set DATASET_PATH=%CD%\common-ts\datasets\tslib_datasets
if not exist "%DATASET_PATH%" (
    echo 错误: 数据集路径不存在: %DATASET_PATH%
    echo 请确保数据集已正确放置
    pause
    exit /b 1
)

echo 当前目录: %CD%
echo 数据集路径: %DATASET_PATH%
echo.
echo 开始运行快速测试（fast_dev_run）...
echo.

python -m xlstm_mixer fit+test ^
    --data ForecastingTSLibDataModule ^
    --data.dataset_name Weather ^
    --data.root_path "%DATASET_PATH%" ^
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

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo 测试成功完成！
    echo ========================================
) else (
    echo.
    echo ========================================
    echo 测试失败，请检查错误信息
    echo ========================================
)

pause


