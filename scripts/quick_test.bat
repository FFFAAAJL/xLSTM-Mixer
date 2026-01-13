@echo off
REM 快速测试脚本 - Windows 版本
REM 使用方法: quick_test.bat [数据集路径]
REM 例如: quick_test.bat E:\LSTM\common-ts\datasets\tslib_datasets

setlocal

REM 设置默认数据集路径
if "%1"=="" (
    REM 默认使用项目目录下的数据集路径
    set DATASET_PATH=%~dp0..\common-ts\datasets\tslib_datasets
) else (
    set DATASET_PATH=%1
)

echo ========================================
echo xLSTM-Mixer 快速测试
echo ========================================
echo 数据集路径: %DATASET_PATH%
echo.

REM 检查数据集路径是否存在
if not exist "%DATASET_PATH%" (
    echo 错误: 数据集路径不存在: %DATASET_PATH%
    echo 请先下载数据集或指定正确的路径
    echo.
    echo 使用方法: quick_test.bat [数据集路径]
    pause
    exit /b 1
)

echo 开始运行快速测试...
echo.

python -m xlstm_mixer fit+test ^
    --data ForecastingTSLibDataModule ^
    --data.dataset_name Weather ^
    --data.root_path %DATASET_PATH% ^
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

