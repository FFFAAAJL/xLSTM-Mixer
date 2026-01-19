@echo off
setlocal enabledelayedexpansion

REM Windows CMD Batch Script - Weather Dataset Experiment
REM Usage: weather.bat [--dev]
REM --dev: Quick test mode (run only one batch)

REM Set WandB API Key

REM Set up Visual Studio environment for CUDA extensions (if needed)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)

REM Default: not dev mode
set dev_run=false

REM Check if --dev flag is passed
if "%1"=="--dev" set dev_run=true

echo ========================================
echo Starting xLSTM-Mixer Experiments
echo Dataset: Weather
echo Dev Run: !dev_run!
echo ========================================

REM Dataset path (Windows format)
set DATASET_PATH=E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets
set dataset=Weather

REM ========== Config 1: pred_len=96 ==========
set pred_len=96
set seq_len=768
set xlstm_dropout=0.25
set lr=0.001
set batch_size=64
set init_token=2
set xlstm_embedding_dim=256
set slstm_num_heads=8
set slstm_conv1d_kernel_size=0
set xlstm_num_blocks=1
set gamma=0.98
set cosine_epochs=15
set warmup_epochs=5
set constant_gamma_epochs=2

echo.
echo [Config 1] pred_len=%pred_len%, seq_len=%seq_len%
echo.

for %%s in (2021 2022 2023) do (
    echo Running seed %%s...
    python -m xlstm_mixer fit+test ^
        --data ForecastingTSLibDataModule ^
        --data.dataset_name %dataset% ^
        --data.root_path "%DATASET_PATH%" ^
        --data.num_workers 0 ^
        --optimizer.lr %lr% ^
        --data.seq_len %seq_len% ^
        --data.pred_len %pred_len% ^
        --data.label_len 0 ^
        --data.batch_size %batch_size% ^
        --model LongTermForecastingExp ^
        --model.criterion torch.nn.L1Loss ^
        --model.architecture xLSTMMixer ^
        --model.architecture.num_mem_tokens %init_token% ^
        --model.architecture.xlstm_num_heads %slstm_num_heads% ^
        --model.architecture.xlstm_num_blocks %xlstm_num_blocks% ^
        --model.architecture.xlstm_embedding_dim %xlstm_embedding_dim% ^
        --model.architecture.xlstm_conv1d_kernel_size %slstm_conv1d_kernel_size% ^
        --model.architecture.xlstm_dropout %xlstm_dropout% ^
        --optimizer.lr %lr% ^
        --lr_scheduler.constant_gamma_epochs %constant_gamma_epochs% ^
        --lr_scheduler.gamma %gamma% ^
        --lr_scheduler.cosine_epochs %cosine_epochs% ^
        --lr_scheduler.warmup_epochs %warmup_epochs% ^
        --trainer.logger.name %dataset%_xlstm-mixer_%pred_len%_%%s ^
        --trainer.logger.project xlstm-mixer ^
        --trainer.max_epochs 40 ^
        --seed_everything %%s ^
        --trainer.fast_dev_run !dev_run!
        
    if errorlevel 1 (
        echo Error occurred with seed %%s
        exit /b 1
    )
)

REM ========== Config 2: pred_len=192 ==========
set pred_len=192
set seq_len=512
set xlstm_dropout=0.1
set lr=0.1
set batch_size=128
set init_token=3
set xlstm_embedding_dim=256
set slstm_num_heads=8
set slstm_conv1d_kernel_size=0
set xlstm_num_blocks=1
set gamma=0.98
set cosine_epochs=15
set warmup_epochs=5
set constant_gamma_epochs=2

echo.
echo [Config 2] pred_len=%pred_len%, seq_len=%seq_len%
echo.

for %%s in (2021 2022 2023) do (
    echo Running seed %%s...
    python -m xlstm_mixer fit+test ^
        --data ForecastingTSLibDataModule ^
        --data.dataset_name %dataset% ^
        --data.root_path "%DATASET_PATH%" ^
        --data.num_workers 0 ^
        --optimizer.lr %lr% ^
        --data.seq_len %seq_len% ^
        --data.pred_len %pred_len% ^
        --data.label_len 0 ^
        --data.batch_size %batch_size% ^
        --model LongTermForecastingExp ^
        --model.criterion torch.nn.L1Loss ^
        --model.architecture xLSTMMixer ^
        --model.architecture.num_mem_tokens %init_token% ^
        --model.architecture.xlstm_num_heads %slstm_num_heads% ^
        --model.architecture.xlstm_num_blocks %xlstm_num_blocks% ^
        --model.architecture.xlstm_embedding_dim %xlstm_embedding_dim% ^
        --model.architecture.xlstm_conv1d_kernel_size %slstm_conv1d_kernel_size% ^
        --model.architecture.xlstm_dropout %xlstm_dropout% ^
        --optimizer.lr %lr% ^
        --lr_scheduler.constant_gamma_epochs %constant_gamma_epochs% ^
        --lr_scheduler.gamma %gamma% ^
        --lr_scheduler.cosine_epochs %cosine_epochs% ^
        --lr_scheduler.warmup_epochs %warmup_epochs% ^
        --trainer.logger.name %dataset%_xlstm-mixer_%pred_len%_%%s ^
        --trainer.logger.project xlstm-mixer ^
        --trainer.max_epochs 40 ^
        --seed_everything %%s ^
        --trainer.fast_dev_run !dev_run!

    if errorlevel 1 (
        echo Error occurred with seed %%s
        exit /b 1
    )
)

REM ========== Config 3: pred_len=336 ==========
set pred_len=336
set seq_len=512
set xlstm_dropout=0.1
set lr=0.001
set batch_size=128
set init_token=1
set xlstm_embedding_dim=256
set slstm_num_heads=8
set slstm_conv1d_kernel_size=0
set xlstm_num_blocks=1
set gamma=0.98
set cosine_epochs=15
set warmup_epochs=5
set constant_gamma_epochs=2

echo.
echo [Config 3] pred_len=%pred_len%, seq_len=%seq_len%
echo.

for %%s in (2021 2022 2023) do (
    echo Running seed %%s...
    python -m xlstm_mixer fit+test ^
        --data ForecastingTSLibDataModule ^
        --data.dataset_name %dataset% ^
        --data.root_path "%DATASET_PATH%" ^
        --data.num_workers 0 ^
        --optimizer.lr %lr% ^
        --data.seq_len %seq_len% ^
        --data.pred_len %pred_len% ^
        --data.label_len 0 ^
        --data.batch_size %batch_size% ^
        --model LongTermForecastingExp ^
        --model.criterion torch.nn.L1Loss ^
        --model.architecture xLSTMMixer ^
        --model.architecture.num_mem_tokens %init_token% ^
        --model.architecture.xlstm_num_heads %slstm_num_heads% ^
        --model.architecture.xlstm_num_blocks %xlstm_num_blocks% ^
        --model.architecture.xlstm_embedding_dim %xlstm_embedding_dim% ^
        --model.architecture.xlstm_conv1d_kernel_size %slstm_conv1d_kernel_size% ^
        --model.architecture.xlstm_dropout %xlstm_dropout% ^
        --optimizer.lr %lr% ^
        --lr_scheduler.constant_gamma_epochs %constant_gamma_epochs% ^
        --lr_scheduler.gamma %gamma% ^
        --lr_scheduler.cosine_epochs %cosine_epochs% ^
        --lr_scheduler.warmup_epochs %warmup_epochs% ^
        --trainer.logger.name %dataset%_xlstm-mixer_%pred_len%_%%s ^
        --trainer.logger.project xlstm-mixer ^
        --trainer.max_epochs 40 ^
        --seed_everything %%s ^
        --trainer.fast_dev_run !dev_run!

    if errorlevel 1 (
        echo Error occurred with seed %%s
        exit /b 1
    )
)

REM ========== Config 4: pred_len=720 ==========
set pred_len=720
set seq_len=768
set xlstm_dropout=0.25
set lr=0.001
set batch_size=32
set init_token=2
set xlstm_embedding_dim=128
set slstm_num_heads=8
set slstm_conv1d_kernel_size=0
set xlstm_num_blocks=1
set gamma=0.98
set cosine_epochs=15
set warmup_epochs=5
set constant_gamma_epochs=2

echo.
echo [Config 4] pred_len=%pred_len%, seq_len=%seq_len%
echo.

for %%s in (2021 2022 2023) do (
    echo Running seed %%s...
    python -m xlstm_mixer fit+test ^
        --data ForecastingTSLibDataModule ^
        --data.dataset_name %dataset% ^
        --data.root_path "%DATASET_PATH%" ^
        --data.num_workers 0 ^
        --optimizer.lr %lr% ^
        --data.seq_len %seq_len% ^
        --data.pred_len %pred_len% ^
        --data.label_len 0 ^
        --data.batch_size %batch_size% ^
        --model LongTermForecastingExp ^
        --model.criterion torch.nn.L1Loss ^
        --model.architecture xLSTMMixer ^
        --model.architecture.num_mem_tokens %init_token% ^
        --model.architecture.xlstm_num_heads %slstm_num_heads% ^
        --model.architecture.xlstm_num_blocks %xlstm_num_blocks% ^
        --model.architecture.xlstm_embedding_dim %xlstm_embedding_dim% ^
        --model.architecture.xlstm_conv1d_kernel_size %slstm_conv1d_kernel_size% ^
        --model.architecture.xlstm_dropout %xlstm_dropout% ^
        --optimizer.lr %lr% ^
        --lr_scheduler.constant_gamma_epochs %constant_gamma_epochs% ^
        --lr_scheduler.gamma %gamma% ^
        --lr_scheduler.cosine_epochs %cosine_epochs% ^
        --lr_scheduler.warmup_epochs %warmup_epochs% ^
        --trainer.logger.name %dataset%_xlstm-mixer_%pred_len%_%%s ^
        --trainer.logger.project xlstm-mixer ^
        --trainer.max_epochs 40 ^
        --seed_everything %%s ^
        --trainer.fast_dev_run !dev_run!

    if errorlevel 1 (
        echo Error occurred with seed %%s
        exit /b 1
    )
)

echo.
echo ========================================
echo All experiments completed!
echo ========================================

endlocal
