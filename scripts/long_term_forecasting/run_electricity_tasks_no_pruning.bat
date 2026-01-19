@echo off
setlocal enabledelayedexpansion
:: Set up MSVC environment for CUDA compilation
:: Check if MSVC environment is already set (cl.exe in path)
where cl >nul 2>nul
if %errorlevel% neq 0 (
    echo Setting up MSVC environment...
    set DISTUTILS_USE_SDK=1
    set MSSdk=1
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

if defined WANDB_API_KEY (
    echo Login to WandB...
    wandb login %WANDB_API_KEY%
) else (
    echo WANDB_API_KEY not set; skipping wandb login.
)

:: === Config: pred_len=96 (No Pruning, Seed 42) ===
:: 无稀疏版本：不进行任何剪枝，作为baseline对比
echo Running seed 42 (pred_len=96, No Pruning)...
python -m xlstm_mixer fit+test ^
    --data ForecastingTSLibDataModule ^
    --data.dataset_name Electricity ^
    --data.root_path "E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets" ^
    --data.num_workers 0 ^
    --optimizer.lr 0.0005 ^
    --data.seq_len 512 ^
    --data.pred_len 96 ^
    --data.label_len 0 ^
    --data.batch_size 32 ^
    --model LongTermForecastingExp ^
    --model.criterion torch.nn.L1Loss ^
    --model.architecture xLSTMMixer ^
    --model.architecture.num_mem_tokens 3 ^
    --model.architecture.xlstm_num_heads 16 ^
    --model.architecture.xlstm_num_blocks 2 ^
    --model.architecture.xlstm_embedding_dim 1024 ^
    --model.architecture.xlstm_conv1d_kernel_size 4 ^
    --model.architecture.xlstm_dropout 0.1 ^
    --lr_scheduler.constant_gamma_epochs 1 ^
    --lr_scheduler.gamma 0.99 ^
    --lr_scheduler.cosine_epochs 10 ^
    --lr_scheduler.warmup_epochs 15 ^
    --trainer.logger.name Electricity_xlstm-mixer_96_42_NoPruning ^
    --trainer.logger.project xlstm-mixer ^
    --trainer.max_epochs 30 ^
    --seed_everything 42 ^
    --forecast_visualize_cb.pred_len 96 ^
    --forecast_visualize_cb.freq_epoch 1 ^
    --pruning_cb.target_sparsity 0.0
if %errorlevel% neq 0 (
    echo ERROR: Command failed
    pause
    exit /b %errorlevel%
)


echo All tasks (No Pruning, Seed 42) completed successfully.
pause
