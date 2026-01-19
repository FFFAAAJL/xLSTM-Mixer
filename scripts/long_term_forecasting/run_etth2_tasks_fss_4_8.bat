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

:: === Config: pred_len=96 (FSS 4:8 Sparsity 50% Seed 42) ===
:: 4:8模式：每8个元素保留4个，稀疏度50%
echo Running seed 42 (pred_len=96, FSS 4:8 sparsity)...
python -m xlstm_mixer fit+test ^
    --data ForecastingTSLibDataModule ^
    --data.dataset_name ETTh2 ^
    --data.root_path "E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets" ^
    --data.num_workers 0 ^
    --optimizer.lr 0.001 ^
    --data.seq_len 768 ^
    --data.pred_len 96 ^
    --data.label_len 0 ^
    --data.batch_size 128 ^
    --model LongTermForecastingExp ^
    --model.criterion torch.nn.L1Loss ^
    --model.architecture xLSTMMixer ^
    --model.architecture.num_mem_tokens 1 ^
    --model.architecture.xlstm_num_heads 8 ^
    --model.architecture.xlstm_num_blocks 1 ^
    --model.architecture.xlstm_embedding_dim 128 ^
    --model.architecture.xlstm_conv1d_kernel_size 0 ^
    --model.architecture.xlstm_dropout 0.1 ^
    --lr_scheduler.constant_gamma_epochs 2 ^
    --lr_scheduler.gamma 0.98 ^
    --lr_scheduler.cosine_epochs 15 ^
    --lr_scheduler.warmup_epochs 5 ^
    --trainer.logger.name ETTh2_xlstm-mixer_96_42_FSS_4_8 ^
    --trainer.logger.project xlstm-mixer ^
    --trainer.max_epochs 40 ^
    --seed_everything 42 ^
    --forecast_visualize_cb.pred_len 96 ^
    --forecast_visualize_cb.freq_epoch 1 ^
    --pruning_cb.target_sparsity 0.5 ^
    --pruning_cb.bank_size 8 ^
    --pruning_cb.warmup_epochs 10
if %errorlevel% neq 0 (
    echo ERROR: Command failed
    pause
    exit /b %errorlevel%
)


echo All FSS tasks (4:8 sparsity, Seed 42) completed successfully.
exit /b 0
