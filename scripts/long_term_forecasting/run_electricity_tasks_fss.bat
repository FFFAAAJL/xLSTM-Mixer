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

set WANDB_API_KEY=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
echo Login to WandB...
wandb login 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

:: === Config: pred_len=96 (FSS 50% Seed 42) ===
echo Running seed 42 (pred_len=96)...
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
    --lr_scheduler.cosine_epochs 20 ^
    --lr_scheduler.warmup_epochs 5 ^
    --trainer.logger.name Electricity_xlstm-mixer_96_42_FSS ^
    --trainer.logger.project xlstm-mixer ^
    --trainer.max_epochs 40 ^
    --seed_everything 42 ^
    --forecast_visualize_cb.pred_len 96 ^
    --forecast_visualize_cb.freq_epoch 5 ^
    --pruning_cb.target_sparsity 0.5 ^
    --pruning_cb.bank_size 4 ^
    --pruning_cb.warmup_epochs 10
if %errorlevel% neq 0 (
    echo !!! Command failed!
    pause
    exit /b %errorlevel%
)

:: === Config: pred_len=192 (FSS 50% Seed 42) ===
echo ^>^>^> Cleaning build artifacts for new embedding dim...
python -c "import shutil, os; p=os.path.join(os.environ.get('LOCALAPPDATA', ''), 'torch_extensions'); shutil.rmtree(p, ignore_errors=True) if p else None; shutil.rmtree('build', ignore_errors=True)"
echo Running seed 42 (pred_len=192)...
python -m xlstm_mixer fit+test ^
    --data ForecastingTSLibDataModule ^
    --data.dataset_name Electricity ^
    --data.root_path "E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets" ^
    --data.num_workers 0 ^
    --optimizer.lr 0.0005 ^
    --data.seq_len 768 ^
    --data.pred_len 192 ^
    --data.label_len 0 ^
    --data.batch_size 64 ^
    --model LongTermForecastingExp ^
    --model.criterion torch.nn.L1Loss ^
    --model.architecture xLSTMMixer ^
    --model.architecture.num_mem_tokens 3 ^
    --model.architecture.xlstm_num_heads 16 ^
    --model.architecture.xlstm_num_blocks 2 ^
    --model.architecture.xlstm_embedding_dim 768 ^
    --model.architecture.xlstm_conv1d_kernel_size 2 ^
    --model.architecture.xlstm_dropout 0.1 ^
    --lr_scheduler.constant_gamma_epochs 1 ^
    --lr_scheduler.gamma 0.99 ^
    --lr_scheduler.cosine_epochs 25 ^
    --lr_scheduler.warmup_epochs 5 ^
    --trainer.logger.name Electricity_xlstm-mixer_192_42_FSS ^
    --trainer.logger.project xlstm-mixer ^
    --trainer.max_epochs 40 ^
    --seed_everything 42 ^
    --forecast_visualize_cb.pred_len 192 ^
    --forecast_visualize_cb.freq_epoch 5 ^
    --pruning_cb.target_sparsity 0.5 ^
    --pruning_cb.bank_size 4 ^
    --pruning_cb.warmup_epochs 10
if %errorlevel% neq 0 (
    echo !!! Command failed!
    pause
    exit /b %errorlevel%
)

:: === Config: pred_len=336 (FSS 50% Seed 42) ===
echo Running seed 42 (pred_len=336)...
python -m xlstm_mixer fit+test ^
    --data ForecastingTSLibDataModule ^
    --data.dataset_name Electricity ^
    --data.root_path "E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets" ^
    --data.num_workers 0 ^
    --optimizer.lr 0.0005 ^
    --data.seq_len 768 ^
    --data.pred_len 336 ^
    --data.label_len 0 ^
    --data.batch_size 16 ^
    --model LongTermForecastingExp ^
    --model.criterion torch.nn.L1Loss ^
    --model.architecture xLSTMMixer ^
    --model.architecture.num_mem_tokens 4 ^
    --model.architecture.xlstm_num_heads 32 ^
    --model.architecture.xlstm_num_blocks 3 ^
    --model.architecture.xlstm_embedding_dim 768 ^
    --model.architecture.xlstm_conv1d_kernel_size 2 ^
    --model.architecture.xlstm_dropout 0.1 ^
    --lr_scheduler.constant_gamma_epochs 1 ^
    --lr_scheduler.gamma 0.99 ^
    --lr_scheduler.cosine_epochs 20 ^
    --lr_scheduler.warmup_epochs 7 ^
    --trainer.logger.name Electricity_xlstm-mixer_336_42_FSS ^
    --trainer.logger.project xlstm-mixer ^
    --trainer.max_epochs 40 ^
    --seed_everything 42 ^
    --forecast_visualize_cb.pred_len 336 ^
    --forecast_visualize_cb.freq_epoch 5 ^
    --pruning_cb.target_sparsity 0.5 ^
    --pruning_cb.bank_size 4 ^
    --pruning_cb.warmup_epochs 10
if %errorlevel% neq 0 (
    echo !!! Command failed!
    pause
    exit /b %errorlevel%
)

:: === Config: pred_len=720 (FSS 50% Seed 42) ===
echo Running seed 42 (pred_len=720)...
python -m xlstm_mixer fit+test ^
    --data ForecastingTSLibDataModule ^
    --data.dataset_name Electricity ^
    --data.root_path "E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets" ^
    --data.num_workers 0 ^
    --optimizer.lr 0.0005 ^
    --data.seq_len 768 ^
    --data.pred_len 720 ^
    --data.label_len 0 ^
    --data.batch_size 16 ^
    --model LongTermForecastingExp ^
    --model.criterion torch.nn.L1Loss ^
    --model.architecture xLSTMMixer ^
    --model.architecture.num_mem_tokens 3 ^
    --model.architecture.xlstm_num_heads 8 ^
    --model.architecture.xlstm_num_blocks 2 ^
    --model.architecture.xlstm_embedding_dim 768 ^
    --model.architecture.xlstm_conv1d_kernel_size 2 ^
    --model.architecture.xlstm_dropout 0.1 ^
    --lr_scheduler.constant_gamma_epochs 1 ^
    --lr_scheduler.gamma 0.99 ^
    --lr_scheduler.cosine_epochs 10 ^
    --lr_scheduler.warmup_epochs 15 ^
    --trainer.logger.name Electricity_xlstm-mixer_720_42_FSS ^
    --trainer.logger.project xlstm-mixer ^
    --trainer.max_epochs 40 ^
    --seed_everything 42 ^
    --forecast_visualize_cb.pred_len 720 ^
    --forecast_visualize_cb.freq_epoch 5 ^
    --pruning_cb.target_sparsity 0.5 ^
    --pruning_cb.bank_size 4 ^
    --pruning_cb.warmup_epochs 10
if %errorlevel% neq 0 (
    echo !!! Command failed!
    pause
    exit /b %errorlevel%
)

echo All FSS tasks (Seed 42) completed successfully.
pause
