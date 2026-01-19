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

echo.
echo === Electricity pruning ablation (pred_len=96, seed=42, bank_size=4, target_sparsity=0.5) ===
echo Groups:
echo   G0 baseline (no pruning)
echo   G1 gates only (igate/fgate/ogate/zgate)
echo   G2 recurrent only (_recurrent_kernel_)
echo   G3 proj only (proj_up/proj_down)
echo   G4 external linear only (mlp_in/mlp_in_trend/pre_encoding/fc/Linear)
echo   G5 all (most weights + recurrent)
echo.

:: -------------------- Group 0 --------------------
echo Running G0 baseline (no pruning)...
python -m xlstm_mixer fit+test --data ForecastingTSLibDataModule --data.dataset_name Electricity --data.root_path "E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets" --data.num_workers 0 --optimizer.lr 0.0005 --data.seq_len 512 --data.pred_len 96 --data.label_len 0 --data.batch_size 32 --model LongTermForecastingExp --model.criterion torch.nn.L1Loss --model.architecture xLSTMMixer --model.architecture.num_mem_tokens 3 --model.architecture.xlstm_num_heads 16 --model.architecture.xlstm_num_blocks 2 --model.architecture.xlstm_embedding_dim 1024 --model.architecture.xlstm_conv1d_kernel_size 4 --model.architecture.xlstm_dropout 0.1 --lr_scheduler.constant_gamma_epochs 1 --lr_scheduler.gamma 0.99 --lr_scheduler.cosine_epochs 10 --lr_scheduler.warmup_epochs 15 --trainer.logger.name Electricity_xlstm-mixer_96_42_ABL_G0_NoPruning --trainer.logger.project xlstm-mixer --trainer.max_epochs 30 --seed_everything 42 --forecast_visualize_cb.pred_len 96 --forecast_visualize_cb.freq_epoch 1 --pruning_cb.target_sparsity 0.0
if %errorlevel% neq 0 (
    echo ERROR: G0 failed
    exit /b %errorlevel%
)

:: -------------------- Group 1 --------------------
echo Running G1 gates only...
python -m xlstm_mixer fit+test --data ForecastingTSLibDataModule --data.dataset_name Electricity --data.root_path "E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets" --data.num_workers 0 --optimizer.lr 0.0005 --data.seq_len 512 --data.pred_len 96 --data.label_len 0 --data.batch_size 32 --model LongTermForecastingExp --model.criterion torch.nn.L1Loss --model.architecture xLSTMMixer --model.architecture.num_mem_tokens 3 --model.architecture.xlstm_num_heads 16 --model.architecture.xlstm_num_blocks 2 --model.architecture.xlstm_embedding_dim 1024 --model.architecture.xlstm_conv1d_kernel_size 4 --model.architecture.xlstm_dropout 0.1 --lr_scheduler.constant_gamma_epochs 1 --lr_scheduler.gamma 0.99 --lr_scheduler.cosine_epochs 10 --lr_scheduler.warmup_epochs 15 --trainer.logger.name Electricity_xlstm-mixer_96_42_ABL_G1_GatesOnly_FSS_2_4 --trainer.logger.project xlstm-mixer --trainer.max_epochs 30 --seed_everything 42 --forecast_visualize_cb.pred_len 96 --forecast_visualize_cb.freq_epoch 1 --pruning_cb.target_sparsity 0.5 --pruning_cb.bank_size 4 --pruning_cb.warmup_epochs 10 --pruning_cb.blocks_only True --pruning_cb.target_modules igate,fgate,ogate,zgate
if %errorlevel% neq 0 (
    echo ERROR: G1 failed
    exit /b %errorlevel%
)

:: -------------------- Group 2 --------------------
echo Running G2 recurrent only...
python -m xlstm_mixer fit+test --data ForecastingTSLibDataModule --data.dataset_name Electricity --data.root_path "E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets" --data.num_workers 0 --optimizer.lr 0.0005 --data.seq_len 512 --data.pred_len 96 --data.label_len 0 --data.batch_size 32 --model LongTermForecastingExp --model.criterion torch.nn.L1Loss --model.architecture xLSTMMixer --model.architecture.num_mem_tokens 3 --model.architecture.xlstm_num_heads 16 --model.architecture.xlstm_num_blocks 2 --model.architecture.xlstm_embedding_dim 1024 --model.architecture.xlstm_conv1d_kernel_size 4 --model.architecture.xlstm_dropout 0.1 --lr_scheduler.constant_gamma_epochs 1 --lr_scheduler.gamma 0.99 --lr_scheduler.cosine_epochs 10 --lr_scheduler.warmup_epochs 15 --trainer.logger.name Electricity_xlstm-mixer_96_42_ABL_G2_RecurrentOnly_FSS_2_4 --trainer.logger.project xlstm-mixer --trainer.max_epochs 30 --seed_everything 42 --forecast_visualize_cb.pred_len 96 --forecast_visualize_cb.freq_epoch 1 --pruning_cb.target_sparsity 0.5 --pruning_cb.bank_size 4 --pruning_cb.warmup_epochs 10 --pruning_cb.blocks_only True --pruning_cb.target_modules _recurrent_kernel_
if %errorlevel% neq 0 (
    echo ERROR: G2 failed
    exit /b %errorlevel%
)

:: -------------------- Group 3 --------------------
echo Running G3 proj only...
python -m xlstm_mixer fit+test --data ForecastingTSLibDataModule --data.dataset_name Electricity --data.root_path "E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets" --data.num_workers 0 --optimizer.lr 0.0005 --data.seq_len 512 --data.pred_len 96 --data.label_len 0 --data.batch_size 32 --model LongTermForecastingExp --model.criterion torch.nn.L1Loss --model.architecture xLSTMMixer --model.architecture.num_mem_tokens 3 --model.architecture.xlstm_num_heads 16 --model.architecture.xlstm_num_blocks 2 --model.architecture.xlstm_embedding_dim 1024 --model.architecture.xlstm_conv1d_kernel_size 4 --model.architecture.xlstm_dropout 0.1 --lr_scheduler.constant_gamma_epochs 1 --lr_scheduler.gamma 0.99 --lr_scheduler.cosine_epochs 10 --lr_scheduler.warmup_epochs 15 --trainer.logger.name Electricity_xlstm-mixer_96_42_ABL_G3_ProjOnly_FSS_2_4 --trainer.logger.project xlstm-mixer --trainer.max_epochs 30 --seed_everything 42 --forecast_visualize_cb.pred_len 96 --forecast_visualize_cb.freq_epoch 1 --pruning_cb.target_sparsity 0.5 --pruning_cb.bank_size 4 --pruning_cb.warmup_epochs 10 --pruning_cb.blocks_only True --pruning_cb.target_modules proj_up,proj_down --pruning_cb.exclude_keywords bias,norm
if %errorlevel% neq 0 (
    echo ERROR: G3 failed
    exit /b %errorlevel%
)

:: -------------------- Group 4 --------------------
echo Running G4 external linear only...
python -m xlstm_mixer fit+test --data ForecastingTSLibDataModule --data.dataset_name Electricity --data.root_path "E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets" --data.num_workers 0 --optimizer.lr 0.0005 --data.seq_len 512 --data.pred_len 96 --data.label_len 0 --data.batch_size 32 --model LongTermForecastingExp --model.criterion torch.nn.L1Loss --model.architecture xLSTMMixer --model.architecture.num_mem_tokens 3 --model.architecture.xlstm_num_heads 16 --model.architecture.xlstm_num_blocks 2 --model.architecture.xlstm_embedding_dim 1024 --model.architecture.xlstm_conv1d_kernel_size 4 --model.architecture.xlstm_dropout 0.1 --lr_scheduler.constant_gamma_epochs 1 --lr_scheduler.gamma 0.99 --lr_scheduler.cosine_epochs 10 --lr_scheduler.warmup_epochs 15 --trainer.logger.name Electricity_xlstm-mixer_96_42_ABL_G4_ExternalLinearOnly_FSS_2_4 --trainer.logger.project xlstm-mixer --trainer.max_epochs 30 --seed_everything 42 --forecast_visualize_cb.pred_len 96 --forecast_visualize_cb.freq_epoch 1 --pruning_cb.target_sparsity 0.5 --pruning_cb.bank_size 4 --pruning_cb.warmup_epochs 10 --pruning_cb.blocks_only False --pruning_cb.target_modules weight --pruning_cb.include_scopes model.mlp_in,model.mlp_in_trend,model.pre_encoding,model.fc,model.Linear --pruning_cb.exclude_keywords bias,norm
if %errorlevel% neq 0 (
    echo ERROR: G4 failed
    exit /b %errorlevel%
)

:: -------------------- Group 5 --------------------
echo Running G5 all...
python -m xlstm_mixer fit+test --data ForecastingTSLibDataModule --data.dataset_name Electricity --data.root_path "E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets" --data.num_workers 0 --optimizer.lr 0.0005 --data.seq_len 512 --data.pred_len 96 --data.label_len 0 --data.batch_size 32 --model LongTermForecastingExp --model.criterion torch.nn.L1Loss --model.architecture xLSTMMixer --model.architecture.num_mem_tokens 3 --model.architecture.xlstm_num_heads 16 --model.architecture.xlstm_num_blocks 2 --model.architecture.xlstm_embedding_dim 1024 --model.architecture.xlstm_conv1d_kernel_size 4 --model.architecture.xlstm_dropout 0.1 --lr_scheduler.constant_gamma_epochs 1 --lr_scheduler.gamma 0.99 --lr_scheduler.cosine_epochs 10 --lr_scheduler.warmup_epochs 15 --trainer.logger.name Electricity_xlstm-mixer_96_42_ABL_G5_All_FSS_2_4 --trainer.logger.project xlstm-mixer --trainer.max_epochs 30 --seed_everything 42 --forecast_visualize_cb.pred_len 96 --forecast_visualize_cb.freq_epoch 1 --pruning_cb.target_sparsity 0.5 --pruning_cb.bank_size 4 --pruning_cb.warmup_epochs 10 --pruning_cb.blocks_only False --pruning_cb.target_modules weight,_recurrent_kernel_ --pruning_cb.exclude_keywords bias,norm
if %errorlevel% neq 0 (
    echo ERROR: G5 failed
    exit /b %errorlevel%
)

echo.
echo All pruning ablation groups completed successfully.
exit /b 0

