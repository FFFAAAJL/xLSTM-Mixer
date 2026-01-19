import os
import sys

# ================= Configuration =================
WANDB_API_KEY = "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"
DATASET_NAME = "Electricity"
DATASET_PATH = r"E:\LSTM\xLSTM-Mixer-main\xLSTM-Mixer\common-ts\datasets\tslib_datasets"
SEEDS = [2021, 2022, 2023]
NUM_WORKERS = "0"
OUTPUT_BAT_FILE = "run_electricity_tasks.bat"
# =================================================

def generate_bat_script():
    # Define Python one-liner command to clean build artifacts (called directly in Windows bat)
    clean_cmd = (
        'python -c "import shutil, os; '
        'p=os.path.join(os.environ.get(\'LOCALAPPDATA\', \'\'), \'torch_extensions\'); '
        'shutil.rmtree(p, ignore_errors=True) if p else None; '
        'shutil.rmtree(\'build\', ignore_errors=True)"'
    )

    # Basic configuration
    configs = [
        # Config 1: pred_len=96
        {
            "pred_len": 96, "seq_len": 512, "batch_size": 32, "lr": 0.0005,
            "xlstm_dropout": 0.1, "init_token": 3, "embedding_dim": 1024,
            "num_heads": 16, "kernel_size": 4, "num_blocks": 2, 
            "gamma": 0.99, "cosine_epochs": 20, "warmup_epochs": 5, "const_gamma": 1
        },
        # Config 2: pred_len=192
        {
            "pred_len": 192, "seq_len": 768, "batch_size": 256, "lr": 0.0005,
            "xlstm_dropout": 0.1, "init_token": 3, "embedding_dim": 768,
            "num_heads": 16, "kernel_size": 2, "num_blocks": 2, 
            "gamma": 0.99, "cosine_epochs": 25, "warmup_epochs": 5, "const_gamma": 1
        },
        # Config 3: pred_len=336
        {
            "pred_len": 336, "seq_len": 768, "batch_size": 16, "lr": 0.0005,
            "xlstm_dropout": 0.1, "init_token": 4, "embedding_dim": 768,
            "num_heads": 32, "kernel_size": 2, "num_blocks": 3, 
            "gamma": 0.99, "cosine_epochs": 20, "warmup_epochs": 7, "const_gamma": 1
        },
        # Config 4: pred_len=720
        {
            "pred_len": 720, "seq_len": 768, "batch_size": 16, "lr": 0.0005,
            "xlstm_dropout": 0.1, "init_token": 3, "embedding_dim": 768,
            "num_heads": 8, "kernel_size": 2, "num_blocks": 2, 
            "gamma": 0.99, "cosine_epochs": 10, "warmup_epochs": 15, "const_gamma": 1
        }
    ]

    lines = []
    lines.append("@echo off")
    lines.append("setlocal enabledelayedexpansion")
    
    lines.append(":: Set up MSVC environment for CUDA compilation")
    lines.append(":: Check if MSVC environment is already set (cl.exe in path)")
    lines.append("where cl >nul 2>nul")
    lines.append("if %errorlevel% neq 0 (")
    lines.append("    echo Setting up MSVC environment...")
    lines.append("    set DISTUTILS_USE_SDK=1")
    lines.append("    set MSSdk=1")
    lines.append(r'    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"')
    lines.append(")")
    lines.append("")

    lines.append(f"set WANDB_API_KEY={WANDB_API_KEY}")
    lines.append("echo Login to WandB...")
    lines.append(f"wandb login {WANDB_API_KEY}")
    lines.append("")

    last_embedding_dim = None

    for cfg in configs:
        lines.append(f":: === Config: pred_len={cfg['pred_len']} ===")
        
        # If embedding_dim changes, insert cleanup command
        if last_embedding_dim is not None and cfg['embedding_dim'] != last_embedding_dim:
            # Fix: Remove >>> symbol to avoid being parsed as redirection by cmd
            lines.append("echo [INFO] Cleaning build artifacts for new embedding dim...")
            lines.append(clean_cmd)
        
        last_embedding_dim = cfg['embedding_dim']

        for seed in SEEDS:
            # Use caret (^) to split long command into multiple lines for readability and to avoid single line too long
            parts = [
                f"python -m xlstm_mixer fit+test",
                f"--data ForecastingTSLibDataModule",
                f"--data.dataset_name {DATASET_NAME}",
                f"--data.root_path \"{DATASET_PATH}\"",
                f"--data.num_workers {NUM_WORKERS}",
                f"--optimizer.lr {cfg['lr']}",
                f"--data.seq_len {cfg['seq_len']}",
                f"--data.pred_len {cfg['pred_len']}",
                f"--data.label_len 0",
                f"--data.batch_size {cfg['batch_size']}",
                f"--model LongTermForecastingExp",
                f"--model.criterion torch.nn.L1Loss",
                f"--model.architecture xLSTMMixer",
                f"--model.architecture.num_mem_tokens {cfg['init_token']}",
                f"--model.architecture.xlstm_num_heads {cfg['num_heads']}",
                f"--model.architecture.xlstm_num_blocks {cfg['num_blocks']}",
                f"--model.architecture.xlstm_embedding_dim {cfg['embedding_dim']}",
                f"--model.architecture.xlstm_conv1d_kernel_size {cfg['kernel_size']}",
                f"--model.architecture.xlstm_dropout {cfg['xlstm_dropout']}",
                f"--lr_scheduler.constant_gamma_epochs {cfg['const_gamma']}",
                f"--lr_scheduler.gamma {cfg['gamma']}",
                f"--lr_scheduler.cosine_epochs {cfg['cosine_epochs']}",
                f"--lr_scheduler.warmup_epochs {cfg['warmup_epochs']}",
                f"--trainer.logger.name {DATASET_NAME}_xlstm-mixer_{cfg['pred_len']}_{seed}",
                f"--trainer.logger.project xlstm-mixer",
                f"--trainer.max_epochs 40",
                f"--seed_everything {seed}",
                f"--forecast_visualize_cb.pred_len {cfg['pred_len']}",
                f"--forecast_visualize_cb.freq_epoch 5"
            ]
            
            # Use ^ for line continuation
            cmd = " ^\n    ".join(parts)
            
            lines.append(f"echo Running seed {seed}...")
            lines.append(cmd)
            # Add error check, pause if error occurs for easy viewing
            lines.append("if %errorlevel% neq 0 (")
            lines.append("    echo !!! Command failed!")
            lines.append("    pause")
            lines.append("    exit /b %errorlevel%")
            lines.append(")")
            lines.append("")

    lines.append("echo All tasks completed successfully.")
    lines.append("pause")

    with open(OUTPUT_BAT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f">>> Successfully generated: {OUTPUT_BAT_FILE}")
    print(">>> Please run this batch file directly in your configured terminal.")

if __name__ == "__main__":
    generate_bat_script()
