@echo off
setlocal enabledelayedexpansion
:: ============================================
:: 运行ETTh2数据集所有稀疏模式实验的主脚本
:: 包括：无稀疏、2:4、1:4、4:8、2:8
:: ============================================

echo ============================================
echo Starting All Sparsity Experiments for ETTh2
echo ============================================
echo.

:: 记录开始时间
set START_TIME=%time%
echo Start Time: %date% %START_TIME%
echo.

:: ============================================
:: 实验1: 无稀疏版本 (Baseline)
:: ============================================
echo [1/6] Running No Pruning (Baseline)...
echo ----------------------------------------
call "%~dp0run_etth2_tasks_no_pruning.bat"
if %errorlevel% neq 0 (
    echo ERROR: Experiment 1 ^(No Pruning^) failed
    pause
    exit /b %errorlevel%
)
echo [1/6] No Pruning completed successfully.
echo.

:: ============================================
:: 实验2: 2:4 稀疏模式
:: ============================================
echo [2/6] Running 2:4 Sparsity...
echo ----------------------------------------
call "%~dp0run_etth2_tasks_fss_2_4.bat"
if %errorlevel% neq 0 (
    echo ERROR: Experiment 2 ^(2:4^) failed
    pause
    exit /b %errorlevel%
)
echo [2/6] 2:4 Sparsity completed successfully.
echo.

:: ============================================
:: 实验3: 1:4 稀疏模式
:: ============================================
echo [3/6] Running 1:4 Sparsity...
echo ----------------------------------------
call "%~dp0run_etth2_tasks_fss_1_4.bat"
if %errorlevel% neq 0 (
    echo ERROR: Experiment 3 ^(1:4^) failed
    pause
    exit /b %errorlevel%
)
echo [3/6] 1:4 Sparsity completed successfully.
echo.

:: ============================================
:: 实验4: 4:8 稀疏模式
:: ============================================
echo [4/6] Running 4:8 Sparsity...
echo ----------------------------------------
call "%~dp0run_etth2_tasks_fss_4_8.bat"
if %errorlevel% neq 0 (
    echo ERROR: Experiment 4 ^(4:8^) failed
    pause
    exit /b %errorlevel%
)
echo [4/6] 4:8 Sparsity completed successfully.
echo.

:: ============================================
:: 实验5: 2:8 稀疏模式
:: ============================================
echo [5/6] Running 2:8 Sparsity...
echo ----------------------------------------
call "%~dp0run_etth2_tasks_fss_2_8.bat"
if %errorlevel% neq 0 (
    echo ERROR: Experiment 5 ^(2:8^) failed
    pause
    exit /b %errorlevel%
)
echo [5/6] 2:8 Sparsity completed successfully.
echo.

:: ============================================
:: 实验6: 高稀疏模式（1:16，约93.75%）
:: ============================================
echo [6/6] Running High Sparsity (1:16, ~93.75%%)...
echo ----------------------------------------
call "%~dp0run_etth2_tasks_fss_1_16.bat"
if %errorlevel% neq 0 (
    echo ERROR: Experiment 6 ^(High Sparsity, 1:16^) failed
    pause
    exit /b %errorlevel%
)
echo [6/6] High Sparsity completed successfully.
echo.

:: ============================================
:: 所有实验完成
:: ============================================
set END_TIME=%time%
echo ============================================
echo All Experiments Completed for ETTh2!
echo ============================================
echo Start Time: %date% %START_TIME%
echo End Time: %date% %END_TIME%
echo.
echo Summary:
echo   - No Pruning (Baseline)
echo   - 2:4 Sparsity (50%%, bank_size=4)
echo   - 1:4 Sparsity (75%%, bank_size=4)
echo   - 4:8 Sparsity (50%%, bank_size=8)
echo   - 2:8 Sparsity (75%%, bank_size=8)
echo   - High Sparsity (approx 1:16, bank_size=16)
echo.
echo All results are logged to WandB project: xlstm-mixer
echo You can compare the results in WandB dashboard.
echo ============================================
pause
