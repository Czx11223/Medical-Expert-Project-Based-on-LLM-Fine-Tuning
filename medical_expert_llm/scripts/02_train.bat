@echo off
REM ============================================================
REM 步骤 2：LoRA 4bit 量化微调训练（核心步骤，耗时最长）
REM 参数在 configs\train_config.yaml 中修改
REM 快速验证流程：python train\train_lora.py --max_train_samples 200
REM ============================================================
cd /d "%~dp0.."

REM 强制使用项目 venv 的 python（系统 PATH 里的 python 未装依赖）
set PYEXE=%~dp0..\venv\Scripts\python.exe
if not exist "%PYEXE%" (
    echo [错误] 未找到项目虚拟环境 venv\Scripts\python.exe
    echo        请先完成环境安装（见 README.md 第 2 节）
    pause
    exit /b 1
)

echo ============================================================
echo  [2/5] LoRA 微调训练
echo  提示：正式训练已迁移至 ModelScope 云端（见 modelscope_migration\）
echo ============================================================
"%PYEXE%" train\train_lora.py

echo.
echo 训练结束，产物位于 outputs\lora_medical_qwen\final_adapter
pause
