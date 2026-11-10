@echo off
REM ============================================================
REM 步骤 3：合并 LoRA 权重为完整独立模型（可选）
REM 输出：outputs\medical_expert_merged（约 15GB）
REM 磁盘不足可跳过本步，WebUI 会直接以"基座+LoRA"方式加载
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
echo  [3/5] 合并 LoRA 权重
echo ============================================================
"%PYEXE%" inference\merge_lora.py

echo.
pause
