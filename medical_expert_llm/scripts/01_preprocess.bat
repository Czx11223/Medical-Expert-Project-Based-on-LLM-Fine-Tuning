@echo off
REM ============================================================
REM 步骤 1：数据预处理（medical_cot_zh.jsonl -> 训练格式）
REM 输出：data\processed\train.json / val.json
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
echo  [1/5] 数据预处理
echo ============================================================
"%PYEXE%" data\preprocess.py

echo.
pause
