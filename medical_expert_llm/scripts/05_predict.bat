@echo off
REM ============================================================
REM 步骤 5：命令行推理测试（交互式问答）
REM 用于训练后快速验证模型效果，不启动网页
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
echo  [5/5] 命令行推理测试
echo ============================================================
"%PYEXE%" inference\predict.py

pause
