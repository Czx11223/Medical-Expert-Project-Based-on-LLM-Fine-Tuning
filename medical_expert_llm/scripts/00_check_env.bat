@echo off
REM ============================================================
REM 步骤 0：环境自检（训练前必跑）
REM 用法：双击运行，或在命令行执行 scripts\00_check_env.bat
REM ============================================================
REM 防止 Windows 控制台中文乱码

REM 回到项目根目录（bat 位于 scripts\ 下，取上一级）
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
echo  [0/5] 环境自检
echo ============================================================
"%PYEXE%" scripts\00_check_env.py

echo.
pause
