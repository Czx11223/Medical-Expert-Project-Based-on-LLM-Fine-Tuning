@echo off
REM ============================================================
REM 步骤 4：启动医疗问诊 Web 界面（Gradio）
REM 启动后自动打开浏览器 http://127.0.0.1:7860
REM 毕设演示：点页面下方示例问题即可快速演示
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
echo  [4/5] 启动问诊 Web 服务
echo  首次加载模型约需 1 分钟，请耐心等待浏览器自动打开
echo ============================================================
"%PYEXE%" app\app.py --port 7860

pause
