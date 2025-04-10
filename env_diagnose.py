import sys
import os
import subprocess
import platform
import pkg_resources

def run(cmd):
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return "", str(e)

def check_environment():
    print("=== Python 环境诊断工具 ===\n")

    print(f"[系统平台] {platform.system()} {platform.release()}")
    print(f"[Python 版本] {platform.python_version()} (路径: {sys.executable})\n")

    # 检查 pip 是否可用
    pip_out, pip_err = run([sys.executable, "-m", "pip", "--version"])
    if pip_err:
        print("[错误] pip 无法正常工作：", pip_err)
    else:
        print("[pip 检查] ", pip_out)

    # 检查常用模块是否安装
    common_packages = ["jupyter", "ipykernel", "notebook", "debugpy"]
    print("\n[模块依赖检查]")
    for pkg in common_packages:
        try:
            dist = pkg_resources.get_distribution(pkg)
            print(f"✔ {pkg}：已安装（版本 {dist.version}）")
        except pkg_resources.DistributionNotFound:
            print(f"✘ {pkg}：未安装")

    # 检查 VS Code 调试支持模块
    print("\n[调试模块检查]")
    try:
        import debugpy
        print("✔ debugpy 模块存在")
    except ImportError:
        print("✘ debugpy 模块缺失，VS Code 调试功能将异常")

    # 检查内核是否注册
    print("\n[Jupyter 内核检查]")
    jupyter_out, jupyter_err = run(["jupyter", "kernelspec", "list"])
    if jupyter_err:
        print("[错误] 获取内核列表失败：", jupyter_err)
    else:
        print(jupyter_out)

    print("\n=== 检查完成 ===")

if __name__ == "__main__":
    check_environment()
