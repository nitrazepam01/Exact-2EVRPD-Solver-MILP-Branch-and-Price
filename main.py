import os
import sys
import subprocess
import questionary

def main():
    print("="*50)
    print(" 2E-VRP-D Exact Algorithm Solver")
    print("="*50)

    # 1. 让用户选择算法
    algo_choice = questionary.select(
        "请选择要运行的求解算法 (Choose the algorithm):",
        choices=[
            "Branch and Price (精确算法)",
            "MILP (CPLEX 数学模型)"
        ]
    ).ask()

    # 检查用户是否按了 Ctrl+C 退出
    if not algo_choice:
        print("已取消运行。")
        sys.exit(0)

    # 2. 自动扫描 data 目录下的所有 txt 文件
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"错误: 找不到 {data_dir} 文件夹，请确保测试数据存放正确。")
        sys.exit(1)

    data_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    if not data_files:
        print(f"错误: 在 {data_dir} 文件夹中没有找到任何 .txt 算例文件。")
        sys.exit(1)

    # 3. 让用户选择算例
    data_choice = questionary.select(
        "请选择要测试的算例 (Choose the dataset):",
        choices=data_files
    ).ask()

    if not data_choice:
        print("已取消运行。")
        sys.exit(0)

    data_path = os.path.join(data_dir, data_choice)

    # 4. 构建并执行对应的命令
    print("\n" + "-"*50)
    print(f"正在启动求解器...")
    print(f"算法: {algo_choice}")
    print(f"算例: {data_path}")
    print("-"*50 + "\n")

    try:
        if "MILP" in algo_choice:
            # 调用 MILP.py，并把选择的算例路径传进去
            subprocess.run([sys.executable, "src/MILP.py", data_path], check=True)
        elif "Branch and Price" in algo_choice:
            # 调用 branch_and_bound.py
            subprocess.run([sys.executable, "src/branch_and_bound.py", data_path], check=True)
            
    except subprocess.CalledProcessError as e:
        print(f"\n[运行出错] 脚本执行中断，错误码: {e.returncode}")
    except KeyboardInterrupt:
        print("\n[用户终止] 运行已手动停止。")

if __name__ == "__main__":
    main()