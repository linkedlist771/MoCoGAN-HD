import torch
import argparse

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Compare two model shapes")
    parser.add_argument('--model0', type=str, required=True, help='Path to the first model (.pt) file')
    parser.add_argument('--model1', type=str, required=True, help='Path to the second model (.pt) file')
    return parser.parse_args()

# 加载并打印模型结构
def print_model_shapes(model, model_name):
    print(f"Shapes of layers in {model_name}:")
    for name, param in model.items():
        print(f"{name}: {param.shape}")

# 主程序
def main():
    # 获取命令行参数
    args = parse_args()

    # 加载两个模型
    model_1 = torch.load(args.model0)
    model_2 = torch.load(args.model1)
    print(f"model2 keys")
    print(model_2.keys())
    model_2 = model_2["g_ema"]
    # 打印两个模型的结构
    print(f"model_1")
    print_model_shapes(model_1, args.model0)
    print(f"model_2")
    print_model_shapes(model_2, args.model1)

if __name__ == "__main__":
    main()
