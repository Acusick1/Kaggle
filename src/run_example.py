import torch
import argparse
from linear_regression import run_linear_reg

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run various PyTorch examples with CPU or GPU')

    parser.add_argument('example', nargs='?', default="linear")
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')

    args = parser.parse_args()
    args.device = None

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    print(f"Running model on: {str(args.device).upper()}")

    if args.example == "linear":
        run_linear_reg(args.device)
