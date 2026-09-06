"""Run with: python example.py --device cpu (or --device cuda)."""

import argparse

import torch

from modules import AIC, GCE, MSA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)

    # Input/output layout: batch, channels, height, width.
    x = torch.randn(2, 64, 32, 48, device=device)
    for module in (AIC(64), GCE(64)):
        module = module.to(device)
        output = module(x)
        output.square().mean().backward()
        print(f'{type(module).__name__}: {tuple(x.shape)} -> {tuple(output.shape)}; backward OK')

    # Typical YOLO11n neck channels for P3, P4, P5.
    p3 = torch.randn(2, 64, 32, 48, device=device)
    p4 = torch.randn(2, 128, 16, 24, device=device)
    p5 = torch.randn(2, 256, 8, 12, device=device)
    msa_p3 = MSA([64, 128]).to(device)
    msa_p4 = MSA([128, 64, 256]).to(device)
    msa_p5 = MSA([256, 128]).to(device)

    # Each branch reads the ORIGINAL neck outputs. Current scale always goes first.
    q3 = msa_p3([p3, p4])
    q4 = msa_p4([p4, p3, p5])
    q5 = msa_p5([p5, p4])
    sum(q.square().mean() for q in (q3, q4, q5)).backward()
    print(f'MSA P3/P4/P5: {tuple(q3.shape)}, {tuple(q4.shape)}, {tuple(q5.shape)}; backward OK')


if __name__ == '__main__':
    main()
