"""Small standalone shape smoke test for AIC, GCE and MSA."""
import torch
from modules import AIC, GCE, MSA


def main() -> None:
    x = torch.randn(1, 64, 80, 80)
    print("AIC:", tuple(AIC(64)(x).shape))
    print("GCE:", tuple(GCE(64)(x).shape))
    msa = MSA([64, 128])
    y = msa([x, torch.randn(1, 128, 40, 40)])
    print("MSA:", tuple(y.shape))


if __name__ == "__main__":
    main()
