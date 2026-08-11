"""Calculate theoretical TinyColorCNN cost changes for RGB versus one-channel input."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn

from train_dl_models import TinyColorCNN


IMAGE_SIZE = 96


def build_model(input_channels: int) -> TinyColorCNN:
    return TinyColorCNN(input_channels=input_channels).eval()


def count_macs(model: nn.Module, input_channels: int) -> int:
    total = 0

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if isinstance(module, nn.Conv2d):
            kernel_operations = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.in_channels
                // module.groups
            )
            total += int(output.numel() * kernel_operations)
        elif isinstance(module, nn.Linear):
            total += int(output.numel() * module.in_features)

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    with torch.no_grad():
        model(torch.zeros(1, input_channels, IMAGE_SIZE, IMAGE_SIZE))
    for handle in handles:
        handle.remove()
    return total


def main() -> None:
    rows = []
    for input_channels in [3, 1]:
        model = build_model(input_channels)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        rows.append(
            {
                "input_channels": input_channels,
                "parameter_count": parameter_count,
                "float32_parameter_bytes": 4 * parameter_count,
                "input_tensor_bytes_per_patch": 4
                * input_channels
                * IMAGE_SIZE
                * IMAGE_SIZE,
                "multiply_accumulate_operations_per_patch": count_macs(
                    model, input_channels
                ),
            }
        )
    output = pd.DataFrame(rows)
    rgb = output.loc[output["input_channels"] == 3].iloc[0]
    one = output.loc[output["input_channels"] == 1].iloc[0]
    output["parameter_reduction_percent_vs_rgb"] = 100 * (
        1 - output["parameter_count"] / rgb["parameter_count"]
    )
    output["input_tensor_reduction_percent_vs_rgb"] = 100 * (
        1 - output["input_tensor_bytes_per_patch"] / rgb["input_tensor_bytes_per_patch"]
    )
    output["mac_reduction_percent_vs_rgb"] = 100 * (
        1
        - output["multiply_accumulate_operations_per_patch"]
        / rgb["multiply_accumulate_operations_per_patch"]
    )
    output_dir = Path("outputs/modeling/feature_efficiency")
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "cnn_channel_reduction_theoretical.csv", index=False)


if __name__ == "__main__":
    main()
