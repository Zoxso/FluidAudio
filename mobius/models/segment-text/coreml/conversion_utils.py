import json
import os
from enum import IntEnum

import coremltools.optimize.coreml as cto_coreml


class Conversion(IntEnum):
    NONE = 0
    PRUNE = 1
    QUANTIZE = 2
    PALETTIZE = 3


def update_manifest_model_name(manifest_path: str, new_name: str) -> None:
    with open(manifest_path, "r") as file:
        manifest = json.load(file)

    for key, value in manifest["itemInfoEntries"].items():
        if value["name"] == "model.mlmodel":
            value["name"] = f"{new_name}.mlmodel"
            value["path"] = f"com.apple.CoreML/{new_name}.mlmodel"

    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=4)

    print(f"Manifest updated. Model name changed to {new_name}.mlmodel")

    old_model_path = os.path.join(
        os.path.dirname(manifest_path), "Data/com.apple.CoreML/model.mlmodel"
    )
    new_model_path = os.path.join(
        os.path.dirname(manifest_path),
        f"Data/com.apple.CoreML/{new_name}.mlmodel",
    )
    if os.path.exists(old_model_path):
        os.rename(old_model_path, new_model_path)
        print(f"Model file renamed from model.mlmodel to {new_name}.mlmodel")
    else:
        print("Warning: model.mlmodel not found. Only manifest was updated.")


def palettize_model(mlpackage, *, bits: int = 8, weight_threshold: int = 512):
    print(f"\nApplying {bits}-bit palettization...")
    try:
        op_config = cto_coreml.OpPalettizerConfig(
            nbits=bits,
            weight_threshold=weight_threshold,
        )
        config = cto_coreml.OptimizationConfig(op_config)
        return cto_coreml.palettize_weights(mlpackage, config)
    except Exception as e:
        print(f"Error palettization failed: {e}")
        return None


def prune_model(mlpackage, *, threshold: float = 0.01):
    print(f"\nApplying pruning quantization...")
    try:
        config = cto_coreml.OptimizationConfig(
            global_config=cto_coreml.OpThresholdPrunerConfig(threshold=threshold)
        )
        return cto_coreml.prune_weights(mlpackage, config)
    except Exception as e:
        print(f"Error pruning failed: {e}")
        return None


def quantize_model(mlpackage, *, dtype: str = "int8", mode: str = "linear_symmetric"):
    if str == "linear":
        print(f"\nApplying {dtype} quantization...")
    else:
        print("\nApplying mixed precision quantization...")

    try:
        op_config = cto_coreml.OpLinearQuantizerConfig(
            mode=mode,
            dtype=dtype,
            granularity="per_block",
            block_size=32,
        )

        config = cto_coreml.OptimizationConfig(global_config=op_config)
        return cto_coreml.linear_quantize_weights(mlpackage, config)
    except Exception as e:
        print(f"INT8 quantization failed: {e}")
        return None


def apply_conversion(mlpackage, conversion_type: Conversion):
    match conversion_type:
        case Conversion.NONE:
            return mlpackage
        case Conversion.PRUNE:
            return prune_model(mlpackage)
        case Conversion.QUANTIZE:
            return quantize_model(mlpackage)
        case Conversion.PALETTIZE:
            return palettize_model(mlpackage)
        case _:
            raise ValueError(f"Unsupported conversion type: {conversion_type}")
