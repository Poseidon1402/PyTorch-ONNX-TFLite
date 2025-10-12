"""Utility script to convert the EdgeFace checkpoint to ONNX and/or TFLite."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping, Tuple

import torch

from backbones import get_model


def _parse_shape(shape_str: str) -> Tuple[int, ...]:
    try:
        dims = tuple(int(part.strip()) for part in shape_str.split(",") if part.strip())
    except ValueError as exc:
        raise SystemExit(f"Invalid shape string '{shape_str}'. Use comma separated integers.") from exc
    if not dims:
        raise SystemExit("Input shape must contain at least one dimension.")
    return dims


def _load_edgeface_model(
    checkpoint_path: Path,
    backbone: str,
) -> torch.nn.Module:
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint_obj, torch.nn.Module):
        model = checkpoint_obj
        if hasattr(model, "eval"):
            model.eval()
        return model

    if isinstance(checkpoint_obj, Mapping):
        state_dict = checkpoint_obj.get("state_dict", checkpoint_obj)
        model = get_model(backbone)

        # Remove common prefixes such as "module." or "model."
        if all(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        if all(key.startswith("model.") for key in state_dict.keys()):
            state_dict = {key[len("model."):]: value for key, value in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys while loading state_dict: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys while loading state_dict: {unexpected}")

        if hasattr(model, "eval"):
            model.eval()
        return model

    raise SystemExit(
        "Checkpoint must be a TorchScript module or a state_dict mapping."
    )


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_shape: Tuple[int, ...],
    opset: int,
    overwrite: bool,
    backbone: str,
) -> Path:
    if output_path.exists() and not overwrite:
        raise SystemExit(f"ONNX file already exists: {output_path}. Use --overwrite to replace it.")

    model = _load_edgeface_model(checkpoint_path, backbone=backbone)

    dummy_input = torch.randn(input_shape)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset,
    )

    return output_path


def export_tflite(
    onnx_path: Path,
    output_path: Path,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        raise SystemExit(f"TFLite file already exists: {output_path}. Use --overwrite to replace it.")

    from onnx_tf.backend import prepare  # type: ignore
    import onnx
    import tensorflow as tf

    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tf_dir = tmp_dir_path / "saved_model"

        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(tf_dir))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
        tflite_bytes = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_bytes)
    return output_path


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the EdgeFace checkpoint to ONNX and/or TFLite.",
    )
    parser.add_argument(
        "--checkpoint",
        default="conversion/models/edgeface_xs_gamma_06.pt",
        help="Path to the EdgeFace checkpoint (.pt)",
    )
    parser.add_argument(
        "--input-shape",
        default="1,3,112,112",
        help="Comma separated input tensor shape used for the dummy export.",
    )
    parser.add_argument(
        "--backbone",
        default="edgeface_xs_gamma_06",
        help="Backbone name to instantiate when loading a state_dict checkpoint.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--formats",
        choices=["onnx", "tflite", "both"],
        default="both",
        help="Which formats to generate.",
    )
    parser.add_argument(
        "--onnx-output",
        default="output/edgeface.onnx",
        help="Destination path for the ONNX file.",
    )
    parser.add_argument(
        "--tflite-output",
        default="output/edgeface.tflite",
        help="Destination path for the TFLite file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing artifacts if they exist.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> None:
    args = parse_args(argv)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    input_shape = _parse_shape(args.input_shape)
    if len(input_shape) < 2:
        raise SystemExit("Input shape must include batch and channel dimensions.")

    formats = args.formats
    onnx_path = Path(args.onnx_output).expanduser().resolve()
    tflite_path = Path(args.tflite_output).expanduser().resolve()

    print(f"Exporting checkpoint {checkpoint_path} with input shape {input_shape}.")

    if formats in {"onnx", "both"}:
        print(f"-> Writing ONNX to {onnx_path}")
        export_onnx(
            checkpoint_path=checkpoint_path,
            output_path=onnx_path,
            input_shape=input_shape,
            opset=args.opset,
            overwrite=args.overwrite,
            backbone=args.backbone,
        )

    if formats in {"tflite", "both"}:
        onnx_source = onnx_path
        if formats == "tflite" and not onnx_source.exists():
            print("ONNX artifact missing; exporting temporarily for TFLite conversion.")
            temp_dir = Path("output/temp_edgeface_onnx")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_onnx = temp_dir / "edgeface_temp.onnx"
            try:
                export_onnx(
                    checkpoint_path=checkpoint_path,
                    output_path=temp_onnx,
                    input_shape=input_shape,
                    opset=args.opset,
                    overwrite=True,
                    backbone=args.backbone,
                )
                onnx_source = temp_onnx
                print(f"-> Writing TFLite to {tflite_path}")
                export_tflite(
                    onnx_path=onnx_source,
                    output_path=tflite_path,
                    overwrite=args.overwrite,
                )
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"-> Writing TFLite to {tflite_path}")
            export_tflite(
                onnx_path=onnx_source,
                output_path=tflite_path,
                overwrite=args.overwrite,
            )

    print("Conversion complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
