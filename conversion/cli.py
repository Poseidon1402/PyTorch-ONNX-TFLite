#!/usr/bin/env python
"""Convenience command-line entry point for model conversion pipeline."""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "conversion" / "models"
DEFAULT_OUTPUT_DIR = Path("output")

TaskFunc = Callable[..., Dict[str, Path]]


def _prompt(prompt_text: str) -> str:
    try:
        return input(prompt_text)
    except EOFError as exc:  # pragma: no cover - defensive guard when stdin ends unexpectedly
        raise SystemExit("Input stream closed while prompting user.") from exc


def _prompt_for_missing_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.task:
        print("Select task:")
        print("  1) torch-to-onnx")
        print("  2) onnx-to-tf")
        print("  3) onnx-to-tflite")
        print("  4) tf-to-tflite")
        print("  5) full")
        choice = _prompt("Enter choice [1-5]: ").strip()
        mapping = {
            "1": "torch-to-onnx",
            "2": "onnx-to-tf",
            "3": "onnx-to-tflite",
            "4": "tf-to-tflite",
            "5": "full",
        }
        args.task = mapping.get(choice)
        if args.task is None:
            raise SystemExit("Invalid task selection.")

    if not args.input_path and args.task != "torch-to-onnx":
        default_hint = DEFAULT_MODELS_DIR
        args.input_path = _prompt(
            f"Enter the path to the input model [default directory: {default_hint}]: "
        ).strip()

    if not args.output_dir:
        args.output_dir = _prompt(
            f"Enter the output directory [default: {DEFAULT_OUTPUT_DIR}]: "
        ).strip() or str(DEFAULT_OUTPUT_DIR)

    if args.output_name is None:
        provided = _prompt("Optional output name (leave blank for default): ").strip()
        args.output_name = provided or None

    if args.task == "torch-to-onnx":
        if not args.torch_module:
            args.torch_module = _prompt(
                "Enter the Python module path containing the model class (e.g. mypkg.models): "
            ).strip()
        if not args.torch_class:
            args.torch_class = _prompt(
                "Enter the model class name (e.g. MobileNetV2): "
            ).strip()
        if not args.torch_checkpoint:
            checkpoint = _prompt(
                f"Optional checkpoint path for state_dict (searched relative to {DEFAULT_MODELS_DIR}): "
            ).strip()
            args.torch_checkpoint = checkpoint or None
        if not args.input_shape:
            args.input_shape = _prompt(
                "Enter input shape as comma separated ints (default 1,3,640,640): "
            ).strip()

    return args


def _resolve_with_default_dirs(candidate: Path) -> Path:
    if candidate.exists():
        return candidate

    if not candidate.is_absolute():
        fallback = DEFAULT_MODELS_DIR / candidate
        if fallback.exists():
            return fallback.resolve()

    return candidate


def _ensure_path(path_value: str, must_exist: bool) -> Path:
    if not path_value:
        raise SystemExit("Path value cannot be empty.")
    raw_path = Path(path_value).expanduser()
    resolved = _resolve_with_default_dirs(raw_path)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path not found: {resolved}")
    return resolved.resolve()


def _parse_shape(shape_str: str) -> Tuple[int, ...]:
    try:
        dims = tuple(int(dim.strip()) for dim in shape_str.split(",") if dim.strip())
    except ValueError as exc:  # pragma: no cover - bad user input handled interactively
        raise SystemExit(f"Invalid input shape '{shape_str}'. Must be comma separated integers.") from exc
    if not dims:
        raise SystemExit("Input shape must contain at least one dimension.")
    return dims


def convert_torch_to_onnx(
    input_path: Optional[Path],
    output_dir: Path,
    output_name: Optional[str] = None,
    overwrite: bool = False,
    *,
    torch_module: Optional[str] = None,
    torch_class: Optional[str] = None,
    torch_checkpoint: Optional[Path] = None,
    input_shape: Tuple[int, ...],
    opset: int,
    **_: object,
) -> Dict[str, Path]:
    """Convert a PyTorch model into an ONNX graph using dynamic import."""
    if not torch_module:
        raise SystemExit("--torch-module is required for torch-to-onnx conversions.")
    if not torch_class:
        raise SystemExit("--torch-class is required for torch-to-onnx conversions.")

    if torch_checkpoint is None and input_path and input_path.is_file():
        torch_checkpoint = input_path

    if torch_checkpoint is not None:
        torch_checkpoint = _resolve_with_default_dirs(torch_checkpoint)
        if not torch_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {torch_checkpoint}")

    module = importlib.import_module(torch_module)
    try:
        model_cls = getattr(module, torch_class)
    except AttributeError as exc:
        raise SystemExit(f"Class '{torch_class}' not found in module '{torch_module}'.") from exc

    import torch  # type: ignore[import-not-found]

    model = model_cls()
    if torch_checkpoint is not None:
        state_dict = torch.load(torch_checkpoint, map_location="cpu")
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(state_dict)
        else:  # pragma: no cover - defensive branch if model lacks load_state_dict
            raise SystemExit("Model does not support load_state_dict; cannot apply checkpoint.")
    model.eval()

    sample_input = torch.rand(input_shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{output_name or model_cls.__name__}.onnx"
    if onnx_path.exists() and not overwrite:
        raise FileExistsError(
            f"ONNX output already exists: {onnx_path}. Use --overwrite to replace it."
        )

    print(
        "Exporting PyTorch model to ONNX with shape "
        f"{input_shape} and opset {opset}."
    )

    torch.onnx.export(
        model,
        sample_input,
        str(onnx_path),
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    return {"onnx_model": onnx_path}


def convert_onnx_to_tf(
    input_path: Path,
    output_dir: Path,
    output_name: Optional[str] = None,
    overwrite: bool = False,
    **_: object,
) -> Dict[str, Path]:
    """Convert an ONNX model into a TensorFlow SavedModel."""
    from onnx_tf.backend import prepare
    import onnx

    if not input_path.is_file():
        raise ValueError(f"ONNX input must be a file: {input_path}")

    export_dir = output_dir / (output_name or f"{input_path.stem}_tf")
    if export_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Export target already exists: {export_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(export_dir)

    export_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading ONNX model from {input_path}")
    onnx_model = onnx.load(str(input_path))

    print("Preparing TensorFlow representation…")
    tf_rep = prepare(onnx_model)

    print(f"Exporting TensorFlow SavedModel to {export_dir}")
    tf_rep.export_graph(str(export_dir))

    return {"tf_model_dir": export_dir}


def convert_tf_to_tflite(
    input_path: Path,
    output_dir: Path,
    output_name: Optional[str] = None,
    overwrite: bool = False,
    **_: object,
) -> Dict[str, Path]:
    """Convert a TensorFlow SavedModel directory into a TFLite flatbuffer."""
    import tensorflow as tf

    if not input_path.is_dir():
        raise ValueError(f"TensorFlow SavedModel directory expected: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = output_dir / f"{output_name or input_path.name}.tflite"

    if tflite_path.exists() and not overwrite:
        raise FileExistsError(
            f"TFLite output already exists: {tflite_path}. Use --overwrite to replace it."
        )

    print(f"Converting TensorFlow model at {input_path} to TFLite")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(input_path))
    tflite_model = converter.convert()

    print(f"Writing TFLite model to {tflite_path}")
    tflite_path.write_bytes(tflite_model)

    return {"tflite_model": tflite_path}


def convert_onnx_to_tflite(
    input_path: Path,
    output_dir: Path,
    output_name: Optional[str] = None,
    overwrite: bool = False,
    **kwargs: object,
) -> Dict[str, Path]:
    """Convert an ONNX model directly into TFLite with an intermediate TF export."""
    base_name = output_name or input_path.stem

    tf_results = convert_onnx_to_tf(
        input_path=input_path,
        output_dir=output_dir,
        output_name=base_name + "_tf",
        overwrite=overwrite,
    )
    tf_dir = tf_results["tf_model_dir"]

    tflite_results = convert_tf_to_tflite(
        input_path=tf_dir,
        output_dir=output_dir,
        output_name=base_name,
        overwrite=overwrite,
    )

    return {**tf_results, **tflite_results}


def run_full_pipeline(
    input_path: Path,
    output_dir: Path,
    output_name: Optional[str] = None,
    overwrite: bool = False,
    **kwargs: object,
) -> Dict[str, Path]:
    """Run ONNX -> TF -> TFLite pipeline in a single command."""
    base_name = output_name or input_path.stem
    tf_dir = output_dir / f"{base_name}_tf"
    tflite_dir = output_dir

    results = convert_onnx_to_tf(
        input_path=input_path,
        output_dir=output_dir,
        output_name=base_name + "_tf",
        overwrite=overwrite,
    )

    tf_dir_actual = results["tf_model_dir"]

    tflite_results = convert_tf_to_tflite(
        input_path=tf_dir_actual,
        output_dir=tflite_dir,
        output_name=base_name,
        overwrite=overwrite,
    )

    return {**results, **tflite_results}


TASKS: Dict[str, TaskFunc] = {
    "torch-to-onnx": convert_torch_to_onnx,
    "onnx-to-tf": convert_onnx_to_tf,
    "onnx-to-tflite": convert_onnx_to_tflite,
    "tf-to-tflite": convert_tf_to_tflite,
    "full": run_full_pipeline,
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entry point for converting models across ONNX, TensorFlow, and TFLite formats.",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASKS.keys()),
        help="Conversion pipeline to execute.",
    )
    parser.add_argument(
        "--input-path",
        dest="input_path",
        help="Path to the input model file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--output-name",
        dest="output_name",
        help="Optional name override for generated artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing artifacts when conflicts occur.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail instead of prompting when required arguments are missing.",
    )
    parser.add_argument(
        "--torch-module",
        help="Python module path containing the PyTorch model class (torch-to-onnx).",
    )
    parser.add_argument(
        "--torch-class",
        help="PyTorch model class name to instantiate (torch-to-onnx).",
    )
    parser.add_argument(
        "--torch-checkpoint",
        help="Checkpoint file to load with torch.load before export (torch-to-onnx).",
    )
    parser.add_argument(
        "--input-shape",
        default="1,3,640,640",
        help="Comma separated tensor shape for synthetic input when exporting from PyTorch.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version for torch.onnx.export.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    required_fields = ["task", "output_dir"]
    if args.task != "torch-to-onnx":
        required_fields.append("input_path")

    missing = [
        name
        for name in required_fields
        if getattr(args, name) in (None, "")
    ]

    if missing and args.non_interactive:
        missing_csv = ", ".join(missing)
        raise SystemExit(f"Missing required arguments in non-interactive mode: {missing_csv}")

    if missing:
        args = _prompt_for_missing_args(args)

    task = args.task
    if task not in TASKS:
        raise SystemExit(f"Unsupported task: {task}")

    if args.task == "torch-to-onnx":
        input_path = (
            _ensure_path(args.input_path, must_exist=True)
            if args.input_path
            else None
        )
    else:
        input_path = _ensure_path(args.input_path, must_exist=True)
    output_dir = _ensure_path(args.output_dir, must_exist=False)

    print(f"Executing task '{task}'")

    task_func = TASKS[task]
    torch_checkpoint = (
        Path(args.torch_checkpoint).expanduser().resolve()
        if args.torch_checkpoint
        else None
    )
    input_shape = _parse_shape(args.input_shape)

    results = task_func(
        input_path=input_path,
        output_dir=output_dir,
        output_name=args.output_name,
        overwrite=args.overwrite,
        torch_module=args.torch_module,
        torch_class=args.torch_class,
        torch_checkpoint=torch_checkpoint,
        input_shape=input_shape,
        opset=args.opset,
    )

    print("\nArtifacts generated:")
    for label, path in results.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
