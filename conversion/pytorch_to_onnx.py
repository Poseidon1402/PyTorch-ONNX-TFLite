import torch
import os
from backbones import get_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
MODEL_NAME = "edgeface_xs_gamma_06"
CHECKPOINT_PATH = f"conversion/models/{MODEL_NAME}.pt"
ONNX_OUTPUT_PATH = f"conversion/models/{MODEL_NAME}.onnx"
BATCH_SIZE = 1
INPUT_SIZE = (3, 112, 112)  # C, H, W for aligned face input

def export_edgeface_to_onnx(
    model_name: str,
    checkpoint_path: str,
    output_path: str,
    batch_size: int = 1,
    input_size: tuple = (3, 112, 112),
    opset_version: int = 11,
    verify: bool = True
):
    """
    Export EdgeFace PyTorch model to ONNX format.
    
    Args:
        model_name: EdgeFace model variant name (e.g., 'edgeface_xs_gamma_06')
        checkpoint_path: path to .pt checkpoint file
        output_path: where to save the .onnx file
        batch_size: fixed batch size for export (default 1)
        input_size: (C, H, W) tuple for aligned face input (default (3, 112, 112))
        opset_version: ONNX opset version (default 11 for broad compatibility)
        verify: if True, test that ONNX output matches PyTorch
        
    Returns:
        None
    """
    print(f"[1/4] Loading model: {model_name}")
    model = get_model(model_name)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"[2/4] Creating dummy input: batch={batch_size}, shape={input_size}")
    dummy_input = torch.randn(batch_size, *input_size, requires_grad=False)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    print(f"[3/4] Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        }
    )
    
    print(f"✓ ONNX model saved to: {output_path}")
    
    if verify:
        print("[4/4] Verifying ONNX output matches PyTorch...")
        import onnxruntime as ort
        import numpy as np
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = model(dummy_input).cpu().numpy()
        
        # ONNX Runtime inference
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(output_path, providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        onnx_output = session.run([output_name], {input_name: dummy_input.numpy()})[0]
        
        # Compare outputs
        max_diff = np.abs(pytorch_output - onnx_output).max()
        mean_diff = np.abs(pytorch_output - onnx_output).mean()
        
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-5:
            print("✓ Verification passed: outputs match within tolerance")
        else:
            print(f"⚠ Warning: outputs differ by {max_diff:.6f} (expected <1e-5)")
    else:
        print("[4/4] Skipping verification")

if __name__ == "__main__":
    try:
        export_edgeface_to_onnx(
            model_name=MODEL_NAME,
            checkpoint_path=CHECKPOINT_PATH,
            output_path=ONNX_OUTPUT_PATH,
            batch_size=BATCH_SIZE,
            input_size=INPUT_SIZE,
            opset_version=11,
            verify=True
        )
        
        print("\n" + "="*60)
        print("Export complete! Use this ONNX model with inference_2.py")
        print(f"Update inference_2.py line 11 to: onnx_path = '{ONNX_OUTPUT_PATH}'")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()