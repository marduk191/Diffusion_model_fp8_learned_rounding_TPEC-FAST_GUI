# FP8 Model Quantization with Learned Rounding

A high-performance tool for converting neural network models to FP8 (float8_e4m3fn) format using advanced learned rounding techniques inspired by AdaRound. This implementation features "TPEC-Quant" (Top-Principal Error Correction Quantization) with SVD-based optimization for superior quantization quality.

## 🚀 Features

- **Advanced Quantization**: Learned rounding algorithm based on AdaRound with SVD optimization
- **FP8 Support**: Converts models to `torch.float8_e4m3fn` format for maximum efficiency
- **Bias Correction**: Automatic bias adjustment to minimize quantization error
- **Model-Specific Optimizations**: Built-in support for T5XXL and distillation models
- **User-Friendly GUI**: Modern tkinter interface for easy operation
- **Memory Efficient**: Tensor-by-tensor processing with automatic memory cleanup
- **Fast Calibration**: PCA-based low-rank approximation for faster optimization

## 📋 Requirements

### System Requirements
- CUDA-capable GPU (recommended for best performance)
- PyTorch with FP8 support (torch >= 2.1.0)

### Hardware Requirements
- **RAM**: 16GB+ recommended for large models
- **VRAM**: 8GB+ for optimal GPU acceleration
- **Storage**: Sufficient space for input/output models (models can be 4-10GB+)

## 🛠️ Installation

1. **Clone the repository:**
```
git clone https://github.com/marduk191/Diffusion_model_fp8_learned_rounding_TPEC-FAST_GUI.git
```

2. **Install dependencies:**
```
pip install -r requirements.txt
or run it in your comfyui venv
```

3. **Verify FP8 support:**
```python
import torch
print(torch.zeros(1, dtype=torch.float8_e4m3fn))  # Should not raise an error
```

## 🎯 Usage

### GUI Interface (Recommended)

Launch the graphical interface for easy operation:

```bash
python fp8_tppec_learned__fast_gui.py
```

**GUI Features:**
- File browser for input/output selection
- Visual parameter controls with sliders
- Real-time conversion progress
- Automatic output filename generation
- Built-in validation and error handling

### Command Line Interface

For advanced users or batch processing:

```bash
python convert_fp8_scaled_learned_svd_fast.py --input model.safetensors [OPTIONS]
```

**Basic Examples:**

```bash
# Convert a standard model
python convert_fp8_scaled_learned_svd_fast.py --input model.safetensors

# Convert T5XXL model with specific settings
python convert_fp8_scaled_learned_svd_fast.py --input t5xxl.safetensors --t5xxl --num_iter 1000

# High-quality conversion with more calibration samples
python convert_fp8_scaled_learned_svd_fast.py --input model.safetensors --calib_samples 8192 --num_iter 800
```

## ⚙️ Configuration Options

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | **Required** | Input safetensors file path |
| `--output` | str | Auto-generated | Output file path |
| `--t5xxl` | flag | False | Enable T5XXL model optimizations |
| `--keep_distillation` | flag | False | Preserve distillation layers |
| `--calib_samples` | int | 3072 | Random calibration samples |
| `--num_iter` | int | 500 | Optimization iterations per tensor |

### Advanced Parameters

- **Calibration Samples** (512-8192): More samples = better quality but slower conversion
- **Optimization Iterations** (100-2000): More iterations = better convergence but longer processing
- **Learning Rate**: Automatically scheduled with early stopping

## 🔧 Technical Details

### Algorithm Overview

1. **Scale Calculation**: Per-tensor asymmetric scaling to FP8 range
2. **SVD Decomposition**: Low-rank approximation using PCA or full SVD
3. **Learned Rounding**: Iterative optimization with gradient descent
4. **Bias Correction**: Compensates for quantization-induced bias
5. **Memory Management**: Automatic cleanup and garbage collection

### Model Compatibility

- ✅ **Supported**: Linear layers, convolutional layers, embeddings
- ⚠️ **Excluded**: Normalization layers, bias terms (unless corrected)
- 🎯 **Optimized for**: T5XXL, FLUX models, diffusion transformers

### Output Format

The converted model includes:
- Quantized weights in FP8 format
- Per-tensor scaling factors
- Bias-corrected parameters
- Metadata for proper dequantization

## 📊 Performance

### Typical Results
- **Model Size**: ~50% reduction compared to FP16
- **Memory Usage**: ~40-60% reduction during inference
- **Quality**: Minimal accuracy loss with proper calibration
- **Speed**: 2-10x faster inference on supported hardware

### Optimization Tips

1. **Increase calibration samples** for critical models
2. **Use more iterations** for better convergence
3. **Enable T5XXL mode** for text encoder models
4. **Monitor GPU memory** for very large models

## 🐛 Troubleshooting

### Common Issues

**"FP8 not supported"**
- Update PyTorch: `pip install torch>=2.1.0`
- Check CUDA compatibility

**"Out of memory"**
- Reduce calibration samples
- Process on CPU (slower but more memory)
- Close other applications

**"Conversion failed"**
- Verify input file is valid safetensors
- Check file permissions
- Ensure sufficient disk space

### Debug Mode

Add verbose logging for troubleshooting:
```bash
python convert_fp8_scaled_learned_svd_fast.py --input model.safetensors -v
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Development Setup

```bash
git clone https://github.com/yourusername/fp8-quantization.git
cd Diffusion_model_fp8_learned_rounding_TPEC-FAST_GUI
pip install -r requirements.txt

```

## 🙏 Acknowledgments

- **AdaRound Paper**: [Up or Down? Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568)
- **PyTorch Team**: For FP8 support and tensor operations
- **Hugging Face**: For safetensors format
- **Community**: Thanks to all contributors and testers
- Clybius for original script 
- marduk191 for the GUI



---


*Star ⭐ this repository if you find it helpful!*