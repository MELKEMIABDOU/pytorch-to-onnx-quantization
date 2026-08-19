# PyTorch to ONNX & Quantization Pipeline

This repository provides a step-by-step guide and implementation for exporting a PyTorch model to the ONNX format and applying INT8 dynamic quantization using ONNX Runtime. 

## 🚀 Why do this?
- **Interoperability:** ONNX is an open standard format built to represent machine learning models. Once in ONNX, you can run your model on multiple platforms (C++, Java, JS) and hardware accelerators (GPUs, NPUs).
- **Quantization:** By converting model weights from 32-bit floating-point (FP32) to 8-bit integers (INT8), you can reduce the model size by roughly 4x and significantly speed up CPU inference without losing much accuracy.

## 📁 Project Structure

* `1_train_model.py`: Downloads a pre-trained MobileNetV2 from PyTorch and saves its weights (`.pth`).
* `2_export_onnx.py`: Loads the `.pth` weights and exports the model to an `.onnx` graph using `torch.onnx.export`.
* `3_quantize_model.py`: Reads the standard `.onnx` model and applies dynamic INT8 quantization using `onnxruntime`.
* `4_benchmark.py`: Runs a comparison between the `.pth`, FP32 `.onnx`, and INT8 `.onnx` models, measuring file size and CPU inference speed.

## 🛠️ Setup & Installation

It is recommended to run this inside a virtual environment. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## 🏃 How to run

Run the steps in order to see the full pipeline:

1. **Get the base model**
   ```bash
   python 1_train_model.py
   ```
2. **Export to ONNX**
   ```bash
   python 2_export_onnx.py
   ```
3. **Quantize the Model**
   ```bash
   python 3_quantize_model.py
   ```
4. **Benchmark Sizes and Speeds**
   ```bash
   python 4_benchmark.py
   ```

## 📊 Expected Results
When you run the benchmark, you should observe:
1. **Size reduction:** The quantized ONNX model will be approximately 1/4 the size of the original PyTorch model (and the standard ONNX model).
2. **Speedup:** The INT8 ONNX model should run significantly faster on CPU compared to standard PyTorch FP32 inference.
