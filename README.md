# PyTorch GRU to ONNX & Quantization Pipeline

This repository provides a step-by-step guide for exporting a Recurrent Neural Network (GRU) from PyTorch to the ONNX format, and applying both **FP16** and **INT8** quantization. 

This mimics workflows used for embedded edge deployment of time-series/telemetry data models.

## 🚀 Why do this?
- **Interoperability:** ONNX handles RNNs (like GRUs) smoothly, allowing you to deploy them to NPUs, edge CPUs, or microcontrollers.
- **Quantization (FP16 & INT8):** 
  - **FP16:** Shrinks the model by exactly 50% with practically zero accuracy loss. Ideal for NPUs and modern GPUs.
  - **INT8:** Shrinks the model by ~75% and uses fast integer math. Ideal for CPU inference.

## 📁 Project Structure

* `model.py`: Defines the PyTorch `SimpleGRU` architecture (simulating a 24-feature, 128-sequence telemetry model).
* `1_train_model.py`: Initializes the GRU and saves the `.pth` weights.
* `2_export_onnx.py`: Loads the `.pth` weights and traces the sequence logic into an `.onnx` graph using `torch.onnx.export`.
* `3_quantize_model.py`: Generates two optimized models: an **FP16** ONNX model and an **INT8** dynamically quantized ONNX model.
* `4_benchmark.py`: Runs a benchmark comparing `.pth`, FP32 `.onnx`, FP16 `.onnx`, and INT8 `.onnx` for size and CPU speed.

## 🛠️ Setup & Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```
*(Note: `onnxconverter-common` is required specifically for the FP16 conversion).*

## 🏃 How to run

Run the steps in order:

1. **Initialize the model**
   ```bash
   python 1_train_model.py
   ```
2. **Export to ONNX**
   ```bash
   python 2_export_onnx.py
   ```
3. **Apply FP16 and INT8 Quantization**
   ```bash
   python 3_quantize_model.py
   ```
4. **Benchmark Sizes and Speeds**
   ```bash
   python 4_benchmark.py
   ```
