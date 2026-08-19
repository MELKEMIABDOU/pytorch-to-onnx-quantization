# PyTorch GRU to ONNX & Quantization Pipeline

This repository provides a step-by-step guide for exporting a Recurrent Neural Network (GRU) from PyTorch to the ONNX format, and applying both **FP16** and **INT8** quantization. 

This mimics workflows used for embedded edge deployment of time-series/telemetry data models.

##  Why do this?
- **Interoperability:** ONNX handles RNNs (like GRUs) smoothly, allowing you to deploy them to NPUs, edge CPUs, or microcontrollers.
- **Faster Inference & Smaller Size:** Quantization dramatically reduces memory footprint and accelerates inference on resource-constrained devices.

##  Understanding Quantization

### The Data Types: FP16 vs. INT8
Quantization shrinks the mathematical numbers inside your model (which default to 32-bit floating point, or FP32).
* **FP16 (16-bit Floating Point):** Reduces the model size by exactly **50%**. It maintains near-perfect accuracy because the numbers remain decimals. It is highly optimized for modern GPUs and NPUs.
* **INT8 (8-bit Integer):** Reduces the model size by **~75%**. It converts complex decimals into whole numbers, which can cause a slight accuracy drop. However, it is incredibly fast on regular CPUs and embedded Edge devices because integer math is computationally cheaper.

### The Methods: Dynamic vs. Static
* **Dynamic Quantization (Used in this repo):** The model's weights are converted to INT8 ahead of time, but the activations (data flowing through the network) are quantized *on-the-fly* during inference. This is highly recommended for **RNNs, GRUs, and LSTMs**, and it requires no extra calibration data.
* **Static Quantization:** Both weights and activations are locked into INT8 ahead of time. This requires feeding the model a "Calibration Dataset" beforehand so it can learn how to scale the integers correctly. This method provides the absolute fastest inference and is strictly required by many hardware accelerators (like the Qualcomm Hexagon NPU).

##  Project Structure

* `model.py`: Defines the PyTorch `SimpleGRU` architecture (simulating a 24-feature, 128-sequence telemetry model).
* `1_train_model.py`: Initializes the GRU and saves the `.pth` weights.
* `2_export_onnx.py`: Loads the `.pth` weights and traces the sequence logic into an `.onnx` graph using `torch.onnx.export`.
* `3_quantize_model.py`: Generates two optimized models: an **FP16** ONNX model and an **INT8** dynamically quantized ONNX model.
* `4_benchmark.py`: Runs a benchmark comparing `.pth`, FP32 `.onnx`, FP16 `.onnx`, and INT8 `.onnx` for size and CPU speed.

##  Setup & Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```
*(Note: `onnxconverter-common` is required specifically for the FP16 conversion).*

##  How to run

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
