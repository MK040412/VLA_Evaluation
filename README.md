# Technical Report with LLM

[![Open in DeepWiki](https://img.shields.io/badge/Open%20in-DeepWiki-blue?logo=book&logoColor=white)](https://app.devin.ai/wiki/MK040412/RDT-1B-Ablation#1)

---

# Robotics Diffusion Transformer (RDT)

This repository contains the code for the Robotics Diffusion Transformer (RDT). This README provides instructions on how to set up the environment, use pre-trained models, and perform evaluation and quantization.

## 1. Installation

To get started, clone the repository and install the required dependencies.

```bash
# 1. Clone the repository
git clone https://github.com/MK040412/RDT-1B-Ablation.git
cd RDT-1B-Ablation

# 2. Install dependencies
pip install -r requirements.txt
```

## 2. Usage

### 2.1. Download Pre-trained Models

Our pre-trained models are hosted on the Hugging Face Hub. You can easily download them using Git LFS.

```bash
# Make sure you have git-lfs installed. If not, uncomment the following line:
# sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install

# Download the model files
git lfs pull
```
The models will be downloaded to the `pretrained_models/rdt` directory.

### 2.2. Evaluation on ManiSkill Benchmark

You can evaluate the performance of the RDT model on the ManiSkill benchmark using the provided evaluation script.

```bash
# Run the evaluation script
python -m  eval_sim.eval_rdt_maniskill --pretrained_path pretrained_models/rdt/mp_rank_00_model_states.pt --live-view

```
This script will run the evaluation and print the success rate and other metrics to the console.

### 2.3. Model Quantization

This repository provides tools to perform post-training quantization on the RDT model. Quantization can reduce the model size and improve inference speed, which is crucial for deployment on real robots with limited computational resources.

#### 2.3.1. Running Quantization Experiments

The `run_quantization_experiments.sh` script allows you to evaluate the model with different quantization configurations (e.g., different bit-widths).

```bash
# Run the quantization experiments
./run_quantization_experiments.sh
```

This script will:
1.  Apply different quantization techniques to the model.
2.  Evaluate the performance of each quantized model.
3.  Save the evaluation results in the `results/` directory.

#### 2.3.2. Analyzing Quantization Results

After running the experiments, you can use the `analyze_quantization.py` script to visualize the results.

```bash
# Analyze the quantization results
python analyze_quantization.py
```

This script will generate a plot (`quantization_analysis.png`) that shows the Mean Squared Error (MSE) distribution for different quantization levels. This helps in understanding the trade-off between model size/performance and the level of quantization.
