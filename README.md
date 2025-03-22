# AI vs Real Image Detection - Model Comparison

This repository contains a comprehensive implementation for comparing MTCNN, ResNet, and MobileNet models for AI-generated vs real image detection.

## Overview

This project provides:
- Data preprocessing pipelines
- Implementations of MTCNN, ResNet, and MobileNet for image classification
- Performance metrics and comparative analysis
- Gradio UI for easy model testing

## Requirements

```
python >= 3.8
torch >= 1.12.0
torchvision >= 0.13.0
tensorflow >= 2.8.0
mtcnn >= 0.1.1
gradio >= 3.8.0
numpy
pandas
matplotlib
scikit-learn
pillow
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Beats119/ai-vs-real-image-classifier-comparison.git
cd ai-vs-real-image-classifier-comparison
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train models with your dataset

```bash
python train.py --data_path /path/to/your/dataset --model_name [resnet/mobilenet/mtcnn] --batch_size 32 --epochs 20
```

The dataset should be structured as follows:
```
dataset/
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── fake/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    ├── real/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── fake/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

### 2. Evaluate models

```bash
python evaluate.py --data_path /path/to/test/dataset --model_name all
```

### 3. Run the Gradio UI for interactive testing

```bash
python app.py
```

Then open your browser at http://localhost:7860

## Model Details

### MTCNN
Multi-task Cascaded Convolutional Neural Network (MTCNN) is primarily used for face detection but has been adapted here for AI vs real image classification by analyzing facial features.

### ResNet
Residual Network (ResNet) is a deep convolutional neural network architecture that uses skip connections to overcome the vanishing gradient problem in very deep networks.

### MobileNet
MobileNet is designed for mobile and embedded vision applications, using depthwise separable convolutions to create lightweight deep neural networks.

## Results

The comparative analysis results and performance metrics can be found in the `results` directory after running the evaluation.

## License

MIT
