# Medicinal Plant Classification with ViT

## Overview
This project fine-tunes a Vision Transformer (ViT) model using the Hugging Face Trainer for the classification of 40 different medicinal plants. The Vision Transformer (ViT) model leverages the power of transformer architecture for image classification tasks.

## Features
- **Fine-tuning**: Fine-tune a pre-trained ViT model on a custom dataset of medicinal plants.
- **Easy Training**: Utilize Hugging Face's Trainer API for streamlined training and evaluation.
- **Performance Monitoring**: Track model performance using evaluation metrics and training logs.

## Prerequisites
- Python 3.6+
- PyTorch
- Hugging Face Transformers
- Datasets library

## Installation

### PyTorch
Install PyTorch by following the instructions on the [official website](https://pytorch.org/get-started/locally/).

### Hugging Face Transformers and Datasets
Install the required libraries:

```sh
pip install transformers datasets
```

## Dataset
Prepare your dataset of 40 medicinal plants. Ensure the dataset is structured with separate folders for each plant class, with images inside each folder. For example:

```
dataset/
  ├── class_1/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── class_2/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── ...
  └── class_40/
      ├── image1.jpg
      ├── image2.jpg
      └── ...
```

## Getting Started

### Step 1: Load and Preprocess the Dataset
Load and preprocess the dataset using the `datasets` library. Below is an example script to load the dataset:

```python
from datasets import load_dataset
from transformers import ViTFeatureExtractor

dataset = load_dataset('imagefolder', data_dir='dataset')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    return inputs

prepared_ds = dataset.with_transform(transform)
```

### Step 2: Fine-tune the ViT Model
Fine-tune the pre-trained ViT model using the Hugging Face Trainer API. Below is an example script:

```python
from transformers import ViTForImageClassification, TrainingArguments, Trainer
import torch

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=40)

training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {'accuracy': (preds == p.label_ids).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds['train'],
    eval_dataset=prepared_ds['test'],
    compute_metrics=compute_metrics
)

trainer.train()
```

### Step 3: Evaluate the Model
After training, evaluate the model on the test dataset to check its performance.

```python
results = trainer.evaluate()
print(results)
```

## Inference
To use the fine-tuned model for inference, you can load the model and feature extractor, and make predictions on new images:

```python
from PIL import Image

model.eval()

def predict(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx

# Example usage
image_path = 'path_to_your_image.jpg'
predicted_class = predict(image_path)
print(f'Predicted class: {predicted_class}')
```

## Conclusion
This project showcases how to fine-tune a Vision Transformer (ViT) model for the classification of 40 medicinal plants using the Hugging Face Trainer API. The use of the Hugging Face ecosystem simplifies the training and evaluation process, enabling efficient model development and deployment.

For more detailed information, refer to the [Hugging Face Transformers documentation](https://huggingface.co/transformers/).

