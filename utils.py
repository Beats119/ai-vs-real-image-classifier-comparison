import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

# Define transforms for different models
def get_transforms(model_name: str) -> Dict[str, transforms.Compose]:
    """
    Get the appropriate transforms for each model type
    
    Args:
        model_name (str): The model name (resnet, mobilenet, mtcnn)
        
    Returns:
        Dict[str, transforms.Compose]: Dictionary with train and test transforms
    """
    if model_name.lower() == 'resnet':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif model_name.lower() == 'mobilenet':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif model_name.lower() == 'mtcnn':
        # MTCNN expects a different input size
        train_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return {'train': train_transform, 'test': test_transform}


def load_data(data_path: str, model_name: str, batch_size: int = 32, 
              val_split: float = 0.2, test_split: float = 0.1) -> Dict[str, DataLoader]:
    """
    Load and prepare data for training and evaluation
    
    Args:
        data_path (str): Path to the dataset
        model_name (str): Model name for appropriate transforms
        batch_size (int): Batch size for dataloaders
        val_split (float): Validation split ratio
        test_split (float): Test split ratio
        
    Returns:
        Dict[str, DataLoader]: Dictionary containing train, val, and test dataloaders
    """
    transforms_dict = get_transforms(model_name)
    
    # Check if the data is already split into train/val/test directories
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    test_dir = os.path.join(data_path, 'test')
    
    # If the data is already split
    if os.path.exists(train_dir) and (os.path.exists(val_dir) or os.path.exists(test_dir)):
        train_dataset = ImageFolder(train_dir, transform=transforms_dict['train'])
        
        if os.path.exists(val_dir):
            val_dataset = ImageFolder(val_dir, transform=transforms_dict['test'])
        else:
            # Split train into train and val
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size], 
                generator=torch.Generator().manual_seed(42)
            )
        
        if os.path.exists(test_dir):
            test_dataset = ImageFolder(test_dir, transform=transforms_dict['test'])
        else:
            # Use validation as test
            test_dataset = val_dataset
    
    # If data is not pre-split
    else:
        full_dataset = ImageFolder(data_path, transform=transforms_dict['train'])
        
        # Calculate splits
        total_size = len(full_dataset)
        test_size = int(test_split * total_size)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size - test_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def save_model(model: nn.Module, model_name: str, save_dir: str = 'models') -> str:
    """
    Save trained model
    
    Args:
        model (nn.Module): The trained model
        model_name (str): Name of the model
        save_dir (str): Directory to save the model
        
    Returns:
        str: Path to the saved model
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_path = os.path.join(save_dir, f"{model_name}_model.pth")
    torch.save(model.state_dict(), model_path)
    
    return model_path


def load_model(model: nn.Module, model_path: str) -> nn.Module:
    """
    Load a trained model
    
    Args:
        model (nn.Module): The model architecture
        model_path (str): Path to the saved model
        
    Returns:
        nn.Module: Loaded model
    """
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate_metrics(y_true: List[int], y_pred: List[int], 
                     class_names: List[str] = ['Real', 'Fake']) -> Dict[str, Any]:
    """
    Calculate and return evaluation metrics
    
    Args:
        y_true (List[int]): Ground truth labels
        y_pred (List[int]): Predicted labels
        class_names (List[str]): Names of the classes
        
    Returns:
        Dict[str, Any]: Dictionary with evaluation metrics
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Get accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Get precision, recall, f1 score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Return all metrics in a dictionary
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }


def plot_metrics(metrics: Dict[str, Any], model_name: str, save_dir: str = 'results') -> Dict[str, str]:
    """
    Plot and save evaluation metrics
    
    Args:
        metrics (Dict[str, Any]): Metrics from evaluate_metrics function
        model_name (str): Name of the model
        save_dir (str): Directory to save the plots
        
    Returns:
        Dict[str, str]: Dictionary with paths to saved plots
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    result_paths = {}
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    result_paths['confusion_matrix'] = cm_path
    
    # Plot classification metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
    })
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Metric', y='Value', data=metrics_df)
    plt.title(f'Classification Metrics - {model_name}')
    plt.ylim(0, 1)
    
    # Add values on top of bars
    for i, v in enumerate(metrics_df['Value']):
        ax.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.png")
    plt.savefig(metrics_path)
    plt.close()
    result_paths['metrics'] = metrics_path
    
    # Create a detailed report
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    
    # Save to CSV
    report_path = os.path.join(save_dir, f"{model_name}_report.csv")
    report_df.to_csv(report_path)
    result_paths['report'] = report_path
    
    return result_paths


def preprocess_image_for_prediction(image_path: str, model_name: str) -> torch.Tensor:
    """
    Preprocess a single image for prediction
    
    Args:
        image_path (str): Path to the image
        model_name (str): Name of the model
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transforms_dict = get_transforms(model_name)
    transform = transforms_dict['test']
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def get_device() -> torch.device:
    """
    Get the available device (CUDA or CPU)
    
    Returns:
        torch.device: Selected device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def compare_models(results: Dict[str, Dict[str, Any]], save_dir: str = 'results') -> str:
    """
    Create comparison plots and tables for different models
    
    Args:
        results (Dict[str, Dict[str, Any]]): Dictionary with results for each model
        save_dir (str): Directory to save the comparison results
        
    Returns:
        str: Path to the comparison report
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Extract key metrics for comparison
    models = list(results.keys())
    accuracy = [results[model]['accuracy'] for model in models]
    precision = [results[model]['precision'] for model in models]
    recall = [results[model]['recall'] for model in models]
    f1_score = [results[model]['f1_score'] for model in models]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    })
    
    # Plot comparison bar chart
    plt.figure(figsize=(12, 8))
    df_melted = pd.melt(df, id_vars=['Model'], var_name='Metric', value_name='Value')
    
    sns.barplot(x='Model', y='Value', hue='Metric', data=df_melted)
    plt.title('Model Comparison')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(comparison_path)
    plt.close()
    
    # Create comparison table
    comparison_table = df.copy()
    
    # Save to CSV
    table_path = os.path.join(save_dir, "model_comparison.csv")
    comparison_table.to_csv(table_path, index=False)
    
    # Create a more detailed HTML report
    html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI vs Real Image Detection - Model Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .best {{ background-color: #d4edda; font-weight: bold; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>AI vs Real Image Detection - Model Comparison</h1>
    <h2>Performance Metrics</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
        </tr>
"""
    
    # Find the best values for highlighting
    best_acc = max(accuracy)
    best_prec = max(precision)
    best_rec = max(recall)
    best_f1 = max(f1_score)
    
    # Add rows to the HTML table
    for i, model in enumerate(models):
        html_report += f"<tr>"
        html_report += f"<td>{model}</td>"
        
        # Add class for best values
        acc_class = " class='best'" if accuracy[i] == best_acc else ""
        prec_class = " class='best'" if precision[i] == best_prec else ""
        rec_class = " class='best'" if recall[i] == best_rec else ""
        f1_class = " class='best'" if f1_score[i] == best_f1 else ""
        
        html_report += f"<td{acc_class}>{accuracy[i]:.4f}</td>"
        html_report += f"<td{prec_class}>{precision[i]:.4f}</td>"
        html_report += f"<td{rec_class}>{recall[i]:.4f}</td>"
        html_report += f"<td{f1_class}>{f1_score[i]:.4f}</td>"
        html_report += "</tr>"
    
    html_report += """
    </table>
    <h2>Comparison Chart</h2>
    <img src="model_comparison.png" alt="Model comparison chart">
"""
    
    # Add individual model sections
    for model in models:
        html_report += f"""
    <h2>{model} Model Results</h2>
    <h3>Confusion Matrix</h3>
    <img src="{model}_confusion_matrix.png" alt="{model} confusion matrix">
    
    <h3>Performance Metrics</h3>
    <img src="{model}_metrics.png" alt="{model} metrics">
"""
    
    html_report += """
</body>
</html>
"""
    
    # Save HTML report
    report_path = os.path.join(save_dir, "comparison_report.html")
    with open(report_path, 'w') as f:
        f.write(html_report)
    
    return report_path
