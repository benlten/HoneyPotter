#!/usr/bin/env python3
"""
Honeypotter Integration Example
==============================

Complete example showing how to integrate honeypots with an existing dataset 
for robustness evaluation, while ensuring honeypots are NEVER used in training.

This example shows:
1. How to generate honeypots for your dataset
2. How to extend the label space correctly  
3. How to create dataloaders that exclude honeypots from training
4. How to evaluate and compute robustness metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score
import json
import subprocess
import os
from pathlib import Path


class HoneypotIntegration:
    """Helper class for integrating honeypots with existing datasets."""
    
    def __init__(self, dataset_path: str, num_original_classes: int):
        self.dataset_path = Path(dataset_path)
        self.num_original_classes = num_original_classes
        self.honeypot_path = None
        self.honeypot_mapping = {}
        
    def generate_honeypots(self, output_dir: str, n_categories: int = 50, per_class: int = 100):
        """Generate honeypots using the CLI."""
        
        self.honeypot_path = Path(output_dir)
        
        # Generate using honeypotter CLI
        cmd = [
            "honeypotter", "generate",
            "--out", str(self.honeypot_path),
            "--n_categories", str(n_categories),
            "--per_class", str(per_class),
            "--families", "checker,stripes,dots,blob",
            "--color_mode", "ood",
            "--seed", "42"
        ]
        
        print(f"Generating {n_categories} honeypot categories...")
        subprocess.run(cmd, check=True)
        
        # Load the generated label mapping
        with open(self.honeypot_path / "labelmap.json", "r") as f:
            self.honeypot_mapping = json.load(f)
            
        print(f"Generated honeypots: {list(self.honeypot_mapping.keys())[:5]}...")
        
    def create_dataloaders(self, batch_size: int = 32):
        """Create train/val dataloaders with proper honeypot handling."""
        
        if not self.honeypot_path:
            raise ValueError("Must generate honeypots first!")
            
        # Standard transforms
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # CRITICAL: Training uses ONLY original dataset
        train_dataset = ImageFolder(
            root=self.dataset_path / "train",
            transform=train_transforms
        )
        
        # Validation: Original dataset
        val_original = ImageFolder(
            root=self.dataset_path / "val",
            transform=val_transforms
        )
        
        # Validation: Honeypots with shifted labels
        val_honeypots = ImageFolder(
            root=self.honeypot_path,
            transform=val_transforms,
            # CRITICAL: Map honeypot labels to extended range
            target_transform=lambda x: x + self.num_original_classes
        )
        
        # Combined validation set
        val_combined = ConcatDataset([val_original, val_honeypots])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_combined, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset)} (original only)")
        print(f"Validation samples: {len(val_original)} original + {len(val_honeypots)} honeypots")
        
        return train_loader, val_loader
    
    def evaluate_model(self, model: nn.Module, val_loader: DataLoader, device: str = "cuda"):
        """Evaluate model and compute honeypot-specific metrics."""
        
        model.eval()
        all_preds, all_labels = [], []
        
        print("Evaluating model on validation set...")
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Separate original and honeypot predictions
        original_mask = all_labels < self.num_original_classes
        honeypot_mask = ~original_mask
        
        # Compute metrics
        results = {}
        
        # 1. Clean accuracy (original validation set)
        if original_mask.any():
            results["clean_accuracy"] = accuracy_score(
                all_labels[original_mask], 
                all_preds[original_mask]
            )
        
        # 2. Honeypot metrics
        if honeypot_mask.any():
            honeypot_preds = all_preds[honeypot_mask]
            
            # Steal rate: fraction of honeypots predicted as original classes
            steal_rate = (honeypot_preds < self.num_original_classes).mean()
            results["steal_rate"] = steal_rate
            
            # Honeypot confusion: fraction predicted as other honeypots
            results["honeypot_confusion"] = 1 - steal_rate
            
            # Most confused original classes
            stolen_classes = honeypot_preds[honeypot_preds < self.num_original_classes]
            if len(stolen_classes) > 0:
                unique, counts = np.unique(stolen_classes, return_counts=True)
                most_confused = unique[np.argmax(counts)]
                results["most_confused_class"] = int(most_confused)
        
        return results
    
    def print_results(self, results: dict):
        """Print evaluation results in a nice format."""
        
        print("\n" + "="*50)
        print("HONEYPOT EVALUATION RESULTS")  
        print("="*50)
        
        if "clean_accuracy" in results:
            print(f"Clean Accuracy:      {results['clean_accuracy']:.3f}")
            
        if "steal_rate" in results:
            print(f"Steal Rate:          {results['steal_rate']:.3f} (lower is better)")
            print(f"Honeypot Confusion:  {results['honeypot_confusion']:.3f} (higher is better)")
            
        if "most_confused_class" in results:
            print(f"Most Confused Class: {results['most_confused_class']}")
            
        print("\nInterpretation:")
        if "steal_rate" in results:
            if results['steal_rate'] < 0.1:
                print("✅ Model shows good robustness to honeypots")
            elif results['steal_rate'] < 0.3:
                print("⚠️  Model shows moderate susceptibility to honeypots") 
            else:
                print("❌ Model is highly susceptible to honeypot distractors")


def main():
    """Example usage of honeypot integration."""
    
    # Example configuration
    DATASET_PATH = "./imagenet"  # Path to your ImageNet dataset
    NUM_CLASSES = 1000           # Number of original classes
    
    # Initialize integration
    integration = HoneypotIntegration(DATASET_PATH, NUM_CLASSES)
    
    # Step 1: Generate honeypots
    integration.generate_honeypots(
        output_dir="./imagenet_honeypots",
        n_categories=50,
        per_class=100
    )
    
    # Step 2: Create dataloaders
    train_loader, val_loader = integration.create_dataloaders(batch_size=64)
    
    # Step 3: Load your pre-trained model
    # model = torch.load("your_model.pth")
    # model = model.to("cuda")
    
    # Step 4: Evaluate with honeypots
    # results = integration.evaluate_model(model, val_loader)
    # integration.print_results(results)
    
    print("Integration setup complete!")
    print("Next steps:")
    print("1. Load your trained model")
    print("2. Run integration.evaluate_model(model, val_loader)")
    print("3. Analyze robustness results")


if __name__ == "__main__":
    main()