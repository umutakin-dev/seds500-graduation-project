"""
Train hybrid diffusion model on Adult dataset.

Usage:
    python src/train_adult.py --epochs 500 --device cuda
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from diffusion import HybridDiffusion
from models import HybridMLPDenoiser


def load_adult_data():
    """Load and preprocess Adult dataset for hybrid diffusion."""
    print("Loading Adult dataset...")
    adult = fetch_openml("adult", version=2, as_frame=True)
    
    df = adult.data.copy()
    y = (adult.target == ">50K").astype(int).values
    
    # Identify numerical and categorical columns
    num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    
    # Handle missing values in categorical columns
    for col in cat_cols:
        df[col] = df[col].cat.add_categories("Missing").fillna("Missing")
    
    # Encode categorical features
    label_encoders = {}
    cat_cardinalities = []
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        cat_cardinalities.append(len(le.classes_))
    
    print(f"Numerical features: {len(num_cols)}")
    print(f"Categorical features: {len(cat_cols)}")
    print(f"Categorical cardinalities: {cat_cardinalities}")
    
    # Extract numerical and categorical data
    X_num = df[num_cols].values.astype(np.float32)
    X_cat = df[cat_cols].values.astype(np.int64)
    
    # Train/test split
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_num, X_cat, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numerical features
    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)
    X_num_train_scaled = np.clip(X_num_train_scaled, -3, 3) / 3
    X_num_test_scaled = np.clip(X_num_test_scaled, -3, 3) / 3
    
    # Convert categorical to one-hot
    def to_onehot(X_cat, cardinalities):
        batch_size = X_cat.shape[0]
        onehot_list = []
        for i, card in enumerate(cardinalities):
            onehot = np.zeros((batch_size, card), dtype=np.float32)
            onehot[np.arange(batch_size), X_cat[:, i]] = 1.0
            onehot_list.append(onehot)
        return np.hstack(onehot_list)
    
    X_cat_train_onehot = to_onehot(X_cat_train, cat_cardinalities)
    X_cat_test_onehot = to_onehot(X_cat_test, cat_cardinalities)
    
    # Concatenate: [numerical, categorical_onehot]
    X_train = np.hstack([X_num_train_scaled, X_cat_train_onehot])
    X_test = np.hstack([X_num_test_scaled, X_cat_test_onehot])
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train labels: {np.bincount(y_train)}")
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_num_train": X_num_train,
        "X_num_test": X_num_test,
        "X_cat_train": X_cat_train,
        "X_cat_test": X_cat_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_cardinalities": cat_cardinalities,
        "num_numerical": len(num_cols),
    }


def train(args):
    # Load data
    data = load_adult_data()
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    
    num_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_cardinalities"]
    
    # Create model and diffusion
    model = HybridMLPDenoiser(
        num_numerical=num_numerical,
        cat_cardinalities=cat_cardinalities,
        hidden_dims=[256, 256, 256],
        dropout=0.1,
    ).to(args.device)
    
    diffusion = HybridDiffusion(
        num_numerical=num_numerical,
        cat_cardinalities=cat_cardinalities,
        num_timesteps=1000,
        beta_schedule="linear",
    ).to(args.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Training loop
    print(f"
Training on {args.device}...")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    
    dataset = torch.utils.data.TensorDataset(X_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_loss_num = 0
        total_loss_cat = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            x_0 = batch[0].to(args.device)
            
            losses = diffusion.training_loss(model, x_0)
            loss = losses["loss"]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_loss_num += losses["loss_num"].item()
            total_loss_cat += losses["loss_cat"].item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        avg_loss_num = total_loss_num / len(dataloader)
        avg_loss_cat = total_loss_cat / len(dataloader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f} (num={avg_loss_num:.4f}, cat={avg_loss_cat:.4f})")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    # Save checkpoint
    checkpoint_dir = Path("checkpoints/adult")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_numerical": num_numerical,
            "cat_cardinalities": cat_cardinalities,
            "hidden_dims": [256, 256, 256],
        },
        "diffusion_config": {
            "num_timesteps": 1000,
            "beta_schedule": "linear",
        },
        "data_config": {
            "num_cols": data["num_cols"],
            "cat_cols": data["cat_cols"],
        },
        "final_loss": best_loss,
    }
    torch.save(checkpoint, checkpoint_dir / "final_model.pt")
    print(f"
Model saved to {checkpoint_dir / 'final_model.pt'}")
    print(f"Final loss: {best_loss:.4f}")
    
    # Save scaler and encoders for evaluation
    import pickle
    with open(checkpoint_dir / "preprocessors.pkl", "wb") as f:
        pickle.dump({
            "scaler": data["scaler"],
            "label_encoders": data["label_encoders"],
        }, f)
    print(f"Preprocessors saved to {checkpoint_dir / 'preprocessors.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    train(args)
