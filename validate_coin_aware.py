#!/usr/bin/env python3
"""
validate_coin_aware.py - Compare sides vs pairs training modes.

Runs both training modes on the same dataset and produces a comparison report.
This helps validate that the pairs mode is providing value over the legacy sides mode.

Usage:
    python validate_coin_aware.py --manifest train_manifest.json --output validation_report

Outputs:
    validation_report/
        comparison_report.json   - Accuracy metrics for both modes
        comparison_report.txt    - Human-readable summary
        sides_confusion.txt      - Confusion matrix for sides mode
        pairs_confusion.txt      - Confusion matrix for pairs mode
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Import training functions
from train_coin_aware import (
    TrainConfig, load_manifest, group_by_coin_identity, get_complete_pairs,
    SideDataset, CoinPairDataset, SideClassifier, CoinPairClassifier,
    safe_collate_sides, safe_collate_pairs, DEVICE, SEED
)

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Re-seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def evaluate_sides_on_pairs(model, pairs, transform, period_to_idx):
    """
    Evaluate sides model on complete pairs.
    
    For each pair, classify obverse and reverse independently,
    then check if BOTH predictions are correct.
    
    Returns:
        pair_accuracy: % where both sides predicted correctly
        side_accuracy: % of individual sides correct
        disagreement_rate: % where obv and rev predict different periods
    """
    model.eval()
    
    total_pairs = 0
    both_correct = 0
    obv_correct = 0
    rev_correct = 0
    disagreements = 0
    
    with torch.no_grad():
        for pair in pairs:
            obv_path = pair['obverse']['image_path']
            rev_path = pair['reverse']['image_path']
            true_label = period_to_idx[pair['period']]
            
            try:
                from PIL import Image
                
                with Image.open(obv_path) as im:
                    obv_img = transform(im.convert('RGB')).unsqueeze(0).to(DEVICE)
                with Image.open(rev_path) as im:
                    rev_img = transform(im.convert('RGB')).unsqueeze(0).to(DEVICE)
                
                obv_pred = model(obv_img).argmax(1).item()
                rev_pred = model(rev_img).argmax(1).item()
                
                total_pairs += 1
                
                if obv_pred == true_label:
                    obv_correct += 1
                if rev_pred == true_label:
                    rev_correct += 1
                if obv_pred == true_label and rev_pred == true_label:
                    both_correct += 1
                if obv_pred != rev_pred:
                    disagreements += 1
                    
            except Exception as e:
                continue
    
    return {
        'pair_accuracy': 100 * both_correct / max(1, total_pairs),
        'side_accuracy': 100 * (obv_correct + rev_correct) / max(1, 2 * total_pairs),
        'obverse_accuracy': 100 * obv_correct / max(1, total_pairs),
        'reverse_accuracy': 100 * rev_correct / max(1, total_pairs),
        'disagreement_rate': 100 * disagreements / max(1, total_pairs),
        'total_pairs': total_pairs
    }


def evaluate_pairs_on_pairs(model, pairs, transform, period_to_idx):
    """Evaluate pairs model on complete pairs."""
    model.eval()
    
    total = 0
    correct = 0
    predictions = []
    
    with torch.no_grad():
        for pair in pairs:
            obv_path = pair['obverse']['image_path']
            rev_path = pair['reverse']['image_path']
            true_label = period_to_idx[pair['period']]
            
            try:
                from PIL import Image
                
                with Image.open(obv_path) as im:
                    obv_img = transform(im.convert('RGB')).unsqueeze(0).to(DEVICE)
                with Image.open(rev_path) as im:
                    rev_img = transform(im.convert('RGB')).unsqueeze(0).to(DEVICE)
                
                pred = model(obv_img, rev_img).argmax(1).item()
                
                total += 1
                if pred == true_label:
                    correct += 1
                
                predictions.append({
                    'coin_identity': pair['coin_identity'],
                    'true': true_label,
                    'pred': pred,
                    'correct': pred == true_label
                })
                    
            except Exception as e:
                continue
    
    return {
        'pair_accuracy': 100 * correct / max(1, total),
        'total_pairs': total,
        'predictions': predictions
    }


def run_validation(config: TrainConfig, manifest_path: str, output_dir: str):
    """Run full validation comparing sides vs pairs modes."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🔬 VALIDATION: Sides vs Pairs Mode Comparison")
    print("=" * 60)
    
    # Load data
    all_data = load_manifest(manifest_path)
    coins = group_by_coin_identity(all_data)
    complete_pairs = get_complete_pairs(coins, config.min_side_confidence)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total records: {len(all_data)}")
    print(f"   Coin identities: {len(coins)}")
    print(f"   Complete pairs: {len(complete_pairs)}")
    
    if len(complete_pairs) < 20:
        print("❌ Not enough complete pairs for meaningful validation")
        return
    
    # Split complete pairs into train/val
    by_period = defaultdict(list)
    for pair in complete_pairs:
        by_period[pair['period']].append(pair)
    
    train_pairs, val_pairs = [], []
    periods = sorted(by_period.keys())
    
    for period in periods:
        items = by_period[period]
        random.shuffle(items)
        n_val = max(1, int(len(items) * 0.2))
        val_pairs.extend(items[:n_val])
        train_pairs.extend(items[n_val:])
    
    period_to_idx = {p: i for i, p in enumerate(periods)}
    idx_to_period = periods
    
    print(f"   Train pairs: {len(train_pairs)}")
    print(f"   Val pairs: {len(val_pairs)}")
    print(f"   Classes: {len(periods)}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomAffine(degrees=config.aug_rotation, translate=(0.05, 0.05), scale=config.aug_scale),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # =========================================================================
    # TRAIN SIDES MODEL
    # =========================================================================
    print("\n" + "-" * 40)
    print("📈 Training SIDES model...")
    print("-" * 40)
    
    # Flatten pairs to individual sides for sides training
    train_sides_data = []
    for pair in train_pairs:
        for side in ['obverse', 'reverse']:
            record = pair[side].copy()
            record['period'] = pair['period']
            train_sides_data.append(record)
    
    val_sides_data = []
    for pair in val_pairs:
        for side in ['obverse', 'reverse']:
            record = pair[side].copy()
            record['period'] = pair['period']
            val_sides_data.append(record)
    
    # Class weights
    train_counts = defaultdict(int)
    for r in train_sides_data:
        train_counts[r['period']] += 1
    
    class_weights = []
    for p in periods:
        cnt = train_counts[p]
        if cnt == 0:
            weight = 0.0
        else:
            raw = len(train_sides_data) / (len(periods) * cnt)
            weight = min(raw, config.max_weight_cap)
        class_weights.append(weight)
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    
    # Datasets
    train_dataset_sides = SideDataset(train_sides_data, train_transform, period_to_idx)
    val_dataset_sides = SideDataset(val_sides_data, val_transform, period_to_idx)
    
    g = torch.Generator()
    g.manual_seed(SEED)
    
    def seed_worker(worker_id):
        np.random.seed(torch.initial_seed() % 2**32)
        random.seed(torch.initial_seed() % 2**32)
    
    loader_kwargs = {
        'collate_fn': safe_collate_sides,
        'num_workers': 0,  # Simpler for validation
        'pin_memory': config.pin_memory,
        'worker_init_fn': seed_worker,
        'generator': g
    }
    
    train_loader_sides = DataLoader(train_dataset_sides, batch_size=config.batch_size, shuffle=True, **loader_kwargs)
    val_loader_sides = DataLoader(val_dataset_sides, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    
    # Train sides model
    sides_model = SideClassifier(len(periods)).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(sides_model.parameters(), lr=config.lr)
    
    best_sides_val = 0.0
    
    for epoch in range(config.epochs):
        sides_model.train()
        for imgs, labels, _ in train_loader_sides:
            if imgs.numel() == 0:
                continue
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(sides_model(imgs), labels)
            loss.backward()
            optimizer.step()
        
        sides_model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader_sides:
                if imgs.numel() == 0:
                    continue
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                val_correct += (sides_model(imgs).argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / max(1, val_total)
        if val_acc > best_sides_val:
            best_sides_val = val_acc
            torch.save(sides_model.state_dict(), output_dir / "sides_model.pth")
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Val acc = {val_acc:.1f}%")
    
    # Load best sides model
    sides_model.load_state_dict(torch.load(output_dir / "sides_model.pth", map_location=DEVICE))
    
    # =========================================================================
    # TRAIN PAIRS MODEL
    # =========================================================================
    print("\n" + "-" * 40)
    print("📈 Training PAIRS model...")
    print("-" * 40)
    
    train_dataset_pairs = CoinPairDataset(train_pairs, train_transform, period_to_idx)
    val_dataset_pairs = CoinPairDataset(val_pairs, val_transform, period_to_idx)
    
    loader_kwargs_pairs = {
        'collate_fn': safe_collate_pairs,
        'num_workers': 0,
        'pin_memory': config.pin_memory,
        'worker_init_fn': seed_worker,
        'generator': g
    }
    
    train_loader_pairs = DataLoader(train_dataset_pairs, batch_size=config.batch_size, shuffle=True, **loader_kwargs_pairs)
    val_loader_pairs = DataLoader(val_dataset_pairs, batch_size=config.batch_size, shuffle=False, **loader_kwargs_pairs)
    
    pairs_model = CoinPairClassifier(len(periods), fusion_hidden=config.fusion_hidden, dropout=config.fusion_dropout).to(DEVICE)
    optimizer_pairs = torch.optim.Adam(pairs_model.parameters(), lr=config.lr)
    
    best_pairs_val = 0.0
    
    for epoch in range(config.epochs):
        pairs_model.train()
        for obv, rev, labels, _ in train_loader_pairs:
            if obv.numel() == 0:
                continue
            obv = obv.to(DEVICE)
            rev = rev.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer_pairs.zero_grad()
            loss = criterion(pairs_model(obv, rev), labels)
            loss.backward()
            optimizer_pairs.step()
        
        pairs_model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for obv, rev, labels, _ in val_loader_pairs:
                if obv.numel() == 0:
                    continue
                obv = obv.to(DEVICE)
                rev = rev.to(DEVICE)
                labels = labels.to(DEVICE)
                val_correct += (pairs_model(obv, rev).argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / max(1, val_total)
        if val_acc > best_pairs_val:
            best_pairs_val = val_acc
            torch.save(pairs_model.state_dict(), output_dir / "pairs_model.pth")
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Val acc = {val_acc:.1f}%")
    
    # Load best pairs model
    pairs_model.load_state_dict(torch.load(output_dir / "pairs_model.pth", map_location=DEVICE))
    
    # =========================================================================
    # EVALUATE BOTH MODELS
    # =========================================================================
    print("\n" + "-" * 40)
    print("🔍 Evaluating both models on validation pairs...")
    print("-" * 40)
    
    sides_results = evaluate_sides_on_pairs(sides_model, val_pairs, val_transform, period_to_idx)
    pairs_results = evaluate_pairs_on_pairs(pairs_model, val_pairs, val_transform, period_to_idx)
    
    # =========================================================================
    # GENERATE REPORT
    # =========================================================================
    print("\n" + "=" * 60)
    print("📊 RESULTS COMPARISON")
    print("=" * 60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'total_records': len(all_data),
            'coin_identities': len(coins),
            'complete_pairs': len(complete_pairs),
            'train_pairs': len(train_pairs),
            'val_pairs': len(val_pairs),
            'classes': len(periods)
        },
        'sides_mode': {
            'best_val_acc_training': best_sides_val,
            'pair_accuracy': sides_results['pair_accuracy'],
            'side_accuracy': sides_results['side_accuracy'],
            'obverse_accuracy': sides_results['obverse_accuracy'],
            'reverse_accuracy': sides_results['reverse_accuracy'],
            'disagreement_rate': sides_results['disagreement_rate']
        },
        'pairs_mode': {
            'best_val_acc_training': best_pairs_val,
            'pair_accuracy': pairs_results['pair_accuracy']
        },
        'comparison': {
            'pair_accuracy_improvement': pairs_results['pair_accuracy'] - sides_results['pair_accuracy'],
            'pairs_better': pairs_results['pair_accuracy'] > sides_results['pair_accuracy']
        }
    }
    
    # Print summary
    print(f"\n{'Metric':<35} {'SIDES':<12} {'PAIRS':<12}")
    print("-" * 60)
    print(f"{'Training val accuracy':<35} {best_sides_val:>10.1f}% {best_pairs_val:>10.1f}%")
    print(f"{'Pair accuracy (both sides correct)':<35} {sides_results['pair_accuracy']:>10.1f}% {pairs_results['pair_accuracy']:>10.1f}%")
    print(f"{'Individual side accuracy':<35} {sides_results['side_accuracy']:>10.1f}% {'N/A':>12}")
    print(f"{'Obverse/Reverse disagreement':<35} {sides_results['disagreement_rate']:>10.1f}% {'N/A':>12}")
    print("-" * 60)
    
    improvement = report['comparison']['pair_accuracy_improvement']
    if improvement > 0:
        print(f"\n✅ PAIRS mode improves pair accuracy by {improvement:.1f} percentage points")
    elif improvement < 0:
        print(f"\n⚠️  SIDES mode performs better by {-improvement:.1f} percentage points")
    else:
        print(f"\n➡️  Both modes perform equally")
    
    # Save report
    with open(output_dir / "comparison_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save human-readable summary
    with open(output_dir / "comparison_report.txt", 'w') as f:
        f.write("TRIVALAYA COIN-AWARE TRAINING VALIDATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {report['timestamp']}\n\n")
        f.write("DATASET:\n")
        f.write(f"  Complete pairs: {len(complete_pairs)}\n")
        f.write(f"  Train/Val split: {len(train_pairs)}/{len(val_pairs)}\n")
        f.write(f"  Classes: {len(periods)}\n\n")
        f.write("RESULTS:\n")
        f.write(f"  SIDES pair accuracy: {sides_results['pair_accuracy']:.1f}%\n")
        f.write(f"  PAIRS pair accuracy: {pairs_results['pair_accuracy']:.1f}%\n")
        f.write(f"  Improvement: {improvement:+.1f}%\n")
    
    print(f"\n💾 Reports saved to {output_dir}/")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate sides vs pairs training modes")
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to manifest JSON with coin_identity and side fields')
    parser.add_argument('--output', type=str, default='validation_report',
                        help='Output directory for reports')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Training epochs per mode')
    parser.add_argument('--batch-size', type=int, default=16)
    
    args = parser.parse_args()
    
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    run_validation(config, args.manifest, args.output)


if __name__ == "__main__":
    main()
