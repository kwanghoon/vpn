#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VPN-style Model Poisoning Checker (CLI)
---------------------------------------
- Algorithm 1: VPN (search over inputs, triggers, labels, with tolerance k)
- Algorithm 2: solve_trigger_for_label (heuristic patch-value search)

Usage (demo):
  python3 vpn_checker.py --mode demo --k 5 --s 2 --classes 3 --time-budget 1.0

Custom data:
  Prepare numpy arrays:
   - images: shape (N, H, W) uint8 (grayscale 0..255)
   - labels: shape (N,) int
  Then run:
  python3 vpn_checker.py --mode npy --images-npy /path/images.npy --labels-npy /path/labels.npy --k 3 --s 3 --classes 10
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import argparse
import numpy as np
import random
import time
import sys
import math
import os

# -------------------------------
# Types
# -------------------------------
Image = np.ndarray  # (H,W) uint8
Label = int

@dataclass(frozen=True)
class Trigger:
    r: int
    c: int
    s: int

@dataclass
class PoisoningWitness:
    trigger: Trigger
    values: np.ndarray   # (s,s) same dtype as images
    y_target: Label
    satisfied_fraction: float
    satisfied_count: int
    total: int

# -------------------------------
# Model interface + toy model
# -------------------------------
class AbstractModel:
    def labels(self) -> Sequence[Label]:
        raise NotImplementedError
    def predict(self, x: Image) -> Label:
        raise NotImplementedError

class ToyModel(AbstractModel):
    """Simple linear model over flattened grayscale images."""
    def __init__(self, image_shape: Tuple[int,int], num_classes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        H, W = image_shape
        D = H * W
        self.Wm = rng.normal(0, 0.02, size=(num_classes, D))
        self.bm = rng.normal(0, 0.1, size=(num_classes,))
        self._labels = list(range(num_classes))
        self.H, self.W = H, W

    def labels(self) -> Sequence[Label]:
        return self._labels

    def predict(self, x: Image) -> Label:
        v = x.reshape(-1).astype(np.float32) / 255.0
        scores = self.Wm @ v + self.bm
        return int(np.argmax(scores))

# -------------------------------
# Helpers
# -------------------------------
def apply_poison(x: Image, trigger: Trigger, values: np.ndarray) -> Image:
    r, c, s = trigger.r, trigger.c, trigger.s
    x2 = x.copy()
    x2[r:r+s, c:c+s] = values
    return x2

def enumerate_triggers(img_shape: Tuple[int,int], s: int) -> Iterable[Trigger]:
    H, W = img_shape
    for r in range(0, H - s + 1):
        for c in range(0, W - s + 1):
            yield Trigger(r, c, s)

def count_poison_successes(
    f: AbstractModel, T: Sequence[Image], trigger: Trigger, values: np.ndarray, y_target: Label
) -> int:
    ok = 0
    for x in T:
        if f.predict(apply_poison(x, trigger, values)) == y_target:
            ok += 1
    return ok

# -------------------------------
# Algorithm 2: Heuristic solver
# -------------------------------
class HeuristicPatchSolver:
    """
    Black-box search for patch values under tolerance k.
    - Priors (all-0, all-255, mean, checkerboard, stripes)
    - Hill-climb with random restarts
    """
    def solve(
        self,
        f: AbstractModel,
        T: Sequence[Image],
        k: int,
        x: Image,
        trigger: Trigger,
        y_target: Label,
        time_budget_s: float = 3.0,
        max_iters: int = 2000,
        value_domain: Tuple[int,int] = (0,255),
        rng: Optional[random.Random] = None,
    ) -> Optional[np.ndarray]:
        if rng is None:
            rng = random.Random(0xC0FFEE)
        lo, hi = value_domain
        s = trigger.s
        goal = len(T) - k

        def score(values: np.ndarray) -> int:
            return count_poison_successes(f, T, trigger, values, y_target)

        # Priors
        priors: List[np.ndarray] = []
        priors.append(np.full((s, s), lo, dtype=x.dtype))
        priors.append(np.full((s, s), hi, dtype=x.dtype))
        mean_val = int(np.clip(np.mean(x), lo, hi))
        priors.append(np.full((s, s), mean_val, dtype=x.dtype))
        cb = np.indices((s, s)).sum(axis=0) % 2
        priors.append((cb * hi + (1 - cb) * lo).astype(x.dtype))
        stripes = np.zeros((s, s), dtype=int)
        stripes[::2, :] = hi
        stripes[1::2, :] = lo
        priors.append(stripes.astype(x.dtype))

        best_values = None
        best_score = -1

        start_time = time.time()
        iters = 0

        def random_neighbor(vals: np.ndarray, edits: int = 1) -> np.ndarray:
            out = vals.copy()
            for _ in range(edits):
                rr = rng.randrange(s)
                cc = rng.randrange(s)
                delta = rng.choice([-64, -32, -16, 16, 32, 64])
                out[rr, cc] = np.clip(int(out[rr, cc]) + delta, lo, hi)
            return out

        # Evaluate priors
        for v0 in priors:
            sc = score(v0)
            if sc > best_score:
                best_score = sc
                best_values = v0.copy()
            if best_score >= goal:
                return best_values

        # Hill-climb with restarts
        while time.time() - start_time < time_budget_s and iters < max_iters:
            iters += 1
            if rng.random() < 0.10:
                cur = rng.choice(priors).copy()
            else:
                cur = best_values.copy() if best_values is not None else rng.choice(priors).copy()

            edits = 1 if iters < 200 else (2 if iters < 600 else 3)
            cand = random_neighbor(cur, edits=edits)
            sc = score(cand)
            if sc >= best_score:
                best_values = cand
                best_score = sc
                if best_score >= goal:
                    return best_values

        return None  # timeout/no solution

# -------------------------------
# Algorithm 1: VPN
# -------------------------------
def VPN(
    f: AbstractModel,
    T: Sequence[Image],
    k: int,
    s: int,
    solver: HeuristicPatchSolver,
    labels: Optional[Sequence[Label]] = None,
    time_budget_per_call_s: float = 3.0,
    max_iters_per_call: int = 2000,
    value_domain: Tuple[int,int] = (0,255),
    rng: Optional[random.Random] = None,
) -> Optional[PoisoningWitness]:
    if rng is None:
        rng = random.Random(1337)
    if labels is None:
        labels = list(f.labels())

    n_unsat = 0
    H, W = T[0].shape
    triggers = list(enumerate_triggers((H, W), s))

    for idx, x in enumerate(T):
        for trig in triggers:
            for y in labels:
                values = solver.solve(
                    f=f, T=T, k=k, x=x, trigger=trig, y_target=y,
                    time_budget_s=time_budget_per_call_s,
                    max_iters=max_iters_per_call,
                    value_domain=value_domain,
                    rng=rng,
                )
                if values is not None:
                    succ = count_poison_successes(f, T, trig, values, y)
                    frac = succ / len(T)
                    return PoisoningWitness(
                        trigger=trig,
                        values=values,
                        y_target=y,
                        satisfied_fraction=frac,
                        satisfied_count=succ,
                        total=len(T),
                    )
        n_unsat += 1
        if n_unsat > k:
            return None  # model poisoning free (under the explored space)

    return None

# -------------------------------
# Demo data + metrics
# -------------------------------
def make_synthetic_dataset(
    n: int = 60,
    image_shape: Tuple[int,int] = (8,8),
    num_classes: int = 3,
    seed: int = 7,
) -> Tuple[List[Image], List[Label]]:
    rng = np.random.default_rng(seed)
    H, W = image_shape
    T: List[Image] = []
    y: List[Label] = []
    for i in range(n):
        cls = i % num_classes
        base = rng.integers(0, 40, size=(H, W))
        if cls == 0:
            base[:, :W//2] += 40
        elif cls == 1:
            base[:H//2, :] += 40
        else:
            base[::2, ::2] += 60
        base = np.clip(base, 0, 255).astype(np.uint8)
        T.append(base)
        y.append(cls)
    return T, y

def evaluate_accuracy(f: AbstractModel, T: Sequence[Image], y: Sequence[Label]) -> float:
    correct = sum(int(f.predict(x)==yy) for x,yy in zip(T,y))
    return correct/len(T) if len(T) else 0.0

def display_synthetic_images(T: List[np.ndarray], y: List[int], num_samples: int = 9):
    """Display synthetic dataset images using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    num_samples = min(num_samples, len(T))
    grid_size = int(math.ceil(math.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle('Synthetic Dataset Examples', fontsize=16)
    
    # Handle single subplot case
    if grid_size == 1:
        axes = [axes]
    elif grid_size > 1:
        axes = axes.flatten()
    
    for i in range(grid_size * grid_size):
        ax = axes[i] if grid_size > 1 else axes[0]
        
        if i < num_samples:
            # Display image
            im = ax.imshow(T[i], cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'Class {y[i]} (Sample {i})', fontsize=10)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.6)
        else:
            # Hide empty subplots
            ax.axis('off')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    # Print dataset info
    print(f"\nDataset Information:")
    print(f"Total images: {len(T)}")
    print(f"Image shape: {T[0].shape}")
    print(f"Classes: {sorted(set(y))}")
    print("\nClass patterns:")
    print("Class 0: Left half brighter (vertical split)")
    print("Class 1: Top half brighter (horizontal split)")
    print("Class 2: Checkerboard pattern (every other pixel)")
    
    # Show pixel value statistics for each class
    for cls in sorted(set(y[:num_samples])):
        class_images = [T[i] for i in range(len(T)) if y[i] == cls]
        if class_images:
            avg_pixel = np.mean([np.mean(img) for img in class_images[:3]])
            print(f"Class {cls} average pixel value: {avg_pixel:.1f}")

def save_sample_images(T: List[np.ndarray], y: List[int], output_dir: str = "sample_images"):
    """Save sample images as individual files."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(min(9, len(T))):
        plt.figure(figsize=(4, 4))
        plt.imshow(T[i], cmap='gray', vmin=0, vmax=255)
        plt.title(f'Class {y[i]} Sample {i}')
        plt.axis('off')
        plt.colorbar(shrink=0.8)
        plt.savefig(f"{output_dir}/sample_{i}_class_{y[i]}.png", 
                   bbox_inches='tight', dpi=150)
        plt.close()
    
    print(f"Saved sample images to {output_dir}/ directory")

# -------------------------------
# CLI
# -------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="VPN-style Model Poisoning Checker (CLI)")
    ap.add_argument("--mode", choices=["demo", "npy"], default="demo",
                    help="demo: synthetic data; npy: load arrays")
    ap.add_argument("--images-npy", type=str, default=None,
                    help="Path to images.npy (shape (N,H,W), uint8)")
    ap.add_argument("--labels-npy", type=str, default=None,
                    help="Path to labels.npy (shape (N,), int)")
    ap.add_argument("--k", type=int, default=5, help="Max tolerated misses")
    ap.add_argument("--s", type=int, default=2, help="Trigger size s (s×s)")
    ap.add_argument("--classes", type=int, default=3, help="Number of classes (for demo/ToyModel)")
    ap.add_argument("--time-budget", type=float, default=1.0,
                    help="Time budget (seconds) per solve call")
    ap.add_argument("--iters", type=int, default=800, help="Max iterations per solve call")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed")
    ap.add_argument("--show-images", action="store_true", 
                    help="Display sample synthetic images (requires matplotlib)")
    ap.add_argument("--save-images", action="store_true",
                    help="Save sample images to files (requires matplotlib)")
    return ap.parse_args()

def load_npy(images_path: str, labels_path: str) -> Tuple[List[np.ndarray], List[int]]:
    if not os.path.exists(images_path):
        raise FileNotFoundError(images_path)
    if not os.path.exists(labels_path):
        raise FileNotFoundError(labels_path)
    X = np.load(images_path)
    y = np.load(labels_path)
    if X.ndim != 3:
        raise ValueError("images.npy must have shape (N,H,W)")
    if X.dtype != np.uint8:
        X = X.astype(np.uint8)
    if y.ndim != 1:
        raise ValueError("labels.npy must have shape (N,)")
    images = [X[i] for i in range(X.shape[0])]
    labels = [int(v) for v in y.tolist()]
    return images, labels

def main():
    args = parse_args()
    rng = random.Random(args.seed)

    if args.mode == "demo":
        H, W = 8, 8
        T, y = make_synthetic_dataset(n=60, image_shape=(H, W), num_classes=args.classes, seed=7)
        model = ToyModel(image_shape=(H, W), num_classes=args.classes, seed=0)
        print("=== Demo: VPN Model Poisoning Check ===")
        print(f"Dataset |T|={len(T)}, image=({H},{W}), classes={args.classes}")
        
        # Display sample images if requested
        if args.show_images:
            print("\nDisplaying sample synthetic images...")
            display_synthetic_images(T, y, num_samples=9)
        
        # Save sample images if requested
        if args.save_images:
            print("\nSaving sample images...")
            save_sample_images(T, y)
        
        acc = evaluate_accuracy(model, T, y)
        print(f"Baseline accuracy on T: {acc*100:.1f}%")
    else:
        if not args.images_npy or not args.labels_npy:
            print("ERROR: --mode npy requires --images-npy and --labels-npy", file=sys.stderr)
            sys.exit(2)
        T, y = load_npy(args.images_npy, args.labels_npy)
        H, W = T[0].shape
        # For external data, we still use ToyModel unless replaced; this just provides an f.
        # Replace with your DNN by implementing AbstractModel.predict.
        model = ToyModel(image_shape=(H, W), num_classes=args.classes, seed=0)
        print("=== NPY: VPN Model Poisoning Check ===")
        print(f"Loaded images: {len(T)} samples; image=({H},{W}); labels provided={len(y)}")
        acc = evaluate_accuracy(model, T, y)
        print(f"Baseline accuracy on provided T: {acc*100:.1f}% (ToyModel)")

    print(f"Parameters: k={args.k}, trigger size s={args.s}x{args.s}, time budget per solve={args.time_budget}s")
    solver = HeuristicPatchSolver()

    witness = VPN(
        f=model,
        T=T,
        k=args.k,
        s=args.s,
        solver=solver,
        labels=list(range(args.classes)),
        time_budget_per_call_s=args.time_budget,
        max_iters_per_call=args.iters,
        value_domain=(0,255),
        rng=rng,
    )

    if witness is None:
        print("\nResult: model poisoning free (under explored triggers/labels and budgets).")
        sys.exit(0)
    else:
        trig = witness.trigger
        print("\nResult: POISONING FOUND")
        print(f"- Trigger: top-left=({trig.r},{trig.c}), size={trig.s}x{trig.s}")
        print(f"- Target label: {witness.y_target}")
        print(f"- Satisfied: {witness.satisfied_count} / {witness.total} ({witness.satisfied_fraction*100:.1f}%)")
        print("- Patch values:")
        print(witness.values)
        # sanity re-check
        successes = count_poison_successes(model, T, trig, witness.values, witness.y_target)
        print(f"- Verified successes on T: {successes} / {len(T)}")
        sys.exit(0)

if __name__ == "__main__":
    main()
