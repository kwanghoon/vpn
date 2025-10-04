#!/usr/bin/env python3
"""
Test script to display synthetic images from vpn_checker
"""

import numpy as np
import sys

# Import from vpn_checker
from vpn_checker import make_synthetic_dataset, display_synthetic_images

def main():
    print("Testing synthetic image generation and display...")
    
    # Generate sample dataset
    T, y = make_synthetic_dataset(n=9, image_shape=(8, 8), num_classes=3, seed=7)
    
    print(f"Generated {len(T)} images")
    print(f"Image shape: {T[0].shape}")
    print(f"Labels: {y}")
    
    # Try to display images
    try:
        display_synthetic_images(T, y, num_samples=9)
    except ImportError as e:
        print(f"Cannot display images: {e}")
        print("Please install matplotlib: pip install matplotlib")
        
        # Fallback: print some image statistics
        print("\nImage statistics (fallback):")
        for i in range(min(3, len(T))):
            img = T[i]
            print(f"Image {i} (Class {y[i]}):")
            print(f"  Shape: {img.shape}")
            print(f"  Min/Max: {img.min()}/{img.max()}")
            print(f"  Mean: {img.mean():.1f}")
            print(f"  First row: {img[0]}")
            print()

if __name__ == "__main__":
    main()