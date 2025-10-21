"""
Orientation Detection Module
Standalone file for detecting medical image orientation using ResNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom
import torchvision.models as models


class ResNetOrientation(nn.Module):
    """ResNet for orientation classification"""

    def __init__(self, num_classes=3):
        super(ResNetOrientation, self).__init__()
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=False)
        # Modify first conv layer for single-channel input
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify final FC layer for orientation classes
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)


class OrientationDetector:
    """
    Orientation detector for medical images using ResNet architecture

    Usage:
        detector = OrientationDetector('model.pth')
        orientation, confidence, probs = detector.predict(volume_array)
    """

    def __init__(self, model_path=None):
        """
        Initialize detector

        Args:
            model_path: Path to .pth model file (optional, can load later)
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNetOrientation(num_classes=3)
        self.loaded = False
        self.orientations = {
            0: 'axial',
            1: 'sagittal',
            2: 'coronal'
        }

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load trained model from file

        Args:
            model_path: Path to .pth model file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()
            self.model.to(self.device)

            # Update orientations from checkpoint if available
            if 'orientations' in checkpoint:
                self.orientations = checkpoint['orientations']

            self.loaded = True
            print(f"✓ Orientation model loaded on {self.device}")
            print(f"  Orientations: {list(self.orientations.values())}")
            return True

        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.loaded = False
            return False

    def preprocess_volume(self, volume, target_size=(224, 224)):
        """
        Preprocess volume for 2D ResNet input by taking central slices

        Args:
            volume: numpy array (Z, Y, X) or (Z, Y, X, C)
            target_size: target dimensions for ResNet input (default 224x224)

        Returns:
            torch.Tensor: preprocessed volume slices
        """
        # Handle 4D volumes (with channel dimension)
        if len(volume.shape) == 4:
            volume = volume[:, :, :, 0]

        # Extract central slices from each orientation
        z, y, x = volume.shape
        axial = volume[z//2, :, :]      # Top view
        sagittal = volume[:, y//2, :]    # Side view
        coronal = volume[:, :, x//2]     # Front view

        # Process each slice
        processed_slices = []
        for slice_img in [axial, sagittal, coronal]:
            # Resize to target size
            zoom_factors = [target_size[i] / slice_img.shape[i]
                            for i in range(2)]
            resized = zoom(slice_img, zoom_factors, order=1)

            # Normalize to [0, 1]
            normalized = (resized - resized.min()) / \
                (resized.max() - resized.min() + 1e-8)
            processed_slices.append(normalized)

        # Stack slices and convert to tensor (B, 1, H, W)
        tensor = torch.FloatTensor(np.stack(processed_slices)).unsqueeze(1)

        return tensor

    def predict(self, volume):
        """
        Predict orientation of volume using central slices

        Args:
            volume: numpy array (Z, Y, X) or (Z, Y, X, C)

        Returns:
            tuple: (orientation: str, confidence: float, probabilities: dict)
                   Returns (None, 0.0, {}) if model not loaded
        """
        if not self.loaded:
            print("⚠ Model not loaded. Call load_model() first.")
            return None, 0.0, {}

        try:
            # Preprocess volume to get central slices
            volume_tensor = self.preprocess_volume(volume)
            volume_tensor = volume_tensor.to(self.device)

            # Get predictions for each slice
            with torch.no_grad():
                outputs = self.model(volume_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = probabilities.max(1)

            # Use the prediction with highest confidence
            best_idx = confidence.argmax().item()
            predicted_class = predicted[best_idx].item()
            confidence_value = confidence[best_idx].item()
            orientation = self.orientations[predicted_class]

            # Create probability dictionary using the best slice
            prob_dict = {
                self.orientations[i]: probabilities[best_idx][i].item()
                for i in range(len(self.orientations))
            }

            return orientation, confidence_value, prob_dict

        except Exception as e:
            print(f"✗ Prediction error: {e}")
            return None, 0.0, {}

    def is_loaded(self):
        """Check if model is loaded"""
        return self.loaded

    def get_device(self):
        """Get current device (cuda/cpu)"""
        return str(self.device)


# ============================================================================
# STANDALONE TESTING
# ============================================================================
if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Orientation Detector - Standalone Test")
    print("=" * 60)

    # Test 1: Initialize detector
    print("\n[Test 1] Initialize detector...")
    detector = OrientationDetector()
    print(f"  Device: {detector.get_device()}")
    print(f"  Loaded: {detector.is_loaded()}")

    # Test 2: Load model
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"\n[Test 2] Loading model from: {model_path}")
        success = detector.load_model(model_path)
        if success:
            print("  ✓ Model loaded successfully")
        else:
            print("  ✗ Failed to load model")
            sys.exit(1)
    else:
        print("\n[Test 2] Skipped (no model path provided)")
        print("  Usage: python orientation_detector.py <model_path.pth>")
        sys.exit(0)

    # Test 3: Test with dummy data
    print("\n[Test 3] Testing with dummy volume...")
    dummy_volume = np.random.randn(
        100, 100, 100).astype(np.float32) * 100 + 500
    print(f"  Volume shape: {dummy_volume.shape}")

    orientation, confidence, probs = detector.predict(dummy_volume)

    if orientation:
        print(f"\n  ✓ Prediction successful!")
        print(f"    Orientation: {orientation}")
        print(f"    Confidence: {confidence*100:.1f}%")
        print(f"    All probabilities:")
        for orient, prob in probs.items():
            print(f"      {orient:12s}: {prob*100:5.1f}%")
    else:
        print("  ✗ Prediction failed")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
