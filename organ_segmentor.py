"""
Organ Segmentation Module - with ROI filtering
Standalone file for TotalSegmentator integration

Save this as: organ_segmentor.py
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import pickle
import tempfile
from totalsegmentator.python_api import totalsegmentator


class OrganSegmentor:
    """
    Real-time organ segmentation using TotalSegmentator
    Supports filtering organs to display only those within ROI boundaries

    Usage:
        segmentor = OrganSegmentor()
        image_data, success = segmentor.segment_nifti('scan.nii.gz')
        organs = segmentor.get_organs_in_slice(slice_index, axis=0)
    """

    def __init__(self, cache_dir="./segmentation_cache"):
        """
        Initialize organ segmentor

        Args:
            cache_dir: Directory to cache segmentation results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.segmentation_mask = None
        self.original_mask = None  # Store original before ROI filtering
        self.organ_labels = None
        self.image_shape = None
        self.affine = None
        self.is_processing = False
        self.processing_complete = False

        # Comprehensive organ name mapping
        self.organ_names = {
            1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
            5: "liver", 6: "stomach", 7: "aorta", 8: "inferior_vena_cava",
            9: "portal_vein_splenic_vein", 10: "pancreas", 11: "adrenal_gland_right",
            12: "adrenal_gland_left", 13: "lung_upper_lobe_left", 14: "lung_lower_lobe_left",
            15: "lung_upper_lobe_right", 16: "lung_middle_lobe_right", 17: "lung_lower_lobe_right",
            18: "vertebrae_L5", 19: "vertebrae_L4", 20: "vertebrae_L3", 21: "vertebrae_L2",
            22: "vertebrae_L1", 23: "vertebrae_T12", 24: "vertebrae_T11", 25: "vertebrae_T10",
            26: "vertebrae_T9", 27: "vertebrae_T8", 28: "vertebrae_T7", 29: "vertebrae_T6",
            30: "vertebrae_T5", 31: "vertebrae_T4", 32: "vertebrae_T3", 33: "vertebrae_T2",
            34: "vertebrae_T1", 35: "vertebrae_C7", 36: "vertebrae_C6", 37: "vertebrae_C5",
            38: "vertebrae_C4", 39: "vertebrae_C3", 40: "vertebrae_C2", 41: "vertebrae_C1",
            42: "heart", 43: "esophagus", 44: "trachea", 45: "thyroid_gland",
            46: "small_bowel", 47: "duodenum", 48: "colon", 49: "urinary_bladder",
            50: "prostate", 51: "kidney_cyst_left", 52: "kidney_cyst_right",
            90: "brain", 91: "cerebellum", 92: "brainstem", 93: "ventricles",
            94: "white_matter", 95: "gray_matter", 96: "corpus_callosum",
            97: "thalamus", 98: "hippocampus", 99: "amygdala"
        }

    def segment_nifti(self, nifti_path, use_cache=True, fast=True, callback=None):
        """
        Load NIfTI file and run TotalSegmentator

        Args:
            nifti_path: Path to .nii or .nii.gz file
            use_cache: Use cached results if available (default True)
            fast: Use fast mode (default True)
            callback: Optional callback function(message) for progress updates

        Returns:
            tuple: (image_data: np.array, is_new: bool)
        """
        nifti_path = Path(nifti_path)
        cache_file = self.cache_dir / f"{nifti_path.stem}_segmentation.pkl"

        # Check cache first
        if use_cache and cache_file.exists():
            if callback:
                callback("Loading cached segmentation...")
            print("Loading cached segmentation...")

            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.segmentation_mask = cache_data['mask'].copy()
                    # Store original
                    self.original_mask = cache_data['mask'].copy()
                    self.organ_labels = cache_data['labels']
                    self.image_shape = cache_data['shape']
                    self.affine = cache_data.get('affine', np.eye(4))

                nii = nib.load(str(nifti_path))
                image_data = nii.get_fdata()

                print(
                    f"✓ Loaded from cache: {len(self.organ_labels)} organs found")
                self.processing_complete = True

                if callback:
                    callback(f"Cache loaded: {len(self.organ_labels)} organs")

                return image_data, False

            except Exception as e:
                print(
                    f"⚠ Cache load failed: {e}, running fresh segmentation...")
                if callback:
                    callback("Cache failed, running fresh segmentation...")

        # Load image
        if callback:
            callback("Loading NIfTI file...")
        print("Loading NIfTI file...")

        nii = nib.load(str(nifti_path))
        image_data = nii.get_fdata()
        self.image_shape = image_data.shape
        self.affine = nii.affine

        print(f"Image shape: {image_data.shape}")

        if callback:
            callback(f"Running TotalSegmentator...")
        print("Running TotalSegmentator...")

        # Run TotalSegmentator
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "segmentation.nii"

            try:
                self.is_processing = True

                if callback:
                    callback(
                        "Segmenting organs (this may take several minutes)...")

                # Run segmentation on FULL VOLUME
                totalsegmentator(
                    str(nifti_path),
                    str(output_path),
                    fast=fast,
                    ml=True
                )

                # Load segmentation result
                seg_nii = nib.load(str(output_path))
                mask = seg_nii.get_fdata().astype(np.uint8)

                self.segmentation_mask = mask.copy()
                self.original_mask = mask.copy()  # Store original

                # Get unique labels
                unique_labels = np.unique(self.segmentation_mask)
                self.organ_labels = {
                    label: self.organ_names.get(label, f"structure_{label}")
                    for label in unique_labels if label > 0
                }

                print(
                    f"✓ Segmentation complete! Found {len(self.organ_labels)} organs")

                # Cache results
                cache_data = {
                    'mask': mask,
                    'labels': self.organ_labels,
                    'shape': self.image_shape,
                    'affine': self.affine
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)

                print(f"Cache saved: {cache_file}")

                self.processing_complete = True
                self.is_processing = False

                if callback:
                    callback(
                        f"Complete! Found {len(self.organ_labels)} organs")

            except Exception as e:
                print(f"✗ Error during segmentation: {e}")
                self.segmentation_mask = np.zeros_like(
                    image_data, dtype=np.uint8)
                self.original_mask = np.zeros_like(image_data, dtype=np.uint8)
                self.organ_labels = {}
                self.is_processing = False
                self.processing_complete = True

                if callback:
                    callback(f"Error: {str(e)}")

                raise e

        return image_data, True

    def get_organs_in_slice(self, slice_index, axis=0, threshold=0.01):
        """
        Get list of organs present in a specific slice

        NOTE: roi_bounds parameter removed - ROI filtering now happens 
        via apply_roi_mask() before visualization

        Args:
            slice_index: Slice index to check
            axis: Axis of the slice (0=axial, 1=coronal, 2=sagittal)
            threshold: Minimum percentage of slice that must contain organ (default 0.01 = 1%)

        Returns:
            list: List of organ names present in the slice (human-readable, sorted)
        """
        if self.segmentation_mask is None:
            return []

        # Extract the slice based on axis
        try:
            if axis == 0:  # Axial (Z-axis)
                slice_mask = self.segmentation_mask[slice_index, :, :]
            elif axis == 1:  # Coronal (Y-axis)
                slice_mask = self.segmentation_mask[:, slice_index, :]
            elif axis == 2:  # Sagittal (X-axis)
                slice_mask = self.segmentation_mask[:, :, slice_index]
            else:
                return []
        except IndexError:
            return []

        # Find unique organs in this slice
        unique_labels, counts = np.unique(slice_mask, return_counts=True)

        total_pixels = slice_mask.size
        organs_present = []

        for label, count in zip(unique_labels, counts):
            if label == 0:  # Skip background
                continue

            percentage = count / total_pixels
            if percentage >= threshold:
                organ_name = self.organ_labels.get(label, f"structure_{label}")
                organ_display = organ_name.replace('_', ' ').title()
                organs_present.append(organ_display)

        return sorted(organs_present)

    def apply_roi_mask(self, roi_bounds):
        """
        Apply ROI bounds to filter the segmentation mask (in-place)

        Args:
            roi_bounds: Dict with keys 'axial', 'sagittal', 'coronal' 
                       containing (start, end) tuples
        """
        if self.original_mask is None:
            return

        # Start from original mask
        self.segmentation_mask = self.original_mask.copy()

        # Get ROI bounds
        z_start, z_end = roi_bounds.get(
            'axial', (0, self.segmentation_mask.shape[0]-1))
        y_start, y_end = roi_bounds.get(
            'coronal', (0, self.segmentation_mask.shape[1]-1))
        x_start, x_end = roi_bounds.get(
            'sagittal', (0, self.segmentation_mask.shape[2]-1))

        # Zero out everything OUTSIDE the ROI
        self.segmentation_mask[:z_start, :, :] = 0
        self.segmentation_mask[z_end+1:, :, :] = 0
        self.segmentation_mask[:, :y_start, :] = 0
        self.segmentation_mask[:, y_end+1:, :] = 0
        self.segmentation_mask[:, :, :x_start] = 0
        self.segmentation_mask[:, :, x_end+1:] = 0

        print(
            f"✓ ROI mask applied: Z[{z_start}-{z_end}] Y[{y_start}-{y_end}] X[{x_start}-{x_end}]")

    def reset_mask(self):
        """Reset segmentation mask to original (remove ROI filtering)"""
        if self.original_mask is not None:
            self.segmentation_mask = self.original_mask.copy()
            print("✓ Mask reset to original (ROI filter removed)")

    def get_mask(self):
        """Get the segmentation mask"""
        return self.segmentation_mask

    def get_organ_labels(self):
        """Get dictionary of detected organ labels"""
        return self.organ_labels

    def is_ready(self):
        """Check if segmentation is complete"""
        return self.processing_complete

    def clear_cache(self):
        """Clear all cached segmentation files"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()
            print(f"✓ Cache cleared: {self.cache_dir}")


def analyze_all_slices(self, axis=0, callback=None):
    """
    Analyze all slices and create CSV report

    Args:
        axis: Which axis to analyze (0=axial, 1=coronal, 2=sagittal)
        callback: Progress callback function

    Returns:
        OrganCSVAnalyzer: Analyzer with results
    """
    from organ_csv_analyzer import OrganCSVAnalyzer

    if not self.is_ready():
        print("⚠ Segmentation not complete yet")
        return None

    analyzer = OrganCSVAnalyzer()

    # Get total slices for this axis
    if axis == 0:
        total_slices = self.segmentation_mask.shape[0]
    elif axis == 1:
        total_slices = self.segmentation_mask.shape[1]
    else:
        total_slices = self.segmentation_mask.shape[2]

    if callback:
        callback(f"Analyzing {total_slices} slices...")

    print(f"Analyzing {total_slices} slices on axis {axis}...")

    # Analyze each slice
    for i in range(total_slices):
        organs = self.get_organs_in_slice(i, axis=axis)
        analyzer.record_slice(i, axis, organs)

        if callback and i % 10 == 0:
            callback(f"Analyzed {i}/{total_slices} slices...")

    if callback:
        callback(f"Analysis complete! {total_slices} slices analyzed")

    print(f"✓ Analysis complete!")
    return analyzer


if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Organ Segmentor - Standalone Test")
    print("=" * 60)

    segmentor = OrganSegmentor(cache_dir="./test_cache")
    print(f"Cache directory: {segmentor.cache_dir}")
    print(f"Known organs: {len(segmentor.organ_names)}")

    if len(sys.argv) > 1:
        nifti_path = sys.argv[1]
        print(f"\n[Test] Segmenting: {nifti_path}")

        def progress_callback(message):
            print(f"  → {message}")

        try:
            image_data, is_new = segmentor.segment_nifti(
                nifti_path,
                use_cache=True,
                fast=True,
                callback=progress_callback
            )

            print(f"\n  ✓ Segmentation complete!")
            print(f"    Image shape: {image_data.shape}")
            print(f"    Organs found: {len(segmentor.get_organ_labels())}")

            mid_slice = image_data.shape[0] // 2
            organs_full = segmentor.get_organs_in_slice(mid_slice, axis=0)
            print(
                f"\n  Full volume organs in slice {mid_slice}: {organs_full}")

            # Test ROI filtering
            roi_bounds = {
                'axial': (mid_slice-10, mid_slice+10),
                'sagittal': (0, image_data.shape[2]//2),
                'coronal': (0, image_data.shape[1]//2)
            }
            segmentor.apply_roi_mask(roi_bounds)
            organs_roi = segmentor.get_organs_in_slice(mid_slice, axis=0)
            print(f"  ROI-filtered organs in slice {mid_slice}: {organs_roi}")

        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n[Test] Skipped (no NIfTI file provided)")

    print("\n" + "=" * 60)
