import numpy as np
import torch

class Medical3DVolumeGenerator:
    def __init__(self, volume_shape=(32, 32, 32)):
        """
        Simulates 3D volumetric medical imagery arrays (e.g., NIfTI/DICOM MRI series)
        matching structural formats found in public datasets like BraTS.
        """
        self.shape = volume_shape

    def generate_synthetic_mri_scan(self):
        """Builds a synthetic 3D spatial scan matrix containing an inner tumor boundary."""
        # 1. Simulate background anatomical tissue intensity fields
        mri_volume = np.random.normal(loc=0.5, scale=0.1, size=self.shape)
        mri_volume = np.clip(mri_volume, 0.0, 1.0)

        # 2. Inject a synthetic high-intensity malignant mass within 3D coordinates
        mask_volume = np.zeros(self.shape, dtype=np.int64)
        
        # Define target spatial center points for tumor placement
        cx, cy, cz = self.shape[0] // 2, self.shape[1] // 2, self.shape[2] // 2
        radius = 6

        # Draw a solid 3D sphere mask representing tumor boundaries
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in range(self.shape[2]):
                    if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 < radius**2:
                        mask_volume[x, y, z] = 1
                        mri_volume[x, y, z] += np.random.uniform(0.3, 0.5) # Tumor contrast flare

        mri_volume = np.clip(mri_volume, 0.0, 1.0)
        
        # Convert raw NumPy arrays to PyTorch Tensor formats [Channels, Depth, Height, Width]
        mri_tensor = torch.tensor(mri_volume, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_volume, dtype=torch.long)
        
        return mri_tensor, mask_tensor

if __name__ == "__main__":
    generator = Medical3DVolumeGenerator()
    scan, mask = generator.generate_synthetic_mri_scan()
    print(f"✅ Spatial 3D Volumetric Array Initialized successfully!")
    print(f"👉 MRI Array Tensor Dimensions: {scan.shape}")
    print(f"👉 Tumor Mask Label Matrix Dimensions: {mask.shape}")
