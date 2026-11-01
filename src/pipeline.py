import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import Medical3DVolumeGenerator

class VolumetricSegmenter3D(nn.Module):
    def __init__(self):
        super(VolumetricSegmenter3D, self).__init__()
        """Custom 3D Convolutional Architecture for Spatial Volumetric Ingestion."""
        # Feature extraction layer processing 3D voxel spaces
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        # Segmentation classifier mapping multi-scale features back to voxel label maps
        self.classifier = nn.Conv3d(16, 2, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        out = self.classifier(features)
        return out

def run_3d_segmentation_training():
    print("🚀 Initializing High-Throughput 3D Spatial Pipeline...")
    
    # Initialize generators and network architecture blocks
    data_engine = Medical3DVolumeGenerator()
    model = VolumetricSegmenter3D()
    
    # Define optimization strategies common in healthcare imaging (CrossEntropy)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Fetch 3D data volumes
    mri_volume, true_tumor_mask = data_engine.generate_synthetic_mri_scan()
    
    # Add a pseudo batch dimension [Batch, Channels, Depth, Height, Width]
    mri_volume = mri_volume.unsqueeze(0) 

    model.train()
    print("⚡ Training 3D Spatial Volumetric Segmentation Model loops...")
    
    for epoch in range(1, 11):
        optimizer.zero_grad()
        
        # Forward execution phase across 3D coordinates
        predictions = model(mri_volume)
        
        # Calculate loss bounds against target ground truth mask metrics
        loss = criterion(predictions, true_tumor_mask.unsqueeze(0))
        
        loss.backward()
        optimizer.step()
        
        print(f" 👉 Epoch [{epoch}/10] Finished -- Volumetric Voxel Loss: {loss.item():.6f}")

    print("\n⚡ Model Training Completed successfully!")
    
    # Evaluation phase to calculate metrics
    model.eval()
    with torch.no_grad():
        final_output = torch.softmax(model(mri_volume), dim=1)
        predicted_classes = torch.argmax(final_output, dim=1).squeeze(0)
        
        # Compute overlapping dice/accuracy representation metrics
        correct_voxels = (predicted_classes == true_tumor_mask).sum().item()
        total_voxels = true_tumor_mask.numel()
        voxel_accuracy = (correct_voxels / total_voxels) * 100
        
        print(f"📊 Final Volumetric Voxel-Wise Matrix Segmentation Accuracy: {voxel_accuracy:.2f}%")

if __name__ == "__main__":
    run_3d_segmentation_training()
    