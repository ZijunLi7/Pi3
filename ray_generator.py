import json
import numpy as np
import os

class LidarRayGenerator:
    """
    A class to generate ray direction vectors for a specific LiDAR model 
    based on a JSON configuration file.
    """
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.rays_dir = None # Cached ray directions

    def _load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def generate_rays(self):
        """
        Generates the unit vectors for all rays in a single full scan (360 degrees).
        
        Returns:
            numpy.ndarray: A specialized array of shape (N, 3), where N is the 
                           total number of points per frame. Each row is (x, y, z).
        """
        # 1. Parsing Vertical Angles (Elevation)
        # Datasheet: "26.8 degree vertical field of view... +2 up to -24.8 down" 
        # Datasheet: "64 equally spaced angular subdivisions" 
        v_min = self.config['lasers']['fov_bottom']
        v_max = self.config['lasers']['fov_top']
        num_lasers = self.config['lasers']['count']
        
        # Note: Usually LiDARs scan from bottom to top or top to bottom. 
        # Using linspace to generate 64 equally spaced angles.
        elevation_angles = np.linspace(v_min, v_max, num_lasers)
        
        # 2. Parsing Horizontal Angles (Azimuth)
        # Datasheet: "360 degree field of view... 0.08 degree angular resolution" 
        h_start = self.config['horizontal']['fov_start']
        h_end = self.config['horizontal']['fov_end']
        h_res = self.config['horizontal']['resolution']
        
        # Using arange to generate horizontal steps (excluding end point usually 360=0)
        azimuth_angles = np.arange(h_start, h_end, h_res)
        
        # 3. Create Meshgrid for all combinations
        # We need to broadcast the vertical array against the horizontal array
        # elevation_grid shape: (num_lasers, num_horizontal_steps)
        # azimuth_grid shape: (num_lasers, num_horizontal_steps)
        azimuth_grid, elevation_grid = np.meshgrid(azimuth_angles, elevation_angles)
        
        # 4. Convert to Radians for Trigonometry
        az_rad = np.deg2rad(azimuth_grid)
        el_rad = np.deg2rad(elevation_grid)
        
        # 5. Spherical to Cartesian Conversion (Unit Vectors)
        # Coordinate system: X-forward, Y-left, Z-up (Standard Right-Handed used in ROS/Vision)
        # x = cos(elevation) * cos(azimuth)
        # y = cos(elevation) * sin(azimuth)
        # z = sin(elevation)
        
        x = np.cos(el_rad) * np.cos(az_rad)
        y = np.cos(el_rad) * np.sin(az_rad)
        z = np.sin(el_rad)
        
        # Stack to shape (num_lasers, num_horizontal_steps, 3)
        rays = np.stack((x, y, z), axis=-1)
        
        # Reshape to (N, 3) list of vectors for easy ray casting processing
        # N = num_lasers * horizontal_steps
        self.rays_dir = rays.reshape(-1, 3)
        
        print(f"Generated {self.rays_dir.shape[0]} rays based on {self.config['model_name']} specs.")
        return self.rays_dir

    def get_rays_world_coords(self, lidar_origin=None, lidar_extrinsics=None):
        """
        Transform ray directions and origins to world coordinates using lidar extrinsics.
        
        Args:
            lidar_origin (list or np.array, optional): [x, y, z] position of lidar center.
                If lidar_extrinsics is provided, this will be ignored.
            lidar_extrinsics (np.array, optional): 4x4 or 3x4 transformation matrix from 
                lidar frame to world frame. Format: [R|t] where R is 3x3 rotation and 
                t is 3x1 translation. If None, uses lidar_origin with identity rotation.
        
        Returns:
            origins (np.array): (N, 3) ray origins in world coordinates
            directions (np.array): (N, 3) ray directions in world coordinates (unit vectors)
        """
        if self.rays_dir is None:
            self.generate_rays()
        
        N = self.rays_dir.shape[0]
        
        # Parse extrinsics matrix
        if lidar_extrinsics is not None:
            extrinsics = np.asarray(lidar_extrinsics)
            if extrinsics.shape == (3, 4):
                R = extrinsics[:3, :3]
                t = extrinsics[:3, 3]
            elif extrinsics.shape == (4, 4):
                R = extrinsics[:3, :3]
                t = extrinsics[:3, 3]
            else:
                raise ValueError(f"lidar_extrinsics must be 3x4 or 4x4, got {extrinsics.shape}")
        elif lidar_origin is not None:
            # Identity rotation, only translation
            R = np.eye(3)
            t = np.asarray(lidar_origin).reshape(3)
        else:
            # No transformation, origin at [0,0,0]
            R = np.eye(3)
            t = np.zeros(3)
        
        # Transform ray directions: only apply rotation (directions are vectors, not points)
        dirs_world = (R @ self.rays_dir.T).T  # (N, 3)
        
        # Normalize to ensure unit vectors (in case rotation matrix has scaling)
        norms = np.linalg.norm(dirs_world, axis=1, keepdims=True)
        dirs_world = dirs_world / norms
        
        # Transform origin: apply rotation and translation
        origins_world = np.tile(t, (N, 1))  # (N, 3)
        
        return origins_world, dirs_world
    
    def visualize_rays(self, origins, directions, num_samples=1000, title="LiDAR Ray Visualization", 
                      ray_length=5.0, save_path=None):
        """
        Visualize LiDAR ray distribution
        
        Args:
            origins: (N, 3) ray origins
            directions: (N, 3) ray directions
            num_samples: number of rays to visualize (sampled from total)
            title: plot title
            ray_length: display length of rays
            save_path: save path (if None, will display instead)
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Random sampling to avoid overly dense visualization
        total_rays = origins.shape[0]
        if num_samples < total_rays:
            indices = np.random.choice(total_rays, num_samples, replace=False)
            origins_sample = origins[indices]
            directions_sample = directions[indices]
        else:
            origins_sample = origins
            directions_sample = directions
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw rays
        for i in range(len(origins_sample)):
            o = origins_sample[i]
            d = directions_sample[i]
            end = o + d * ray_length
            
            ax.plot([o[0], end[0]], [o[1], end[1]], [o[2], end[2]], 
                   'b-', alpha=0.3, linewidth=0.5)
        
        # Draw origin point
        ax.scatter(origins_sample[0, 0], origins_sample[0, 1], origins_sample[0, 2], 
                  c='r', s=100, marker='o', label='LiDAR Center')
        
        ax.set_xlabel('X (Forward)')
        ax.set_ylabel('Y (Left)')
        ax.set_zlabel('Z (Up)')
        ax.set_title(title)
        ax.legend()
        
        # Set equal axis ranges
        max_range = ray_length + 2
        ax.set_xlim([origins_sample[0, 0] - 2, origins_sample[0, 0] + max_range])
        ax.set_ylim([origins_sample[0, 1] - max_range/2, origins_sample[0, 1] + max_range/2])
        ax.set_zlim([origins_sample[0, 2] - max_range/2, origins_sample[0, 2] + max_range/2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_comparison(self, origins1, dirs1, origins2, dirs2, 
                           num_samples=500, ray_length=5.0, save_path=None):
        """
        Compare and visualize two sets of rays (e.g., before and after rotation)
        
        Args:
            origins1, dirs1: first set of rays
            origins2, dirs2: second set of rays
            num_samples: number of rays to sample from each set
            ray_length: display length of rays
            save_path: save path
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Sampling
        total_rays = origins1.shape[0]
        if num_samples < total_rays:
            indices = np.random.choice(total_rays, num_samples, replace=False)
            origins1_sample = origins1[indices]
            dirs1_sample = dirs1[indices]
            origins2_sample = origins2[indices]
            dirs2_sample = dirs2[indices]
        else:
            origins1_sample = origins1
            dirs1_sample = dirs1
            origins2_sample = origins2
            dirs2_sample = dirs2
        
        fig = plt.figure(figsize=(20, 8))
        
        # First set of rays
        ax1 = fig.add_subplot(131, projection='3d')
        for i in range(len(origins1_sample)):
            o = origins1_sample[i]
            d = dirs1_sample[i]
            end = o + d * ray_length
            ax1.plot([o[0], end[0]], [o[1], end[1]], [o[2], end[2]], 
                    'b-', alpha=0.3, linewidth=0.5)
        ax1.scatter(origins1_sample[0, 0], origins1_sample[0, 1], origins1_sample[0, 2], 
                   c='r', s=100, marker='o')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Original Rays')
        
        # Second set of rays
        ax2 = fig.add_subplot(132, projection='3d')
        for i in range(len(origins2_sample)):
            o = origins2_sample[i]
            d = dirs2_sample[i]
            end = o + d * ray_length
            ax2.plot([o[0], end[0]], [o[1], end[1]], [o[2], end[2]], 
                    'g-', alpha=0.3, linewidth=0.5)
        ax2.scatter(origins2_sample[0, 0], origins2_sample[0, 1], origins2_sample[0, 2], 
                   c='r', s=100, marker='o')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Transformed Rays')
        
        # Overlay display
        ax3 = fig.add_subplot(133, projection='3d')
        for i in range(len(origins1_sample)):
            o1 = origins1_sample[i]
            d1 = dirs1_sample[i]
            end1 = o1 + d1 * ray_length
            ax3.plot([o1[0], end1[0]], [o1[1], end1[1]], [o1[2], end1[2]], 
                    'b-', alpha=0.2, linewidth=0.5, label='Original' if i == 0 else '')
            
            o2 = origins2_sample[i]
            d2 = dirs2_sample[i]
            end2 = o2 + d2 * ray_length
            ax3.plot([o2[0], end2[0]], [o2[1], end2[1]], [o2[2], end2[2]], 
                    'g-', alpha=0.2, linewidth=0.5, label='Transformed' if i == 0 else '')
        
        ax3.scatter(origins1_sample[0, 0], origins1_sample[0, 1], origins1_sample[0, 2], 
                   c='b', s=100, marker='o', label='Origin 1')
        ax3.scatter(origins2_sample[0, 0], origins2_sample[0, 1], origins2_sample[0, 2], 
                   c='g', s=100, marker='o', label='Origin 2')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('Overlay Comparison')
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_ray_distribution(self, directions, save_path=None):
        """
        Visualize ray direction distribution (spherical projection)
        
        Args:
            directions: (N, 3) ray direction vectors
            save_path: save path
        """
        import matplotlib.pyplot as plt
        
        # Convert to spherical coordinates
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        azimuth = np.arctan2(y, x)  # azimuth angle
        elevation = np.arcsin(np.clip(z, -1, 1))  # elevation angle
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Azimuth-Elevation distribution
        ax1 = axes[0, 0]
        scatter = ax1.scatter(np.rad2deg(azimuth), np.rad2deg(elevation), 
                            c=np.rad2deg(elevation), cmap='viridis', s=1, alpha=0.5)
        ax1.set_xlabel('Azimuth (degrees)')
        ax1.set_ylabel('Elevation (degrees)')
        ax1.set_title('Ray Distribution (Azimuth vs Elevation)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Elevation (deg)')
        
        # 2. Azimuth histogram
        ax2 = axes[0, 1]
        ax2.hist(np.rad2deg(azimuth), bins=72, color='blue', alpha=0.7)
        ax2.set_xlabel('Azimuth (degrees)')
        ax2.set_ylabel('Count')
        ax2.set_title('Azimuth Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Elevation histogram
        ax3 = axes[1, 0]
        ax3.hist(np.rad2deg(elevation), bins=64, color='green', alpha=0.7, orientation='horizontal')
        ax3.set_ylabel('Elevation (degrees)')
        ax3.set_xlabel('Count')
        ax3.set_title('Elevation Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. XY plane projection
        ax4 = axes[1, 1]
        ax4.scatter(x, y, c=z, cmap='coolwarm', s=1, alpha=0.5)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_title('XY Plane Projection (colored by Z)')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(ax4.collections[0], ax=ax4, label='Z component')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Distribution visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

# --- Usage Examples ---
if __name__ == "__main__":
    generator = LidarRayGenerator('lidar_config.json')
    
    directions = generator.generate_rays()
    
    # Example 1: Position only (no rotation)
    print("\n=== Example 1: Position only ===")
    lidar_center_pos = np.array([10.5, 5.0, 1.8])  # Assume LiDAR is on vehicle roof
    origins, dirs = generator.get_rays_world_coords(lidar_origin=lidar_center_pos)
    
    print(f"Ray Origins shape: {origins.shape}")     # (288000, 3)
    print(f"Ray Directions shape: {dirs.shape}")     # (288000, 3)
    print(f"First origin: {origins[0]}")
    print(f"First direction: {dirs[0]}")
    norm = np.linalg.norm(dirs[0])
    print(f"Vector Norm Check: {norm:.4f}")
    
    # Example 2: Full extrinsics (position + rotation)
    print("\n=== Example 2: Full extrinsics (position + rotation) ===")
    # Create an extrinsics matrix with 45-degree rotation around Z-axis
    angle = np.deg2rad(45)
    R_z = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])
    t = np.array([10.5, 5.0, 1.8])
    
    # Build 3x4 extrinsics matrix
    extrinsics_3x4 = np.hstack([R_z, t.reshape(3, 1)])
    print(f"Extrinsics matrix (3x4):\n{extrinsics_3x4}")
    
    origins2, dirs2 = generator.get_rays_world_coords(lidar_extrinsics=extrinsics_3x4)
    print(f"\nRay Origins shape: {origins2.shape}")
    print(f"Ray Directions shape: {dirs2.shape}")
    print(f"First origin: {origins2[0]}")
    print(f"First direction (rotated): {dirs2[0]}")
    norm2 = np.linalg.norm(dirs2[0])
    print(f"Vector Norm Check: {norm2:.4f}")
    
    # Example 3: Using 4x4 extrinsics matrix
    print("\n=== Example 3: 4x4 extrinsics matrix ===")
    extrinsics_4x4 = np.vstack([extrinsics_3x4, np.array([[0, 0, 0, 1]])])
    print(f"Extrinsics matrix (4x4):\n{extrinsics_4x4}")
    
    origins3, dirs3 = generator.get_rays_world_coords(lidar_extrinsics=extrinsics_4x4)
    print(f"\nRay Origins shape: {origins3.shape}")
    print(f"First origin: {origins3[0]}")
    print(f"First direction: {dirs3[0]}")
    
    # Verify rotation is correctly applied
    print("\n=== Verification ===")
    print(f"Original direction (no rotation): {dirs[0]}")
    print(f"Rotated direction (45° around Z): {dirs2[0]}")
    
    # Angular difference between 3D vectors
    euclidean_dist = np.linalg.norm(dirs[0] - dirs2[0])
    cos_angle = np.dot(dirs[0], dirs2[0])
    angle_diff_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_diff_deg = np.rad2deg(angle_diff_rad)
    
    print(f"\n3D Vector Comparison:")
    print(f"  Euclidean distance: {euclidean_dist:.4f}")
    print(f"  Cosine similarity: {cos_angle:.4f}")
    print(f"  Angular difference (3D): {angle_diff_deg:.2f}° ({angle_diff_rad:.4f} rad)")
    
    # Angular difference in XY plane projection (this is the actual Z-axis rotation angle)
    xy_proj_orig = dirs[0][:2]  # [x, y]
    xy_proj_rot = dirs2[0][:2]
    
    # Calculate azimuth angle on XY plane
    azimuth_orig = np.arctan2(xy_proj_orig[1], xy_proj_orig[0])
    azimuth_rot = np.arctan2(xy_proj_rot[1], xy_proj_rot[0])
    azimuth_diff = azimuth_rot - azimuth_orig
    azimuth_diff_deg = np.rad2deg(azimuth_diff)
    
    print(f"\nXY Plane Projection (Azimuth):")
    print(f"  Original azimuth: {np.rad2deg(azimuth_orig):.2f}°")
    print(f"  Rotated azimuth: {np.rad2deg(azimuth_rot):.2f}°")
    print(f"  Azimuth difference: {azimuth_diff_deg:.2f}° (actual Z-axis rotation)")
    print(f"\nNote: 3D angular difference ≠ Z-axis rotation angle because the ray")
    print(f"      has a vertical component (elevation = {np.rad2deg(np.arcsin(dirs[0][2])):.2f}°)")
    
    # Visualization examples
    print("\n=== Generating Visualizations ===")
    
    # 1. Visualize original ray distribution
    print("1. Visualizing original ray distribution...")
    generator.visualize_ray_distribution(dirs, save_path='outputs/ray_distribution.png')
    
    # 2. Visualize 3D rays (original)
    print("2. Visualizing original 3D rays...")
    generator.visualize_rays(origins, dirs, num_samples=1000, 
                            title="Original LiDAR Rays (No Rotation)",
                            ray_length=3.0,
                            save_path='outputs/rays_3d_original.png')
    
    # 3. Visualize rotated 3D rays
    print("3. Visualizing rotated 3D rays...")
    generator.visualize_rays(origins2, dirs2, num_samples=1000,
                            title="Rotated LiDAR Rays (45° around Z-axis)",
                            ray_length=3.0,
                            save_path='outputs/rays_3d_rotated.png')
    
    # 4. Comparison visualization
    print("4. Visualizing comparison between original and rotated...")
    generator.visualize_comparison(origins, dirs, origins2, dirs2,
                                  num_samples=500,
                                  ray_length=3.0,
                                  save_path='outputs/rays_comparison.png')
    
    print("\n✅ All visualizations saved to 'outputs/' directory")
    print("   - ray_distribution.png: Ray direction distribution statistics")
    print("   - rays_3d_original.png: Original 3D rays")
    print("   - rays_3d_rotated.png: Rotated 3D rays")
    print("   - rays_comparison.png: Before/after comparison")