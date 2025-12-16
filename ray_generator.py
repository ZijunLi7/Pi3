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

    def get_rays_world_coords(self, lidar_origin):
        """
        Optional utility: If you have the lidar center point (x,y,z), 
        this returns the origin array repeated for the backend caster.
        
        Args:
            lidar_origin (list or np.array): [x, y, z] position of lidar center.
        """
        if self.rays_dir is None:
            self.generate_rays()
            
        N = self.rays_dir.shape[0]
        origins = np.tile(lidar_origin, (N, 1))
        return origins, self.rays_dir

# --- 使用示例 ---
if __name__ == "__main__":
    generator = LidarRayGenerator('hdl64e_config.json')
    
    directions = generator.generate_rays()
    lidar_center_pos = np.array([10.5, 5.0, 1.8]) # 假设 Lidar 在车顶
    
    origins, dirs = generator.get_rays_world_coords(lidar_center_pos)
    
    print(f"Ray Origins shape: {origins.shape}")     # (288000, 3)
    print(f"Ray Directions shape: {dirs.shape}")     # (288000, 3)
    
    norm = np.linalg.norm(dirs[0])
    print(f"Vector Norm Check: {norm:.4f}")