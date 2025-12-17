import os
import open3d as o3d

from numpy.typing import NDArray

def ray_casting(mesh: o3d.geometry.TriangleMesh, rays: NDArray, max_distance: float | None = None) -> o3d.t.geometry.PointCloud:
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    mesh_id = scene.add_triangles(mesh_t)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays)
    hit = ans['t_hit'].isfinite() if max_distance is None else (ans['t_hit'].isfinite() & (ans['t_hit'] < max_distance))
    points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
    pcd = o3d.t.geometry.PointCloud(points)
    # todo: transfer ptc from world to lidar coordinates

    return pcd

if __name__ == "__main__":
    import glob
    import torch
    import utils3d
    import numpy as np
    from pt3d2mesh import transfer_torch2numpy, TSDFMeshExtrator, w2c_to_c2w

    img_dir = 'examples/temp_traj1_img'

    os.makedirs('outputs', exist_ok=True)
    from pi3.models.pi3 import Pi3
    from pi3.utils.basic import load_images_as_tensor

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    imgs = load_images_as_tensor(img_dir).to(device)
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    img_ids = [os.path.splitext(os.path.basename(f))[0] for f in img_files]

    # Run inference
    print("\nRunning model inference...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            results = model(imgs[None])
    print("Reconstruction complete!")

    # Process results
    cam_points = results["local_points"]
    masks = torch.sigmoid(results["conf"][..., 0]) > 0.1
    original_height, original_width = cam_points.shape[-3:-1]
    
    # Compute intrinsics
    cx_pixel = 0.5 * original_width
    cy_pixel = 0.5 * original_height
    fx_pixel = 409.11
    fy_pixel = 409.51
    intrinsics = utils3d.torch.intrinsics_from_focal_center(fx_pixel, fy_pixel, cx_pixel, cy_pixel).to(cam_points.device)
    intrinsics = intrinsics.unsqueeze(0).repeat(cam_points.shape[1], 1, 1)

    # Compute depth
    extrinsics = results["camera_poses"]
    depth = torch.einsum('bnij, bnhwj -> bnhwi', intrinsics.unsqueeze(0), cam_points)[..., 2]
    masks &= depth > 0
    depth[~masks] = 0.0
    
    # Convert to numpy
    imgs_np, depth_np, intrinsics_np, extrinsics_np = transfer_torch2numpy(imgs, depth, intrinsics, extrinsics)
    # Build TSDF mesh
    print("\n=== Building TSDF Mesh ===")
    depth_trunc = 0.99
    depth_trunc = np.percentile(depth_np[masks.squeeze().cpu().numpy()], 99)
    print(f"Auto-computed depth_trunc: {depth_trunc:.3f}")

    mesh_extractor = TSDFMeshExtrator(
        depth_trunc=depth_trunc,
        voxel_length=0.006,
        mesh_res=4096,
        clean_depth=True,
    )
    
    for i in range(len(imgs_np)):
        pts3d = mesh_extractor.add_frame(
            rgb=imgs_np[i],
            depth=depth_np[i],
            intrinsics=intrinsics_np[i],
            extrinsics=w2c_to_c2w(extrinsics_np[i]),
        )

        # Save mesh
    mesh_path = os.path.join('outputs', 'mesh.ply')
    mesh = mesh_extractor.extract_and_save(mesh_path)

    # generate rays for ray casting
    from ray_generator import LidarRayGenerator
    generator = LidarRayGenerator('lidar_config.json')
    directions = generator.generate_rays()
    lidar2cam = np.array([[0, 1, 0, 0],
                          [0, 0, -1, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
    lidar2world = np.einsum('bij,jk->bik', extrinsics_np, lidar2cam)
    for i, pose in enumerate(lidar2world):
        origins_world, dirs_world = generator.get_rays_world_coords(lidar_extrinsics = pose)
        rays = np.concatenate([origins_world, dirs_world], axis=-1).astype(np.float32)

        pts3d = ray_casting(mesh, rays)
        o3d.io.write_point_cloud(os.path.join('outputs', f'lidar_points_{img_ids[i]}.ply'), pts3d.to_legacy())

