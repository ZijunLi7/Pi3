import os
import utils3d
import numpy as np
import open3d as o3d
import glob
import torch
import argparse
from torch import Tensor
from functools import partial
import torch.nn.functional as F
from numpy.typing import NDArray
from typing import Tuple

#! reference: https://github.com/microsoft/MoGe/blob/main/moge/utils/geometry_torch.py#L115
#! reference: https://github.com/yyfz/Pi3/issues/2


def create_edge_mask(depth_map, threshold=0.1):
    edge_mask = utils3d.depth_map_edge(depth_map, rtol=threshold)
    depth_map[edge_mask.astype(bool)] = 0
    return depth_map


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    return optim_shift


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, 
                            dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None, focal: torch.Tensor = None, 
                       downsample_size: Tuple[int, int] = (64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0
    
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift


def w2c_to_c2w(Rt_w2c):
    """将世界坐标系到相机坐标系(W2C)的变换矩阵转换为相机坐标系到世界坐标系(C2W)"""
    is_torch = isinstance(Rt_w2c, torch.Tensor)
    R_w2c = Rt_w2c[:3, :3]
    t_w2c = Rt_w2c[:3, 3]
    
    if is_torch:
        R_c2w = R_w2c.T
        t_c2w = -torch.matmul(R_c2w, t_w2c.reshape(3, 1)).reshape(3)
        Rt_c2w = torch.zeros((3, 4), device=Rt_w2c.device, dtype=Rt_w2c.dtype)
    else:
        R_c2w = R_w2c.T
        t_c2w = -np.matmul(R_c2w, t_w2c.reshape(3, 1)).reshape(3)
        Rt_c2w = np.zeros((3, 4), dtype=Rt_w2c.dtype)
    
    Rt_c2w[:3, :3] = R_c2w
    Rt_c2w[:3, 3] = t_c2w
    return Rt_c2w


def transfer_torch2numpy(imgs, depth, intrinsics, extrinsics):
    "Transfer torch tensors to numpy arrays"
    imgs_np = imgs.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy() * 255
    imgs_np = np.asarray(imgs_np, dtype=np.uint8, order="C")
    depth_np = depth.squeeze(0).detach().cpu().numpy()
    intrinsics_np = intrinsics.squeeze(0).detach().cpu().numpy()
    extrinsics_np = extrinsics.squeeze(0).detach().cpu().numpy()
    return imgs_np, depth_np, intrinsics_np, extrinsics_np


class TSDFMeshExtrator:
    def __init__(self, depth_trunc: float = 4.0, voxel_length: float | None = None,
                 sdf_trunc: float | None = None, mesh_res: int = 1024, clean_depth: bool = False):
        voxel_length = depth_trunc / mesh_res if voxel_length is None else voxel_length
        sdf_trunc = 5.0 * voxel_length if sdf_trunc is None else sdf_trunc

        self._depth_trunc = depth_trunc
        self._voxel_length = voxel_length
        self._sdf_trunc = sdf_trunc
        self._clean_depth = clean_depth

        self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self._voxel_length,
            sdf_trunc=self._sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def add_frame(self, rgb: NDArray, depth: NDArray, intrinsics: NDArray, extrinsics: NDArray):
        rgb = np.asarray(rgb, dtype=np.uint8, order="C")
        depth = np.asarray(depth, dtype=np.float32, order="C")
        if self._clean_depth:
            depth = create_edge_mask(depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=self._depth_trunc,
            convert_rgb_to_intensity=False,
        )

        intrinsics = intrinsics.astype(np.float64)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            depth.shape[1], depth.shape[0],
            intrinsics[0, 0], intrinsics[1, 1],
            intrinsics[0, 2], intrinsics[1, 2],
        )
        
        extrinsics = extrinsics.astype(np.float64)
        extrinsics = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)

        self._volume.integrate(rgbd, intrinsics, extrinsics)

        pts3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, extrinsics)
        return pts3d

    def extract_and_save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mesh = self._volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh(path, mesh)


def parse_args():
    parser = argparse.ArgumentParser(description='Point cloud to mesh reconstruction with Pi3')
    parser.add_argument('--img_dir', type=str, default='examples/temp_traj1_img',
                       help='Input image directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--mesh_name', type=str, default='pt3d2mesh.ply',
                       help='Output mesh filename')
    parser.add_argument('--depth_trunc', type=float, default=None,
                       help='Depth truncation value (auto-computed if not set)')
    parser.add_argument('--voxel_length', type=float, default=0.01,
                       help='Voxel length for TSDF volume')
    parser.add_argument('--mesh_res', type=int, default=4096,
                       help='Mesh resolution')
    parser.add_argument('--clean_depth', action='store_true',
                       help='Clean depth edges')
    parser.add_argument('--fx', type=float, default=409.11,
                       help='Focal length x')
    parser.add_argument('--fy', type=float, default=409.51,
                       help='Focal length y')
    parser.add_argument('--save_pointclouds', action='store_true',
                       help='Save individual point clouds for each frame')
    
    # Reprojection error evaluation
    parser.add_argument('--eval_reprojection', action='store_true', default=True,
                       help='Evaluate reprojection error (default: True)')
    parser.add_argument('--match_dir', type=str, default=None,
                       help='Match files directory (auto-detect if not set)')
    parser.add_argument('--match_scale', type=float, default=0.35,
                       help='Match coordinate scale factor')
    parser.add_argument('--vis_interval', type=int, default=5,
                       help='Visualization interval for reprojection')
    parser.add_argument('--max_vis_error', type=float, default=20.0,
                       help='Maximum error for visualization color mapping')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    from pi3.models.pi3 import Pi3
    from pi3.utils.basic import load_images_as_tensor
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    imgs = load_images_as_tensor(args.img_dir).to(device)
    print(f"Loaded {len(imgs)} images from {args.img_dir}")
    
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
    fx_pixel = args.fx
    fy_pixel = args.fy
    intrinsics = utils3d.torch.intrinsics_from_focal_center(fx_pixel, fy_pixel, cx_pixel, cy_pixel).to(cam_points.device)
    intrinsics = intrinsics.unsqueeze(0).repeat(cam_points.shape[1], 1, 1)
    
    # Compute depth
    extrinsics = results["camera_poses"]
    depth = torch.einsum('bnij, bnhwj -> bnhwi', intrinsics.unsqueeze(0), cam_points)[..., 2]
    masks &= depth > 0
    depth[~masks] = 0.0
    
    # Convert to numpy
    imgs_np, depth_np, intrinsics_np, extrinsics_np = transfer_torch2numpy(imgs, depth, intrinsics, extrinsics)
    
    # Evaluate reprojection error
    if args.eval_reprojection:
        from reprojection_error import evaluate_reprojection_errors
        
        # Auto-detect match directory
        match_dir = args.match_dir
        if match_dir is None:
            base_name = os.path.basename(args.img_dir.rstrip('/'))
            match_dir = os.path.join(os.path.dirname(args.img_dir), f"{base_name.replace('_img', '_matches')}")
        
        if os.path.exists(match_dir):
            img_files = sorted(glob.glob(os.path.join(args.img_dir, '*.png')))
            img_ids = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
            
            evaluate_reprojection_errors(
                imgs=imgs_np,
                depth=depth_np,
                intrinsics=intrinsics_np,
                extrinsics=extrinsics_np,
                match_dir=match_dir,
                img_ids=img_ids,
                match_scale=args.match_scale,
                output_dir=args.output_dir,
                visualize_interval=args.vis_interval,
                max_vis_error=args.max_vis_error
            )
        else:
            print(f"\n⚠️  Match directory not found: {match_dir}")
            print("Skipping reprojection error evaluation")
    
    # Build TSDF mesh
    print("\n=== Building TSDF Mesh ===")
    depth_trunc = args.depth_trunc
    if depth_trunc is None:
        depth_trunc = np.percentile(depth_np[masks.squeeze().cpu().numpy()], 99)
        print(f"Auto-computed depth_trunc: {depth_trunc:.3f}")
    
    mesh_extractor = TSDFMeshExtrator(
        depth_trunc=depth_trunc,
        voxel_length=args.voxel_length,
        mesh_res=args.mesh_res,
        clean_depth=args.clean_depth,
    )
    
    img_files = sorted(glob.glob(os.path.join(args.img_dir, '*.png')))
    img_ids = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
    
    for i in range(len(imgs_np)):
        pts3d = mesh_extractor.add_frame(
            rgb=imgs_np[i],
            depth=depth_np[i],
            intrinsics=intrinsics_np[i],
            extrinsics=w2c_to_c2w(extrinsics_np[i]),
        )
        
        if args.save_pointclouds:
            pc_path = os.path.join(args.output_dir, f'pointcloud_{img_ids[i]}.ply')
            o3d.io.write_point_cloud(pc_path, pts3d)
    
    # Save mesh
    mesh_path = os.path.join(args.output_dir, args.mesh_name)
    mesh_extractor.extract_and_save(mesh_path)
    print(f"\n✅ Mesh saved to {mesh_path}")


if __name__ == "__main__":
    main()
