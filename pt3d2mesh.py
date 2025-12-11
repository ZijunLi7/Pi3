import os
import utils3d
import numpy as np
import open3d as o3d
import glob, os, torch
from torch import Tensor
from functools import partial
import torch.nn.functional as F
from numpy.typing import NDArray
from collections import namedtuple
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
        xy_proj = xy / (z + shift)[: , None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[: , None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal

def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[: , None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    return optim_shift

def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
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

def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None, focal: torch.Tensor = None, downsample_size: Tuple[int, int] = (64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: torch.Tensor of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

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
    """
    将世界坐标系到相机坐标系(W2C)的变换矩阵转换为相机坐标系到世界坐标系(C2W)

    对于变换矩阵 Rt = [R | t]，其逆为：
    Rt^(-1) = [R^T | -R^T * t]

    Args:
        Rt_w2c: 3x4 或 4x4 的变换矩阵，从世界坐标系到相机坐标系

    Returns:
        相同大小的变换矩阵，从相机坐标系到世界坐标系
    """
    # 检查输入类型
    is_torch = isinstance(Rt_w2c, torch.Tensor)

    # 提取旋转矩阵和平移向量
    R_w2c = Rt_w2c[:3, :3]
    t_w2c = Rt_w2c[:3, 3]

    # 计算逆矩阵的旋转部分: R_c2w = R_w2c^T
    if is_torch:
        R_c2w = R_w2c.T
        # 计算逆矩阵的平移部分: t_c2w = -R_w2c^T * t_w2c
        t_c2w = -torch.matmul(R_c2w, t_w2c.reshape(3, 1)).reshape(3)
    else:
        R_c2w = R_w2c.T
        # 计算逆矩阵的平移部分: t_c2w = -R_w2c^T * t_w2c
        t_c2w = -np.matmul(R_c2w, t_w2c.reshape(3, 1)).reshape(3)

    # 构建C2W变换矩阵
    # 4x4矩阵
    if is_torch:
        Rt_c2w = torch.zeros((3, 4), device=Rt_w2c.device, dtype=Rt_w2c.dtype)
    else:
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
    def __init__(
        self,
        depth_trunc: float = 4.0,
        voxel_length: float | None = None,
        sdf_trunc: float | None = None,
        mesh_res: int = 1024,
        clean_depth: bool = False,
    ) -> None:
        voxel_length = depth_trunc / mesh_res if voxel_length is None else voxel_length
        sdf_trunc = 5.0 * voxel_length if sdf_trunc is None else sdf_trunc

        self._depth_trunc = depth_trunc
        self._voxel_length = voxel_length
        self._sdf_trunc = 5.0 * voxel_length
        self._clean_depth = clean_depth

        self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self._voxel_length,
            sdf_trunc=self._sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def add_frame(
        self,
        rgb: NDArray,
        depth: NDArray,
        intrinsics: NDArray,
        extrinsics: NDArray,
    ) -> None:
        rgb = np.asarray(rgb, dtype=np.uint8, order="C")
        depth = np.asarray(depth, dtype=np.float32, order="C")
        if self._clean_depth:
            depth = create_edge_mask(depth)
        
        # Ensure depth is contiguous
        # rgb = np.ascontiguousarray(rgb)
        # depth = np.ascontiguousarray(depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=self._depth_trunc,
            convert_rgb_to_intensity=False,
        )

        intrinsics = intrinsics.astype(np.float64)

        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            depth.shape[1],
            depth.shape[0],
            intrinsics[0, 0],
            intrinsics[1, 1],
            intrinsics[0, 2],
            intrinsics[1, 2],
        )
        
        extrinsics = extrinsics.astype(np.float64)
        extrinsics = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)

        self._volume.integrate(rgbd, intrinsics, extrinsics)

    def extract_and_save(self, path: str) -> None:
        base_dir = os.path.dirname(path)
        os.makedirs(base_dir, exist_ok=True)

        mesh = self._volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh(path, mesh)

if __name__ == "__main__":
    from pi3.models.pi3 import Pi3
    from pi3.utils.basic import load_images_as_tensor # Assuming you have a helper function

    # --- Configuration ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    imgs = load_images_as_tensor('examples/temp_traj2_img').to(device)

    # --- Inference ---
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            # Add a batch dimension -> (1, N, 3, H, W)
            results = model(imgs[None])
    print("Reconstruction complete!")

    # --- Process Results ---
    # result is the output from the Pi3 model
    cam_points = results["local_points"]
    masks = torch.sigmoid(results["conf"][..., 0]) > 0.1
    original_height, original_width = cam_points.shape[-3:-1]
    aspect_ratio = original_width / original_height
    # use recover_focal_shift function from MoGe
    focal, shift = recover_focal_shift(cam_points, masks)
    fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
    # Convert normalized focal lengths and principal points to pixel coordinates
    fx_pixel = fx * original_width
    fy_pixel = fy * original_height
    cx_pixel = 0.5 * original_width
    cy_pixel = 0.5 * original_height
    intrinsics = utils3d.torch.intrinsics_from_focal_center(fx_pixel, fy_pixel, cx_pixel, cy_pixel)
    cam_points[..., 2] += shift[..., None, None]
    masks &= cam_points[..., 2] > 0        # in case depth is contains negative values (which should never happen in practice)
    depth = cam_points[..., 2].clone()
    depth[~masks] = 0.0
    extrinsics = results["camera_poses"]
    imgs, depth, intrinsics, extrinsics = transfer_torch2numpy(imgs, depth, intrinsics, extrinsics)

    depth_thresh = np.percentile(depth[masks.squeeze().cpu().numpy()], 90)
    MeshExtractor = TSDFMeshExtrator(
        depth_trunc=depth_thresh,
        voxel_length=0.004,
        mesh_res=4096,
        clean_depth=True,
    )

    for i in range(len(imgs)):
        MeshExtractor.add_frame(
            rgb=imgs[i],
            depth=depth[i],
            intrinsics=intrinsics[i],
            extrinsics=w2c_to_c2w(extrinsics[i]),
        )

    save_path = "outputs/pt3d2mesh_traj2.ply"
    MeshExtractor.extract_and_save(save_path)

    