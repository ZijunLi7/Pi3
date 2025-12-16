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

def compute_reprojection_error(matches, depth1, depth2, intrinsics1, intrinsics2, extrinsics1, extrinsics2):
    """
    计算重投影误差
    
    Args:
        matches: (N, 4) 匹配对，前两维是前一帧的(x,y)，后两维是下一帧的(x,y)
        depth1: (H, W) 前一帧深度图
        depth2: (H, W) 下一帧深度图
        intrinsics1: (3, 3) 前一帧内参
        intrinsics2: (3, 3) 下一帧内参
        extrinsics1: (3, 4) 或 (4, 4) 前一帧外参 (W2C)
        extrinsics2: (3, 4) 或 (4, 4) 下一帧外参 (W2C)
    
    Returns:
        reprojection_errors: (N,) 每个匹配对的重投影误差
        valid_mask: (N,) 有效匹配的mask
        projected_points: (N, 2) 重投影后的点坐标
    """
    x1, y1 = matches[:, 0], matches[:, 1]
    x2_gt, y2_gt = matches[:, 2], matches[:, 3]
    
    # 转换为整数坐标用于索引深度图
    x1_int = np.clip(x1.astype(int), 0, depth1.shape[1] - 1)
    y1_int = np.clip(y1.astype(int), 0, depth1.shape[0] - 1)
    
    # 获取深度值
    z1 = depth1[y1_int, x1_int]
    
    # 过滤无效深度
    valid_mask = z1 > 0
    
    # 反投影到3D空间（相机坐标系）
    fx1, fy1 = intrinsics1[0, 0], intrinsics1[1, 1]
    cx1, cy1 = intrinsics1[0, 2], intrinsics1[1, 2]
    
    X1 = (x1 - cx1) * z1 / fx1
    Y1 = (y1 - cy1) * z1 / fy1
    Z1 = z1
    
    # 构建齐次坐标
    points_cam1 = np.stack([X1, Y1, Z1, np.ones_like(X1)], axis=1)  # (N, 4)
    
    # 转换到世界坐标系
    c2w1 = extrinsics1[:3, :]
    # 添加最后一行 [0, 0, 0, 1]
    c2w1_4x4 = np.vstack([c2w1, np.array([[0, 0, 0, 1]])])
    points_world = (c2w1_4x4 @ points_cam1.T).T  # (N, 4)
    
    # 转换到第二个相机坐标系
    w2c2 = w2c_to_c2w(extrinsics2[:3, :])
    w2c2_4x4 = np.vstack([w2c2, np.array([[0, 0, 0, 1]])])
    points_cam2 = (w2c2_4x4 @ points_world.T).T  # (N, 4)
    
    # 投影到第二帧图像
    fx2, fy2 = intrinsics2[0, 0], intrinsics2[1, 1]
    cx2, cy2 = intrinsics2[0, 2], intrinsics2[1, 2]
    
    X2, Y2, Z2 = points_cam2[:, 0], points_cam2[:, 1], points_cam2[:, 2]
    
    # 过滤深度为负的点
    valid_mask &= Z2 > 0
    
    x2_proj = fx2 * X2 / Z2 + cx2
    y2_proj = fy2 * Y2 / Z2 + cy2
    
    projected_points = np.stack([x2_proj, y2_proj], axis=1)
    
    # 计算重投影误差
    reprojection_errors = np.sqrt((x2_proj - x2_gt) ** 2 + (y2_proj - y2_gt) ** 2)
    
    return reprojection_errors, valid_mask, projected_points

def visualize_reprojection(img1, img2, matches, projected_points, errors, valid_mask, 
                          max_error=10.0, save_path=None):
    """
    可视化重投影结果
    
    Args:
        img1: (H, W, 3) 前一帧图像
        img2: (H, W, 3) 下一帧图像
        matches: (N, 4) 匹配对
        projected_points: (N, 2) 重投影后的点
        errors: (N,) 重投影误差
        valid_mask: (N,) 有效匹配mask
        max_error: float 用于颜色映射的最大误差值
        save_path: str 保存路径
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 显示两帧图像
    axes[0].imshow(img1)
    axes[0].set_title('Frame 1 (Source)')
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title('Frame 2 (Target) with Reprojection')
    axes[1].axis('off')
    
    # 在第一帧上绘制原始点
    x1, y1 = matches[valid_mask, 0], matches[valid_mask, 1]
    axes[0].scatter(x1, y1, c='green', s=20, alpha=0.6, label='Source points')
    
    # 在第二帧上绘制ground truth和重投影点
    x2_gt, y2_gt = matches[valid_mask, 2], matches[valid_mask, 3]
    x2_proj, y2_proj = projected_points[valid_mask, 0], projected_points[valid_mask, 1]
    valid_errors = errors[valid_mask]
    
    # 根据误差着色
    norm = plt.Normalize(vmin=0, vmax=max_error)
    colors = cm.jet(norm(valid_errors))
    
    # 绘制连线显示偏移
    for i in range(len(x2_gt)):
        axes[1].plot([x2_gt[i], x2_proj[i]], [y2_gt[i], y2_proj[i]], 
                    'r-', alpha=0.3, linewidth=0.5)
    
    # 绘制ground truth点（蓝色）和重投影点（根据误差着色）
    axes[1].scatter(x2_gt, y2_gt, c='blue', s=30, alpha=0.6, marker='o', label='GT matches')
    axes[1].scatter(x2_proj, y2_proj, c=colors, s=30, alpha=0.8, marker='x', label='Reprojected')
    
    # 添加颜色条
    sm = cm.ScalarMappable(cmap=cm.jet, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Reprojection Error (pixels)', rotation=270, labelpad=20)
    
    axes[0].legend()
    axes[1].legend()
    
    # 添加统计信息
    mean_error = np.mean(valid_errors)
    median_error = np.median(valid_errors)
    max_valid_error = np.max(valid_errors)
    
    stats_text = f'Valid matches: {np.sum(valid_mask)}/{len(matches)}\n'
    stats_text += f'Mean error: {mean_error:.2f} px\n'
    stats_text += f'Median error: {median_error:.2f} px\n'
    stats_text += f'Max error: {max_valid_error:.2f} px'
    
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return mean_error, median_error

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

        pts3d = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsics,
            extrinsics,
        )
        return pts3d

    def extract_and_save(self, path: str) -> None:
        base_dir = os.path.dirname(path)
        os.makedirs(base_dir, exist_ok=True)

        mesh = self._volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh(path, mesh)

if __name__ == "__main__":
    from pi3.models.pi3 import Pi3
    from pi3.utils.basic import load_images_as_tensor # Assuming you have a helper function

    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)

    # --- Configuration ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    imgs = load_images_as_tensor('examples/temp_traj1_img').to(device)

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
    # aspect_ratio = original_width / original_height
    # # use recover_focal_shift function from MoGe
    # focal, shift = recover_focal_shift(cam_points, masks)
    # fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
    # Convert normalized focal lengths and principal points to pixel coordinates
    # fx_pixel = fx * original_width
    # fy_pixel = fy * original_height
    cx_pixel = 0.5 * original_width
    cy_pixel = 0.5 * original_height
    fx_pixel = 409.11
    fy_pixel = 409.51
    intrinsics = utils3d.torch.intrinsics_from_focal_center(fx_pixel, fy_pixel, cx_pixel, cy_pixel).to(cam_points.device)
    intrinsics = intrinsics.unsqueeze(0).repeat(cam_points.shape[1], 1, 1)  # (1, B, 3, 3)
    # cam_points[..., 2] += shift[..., None, None]
    extrinsics = results["camera_poses"]
    depth = torch.einsum('bnij, bnhwj -> bnhwi', intrinsics.unsqueeze(0), cam_points)[..., 2]
    masks &= depth > 0        # in case depth is contains negative values (which should never happen in practice)
    depth[~masks] = 0.0
    imgs, depth, intrinsics, extrinsics = transfer_torch2numpy(imgs, depth, intrinsics, extrinsics)

    # --- 计算重投影误差 ---
    import glob
    match_dir = 'examples/temp_traj1_matches'  # 匹配文件所在目录
    img_dir = 'examples/temp_traj1_img'
    
    # 获取图像ID列表（按文件名排序）
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    img_ids = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
    
    print("\n=== Computing Reprojection Errors ===")
    all_errors = []
    
    for i in range(len(img_ids) - 1):
        im1_id = img_ids[i]
        im2_id = img_ids[i + 1]
        
        # 查找匹配文件（格式：{im1_id}_{random_id}.npy）
        match_files = glob.glob(os.path.join(match_dir, f'{im1_id}_*.npy'))
        
        if not match_files:
            print(f"Warning: No match file found for {im1_id}")
            continue
        
        # 如果有多个匹配文件，使用第一个
        match_file = match_files[0]
        matches = np.load(match_file)*0.35

        left = matches[:, :2]
        right = matches[:, 2:]
        boundary_mask = (left[:, 0] >= 0) & (left[:, 0] <= (depth[i].shape[1] - 1)) & \
            (left[:, 1] >= 0) & (left[:, 1] <= (depth[i].shape[0] - 1)) & \
            (right[:, 0] >= 0) & (right[:, 0] <= (depth[i + 1].shape[1] - 1)) & \
            (right[:, 1] >= 0) & (right[:, 1] <= (depth[i + 1].shape[0] - 1))
        matches = matches[boundary_mask]
        left = matches[:, :2]
        right = matches[:, 2:]
        depth_mask = (depth[i][left[:, 1].astype(int), left[:, 0].astype(int)] > 0) & \
            (depth[i + 1][right[:, 1].astype(int), right[:, 0].astype(int)] > 0)
        matches = matches[depth_mask]
        
        print(f"\nProcessing: {im1_id} -> {im2_id}")
        print(f"  Matches loaded: {len(matches)}")
        
        # 计算重投影误差
        errors, valid_mask, projected_points = compute_reprojection_error(
            matches=matches,
            depth1=depth[i],
            depth2=depth[i + 1],
            intrinsics1=intrinsics[i],
            intrinsics2=intrinsics[i + 1],
            extrinsics1=extrinsics[i],
            extrinsics2=extrinsics[i + 1]
        )
        
        valid_errors = errors[valid_mask]
        all_errors.extend(valid_errors.tolist())
        
        print(f"  Valid matches: {np.sum(valid_mask)}/{len(matches)}")
        print(f"  Mean error: {np.mean(valid_errors):.2f} px")
        print(f"  Median error: {np.median(valid_errors):.2f} px")
        print(f"  Max error: {np.max(valid_errors):.2f} px")
        
        # 可视化（每隔几帧可视化一次，避免生成太多图片）
        vis_save_path = f'outputs/reprojection_{im1_id}_to_{im2_id}.png'
        mean_err, median_err = visualize_reprojection(
            img1=imgs[i],
            img2=imgs[i + 1],
            matches=matches,
            projected_points=projected_points,
            errors=errors,
            valid_mask=valid_mask,
            max_error=20.0,
            save_path=vis_save_path
        )
    
    # 打印整体统计
    if all_errors:
        print("\n=== Overall Reprojection Error Statistics ===")
        print(f"Total valid matches: {len(all_errors)}")
        print(f"Mean error: {np.mean(all_errors):.2f} px")
        print(f"Median error: {np.median(all_errors):.2f} px")
        print(f"Std error: {np.std(all_errors):.2f} px")
        print(f"Min error: {np.min(all_errors):.2f} px")
        print(f"Max error: {np.max(all_errors):.2f} px")
    
    # --- 构建TSDF网格 ---
    print("\n=== Building TSDF Mesh ===")
    depth_thresh = np.percentile(depth[masks.squeeze().cpu().numpy()], 99)
    MeshExtractor = TSDFMeshExtrator(
        depth_trunc=depth_thresh,
        voxel_length=0.01,
        mesh_res=4096,
        clean_depth=True,
    )

    for i in range(len(imgs)):
        pts3d = MeshExtractor.add_frame(
            rgb=imgs[i],
            depth=depth[i],
            intrinsics=intrinsics[i],
            extrinsics=w2c_to_c2w(extrinsics[i]),
        )
        o3d.io.write_point_cloud(f"outputs/pt3d2mesh_traj1_{img_ids[i]}.ply", pts3d)

    save_path = "outputs/pt3d2mesh_traj1.ply"
    MeshExtractor.extract_and_save(save_path)
    print(f"\nMesh saved to {save_path}")

    