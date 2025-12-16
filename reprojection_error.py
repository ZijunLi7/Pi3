"""
重投影误差计算和可视化模块
"""
import os
import glob
import numpy as np
from typing import Tuple, List


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
    # 提取旋转矩阵和平移向量
    R_w2c = Rt_w2c[:3, :3]
    t_w2c = Rt_w2c[:3, 3]
    
    # 计算逆矩阵的旋转部分: R_c2w = R_w2c^T
    R_c2w = R_w2c.T
    # 计算逆矩阵的平移部分: t_c2w = -R_w2c^T * t_w2c
    t_c2w = -np.matmul(R_c2w, t_w2c.reshape(3, 1)).reshape(3)
    
    # 构建C2W变换矩阵
    Rt_c2w = np.zeros((3, 4), dtype=Rt_w2c.dtype)
    Rt_c2w[:3, :3] = R_c2w
    Rt_c2w[:3, 3] = t_c2w
    
    return Rt_c2w


def compute_reprojection_error(matches, depth1, depth2, intrinsics1, intrinsics2, 
                              extrinsics1, extrinsics2):
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


def filter_matches_by_boundary(matches, depth1_shape, depth2_shape):
    """过滤边界外的匹配点"""
    left = matches[:, :2]
    right = matches[:, 2:]
    boundary_mask = (left[:, 0] >= 0) & (left[:, 0] <= (depth1_shape[1] - 1)) & \
        (left[:, 1] >= 0) & (left[:, 1] <= (depth1_shape[0] - 1)) & \
        (right[:, 0] >= 0) & (right[:, 0] <= (depth2_shape[1] - 1)) & \
        (right[:, 1] >= 0) & (right[:, 1] <= (depth2_shape[0] - 1))
    return matches[boundary_mask]


def filter_matches_by_depth(matches, depth1, depth2):
    """过滤深度无效的匹配点"""
    left = matches[:, :2]
    right = matches[:, 2:]
    depth_mask = (depth1[left[:, 1].astype(int), left[:, 0].astype(int)] > 0) & \
        (depth2[right[:, 1].astype(int), right[:, 0].astype(int)] > 0)
    return matches[depth_mask]


def evaluate_reprojection_errors(imgs, depth, intrinsics, extrinsics, 
                                 match_dir: str, img_ids: List[str], 
                                 match_scale: float = 0.35,
                                 output_dir: str = 'outputs',
                                 visualize_interval: int = 5,
                                 max_vis_error: float = 20.0):
    """
    评估重投影误差
    
    Args:
        imgs: (N, H, W, 3) 图像数组
        depth: (N, H, W) 深度图数组
        intrinsics: (N, 3, 3) 内参矩阵
        extrinsics: (N, 3, 4) 外参矩阵
        match_dir: 匹配文件目录
        img_ids: 图像ID列表
        match_scale: 匹配坐标缩放系数
        output_dir: 输出目录
        visualize_interval: 可视化间隔（每N帧可视化一次）
        max_vis_error: 可视化时的最大误差值
    
    Returns:
        all_errors: 所有有效匹配的重投影误差列表
    """
    print("\n=== Computing Reprojection Errors ===")
    all_errors = []
    
    for i in range(len(img_ids) - 1):
        im1_id = img_ids[i]
        im2_id = img_ids[i + 1]
        
        # 查找匹配文件
        match_files = glob.glob(os.path.join(match_dir, f'{im1_id}_*.npy'))
        
        if not match_files:
            print(f"Warning: No match file found for {im1_id}")
            continue
        
        # 加载并缩放匹配
        match_file = match_files[0]
        matches = np.load(match_file) * match_scale
        
        # 过滤无效匹配
        matches = filter_matches_by_boundary(matches, depth[i].shape, depth[i + 1].shape)
        matches = filter_matches_by_depth(matches, depth[i], depth[i + 1])
        
        print(f"\nProcessing: {im1_id} -> {im2_id}")
        print(f"  Valid matches: {len(matches)}")
        
        if len(matches) == 0:
            print(f"  Skipped: no valid matches")
            continue
        
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
        if len(valid_errors) == 0:
            print(f"  Skipped: no valid reprojections")
            continue
            
        all_errors.extend(valid_errors.tolist())
        
        print(f"  Valid reprojections: {np.sum(valid_mask)}/{len(matches)}")
        print(f"  Mean error: {np.mean(valid_errors):.2f} px")
        print(f"  Median error: {np.median(valid_errors):.2f} px")
        print(f"  Max error: {np.max(valid_errors):.2f} px")
        
        # 可视化
        if i % visualize_interval == 0:
            vis_save_path = os.path.join(output_dir, f'reprojection_{im1_id}_to_{im2_id}.png')
            visualize_reprojection(
                img1=imgs[i],
                img2=imgs[i + 1],
                matches=matches,
                projected_points=projected_points,
                errors=errors,
                valid_mask=valid_mask,
                max_error=max_vis_error,
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
    else:
        print("\n⚠️  No valid reprojection errors computed")
    
    return all_errors
