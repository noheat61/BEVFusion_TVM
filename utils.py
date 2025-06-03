import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_depth(img, points, lidar2image, img_aug_matrix, lidar_aug_matrix):

    image_size = [256, 704]
    batch_size = len(points)
    depth = torch.zeros(batch_size, img.shape[1], 1,
                        *image_size).to(points[0].device)

    for b in range(batch_size):
        cur_coords = points[b][:, :3]
        cur_img_aug_matrix = img_aug_matrix[b]
        cur_lidar_aug_matrix = lidar_aug_matrix[b]
        cur_lidar2image = lidar2image[b]

        # inverse aug
        cur_coords -= cur_lidar_aug_matrix[:3, 3]
        cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
            cur_coords.transpose(1, 0))
        # lidar2image
        cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
        cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
        # get 2d coords
        dist = cur_coords[:, 2, :]
        cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
        cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

        # imgaug
        cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
        cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
        cur_coords = cur_coords[:, :2, :].transpose(1, 2)

        # normalize coords for grid sample
        cur_coords = cur_coords[..., [1, 0]]

        on_img = ((cur_coords[..., 0] < image_size[0])
                    & (cur_coords[..., 0] >= 0)
                    & (cur_coords[..., 1] < image_size[1])
                    & (cur_coords[..., 1] >= 0))
        for c in range(on_img.shape[0]):
            masked_coords = cur_coords[c, on_img[c]].long()
            masked_dist = dist[c, on_img[c]]
            depth = depth.to(masked_dist.dtype)
            depth[b, c, 0, masked_coords[:, 0],
                    masked_coords[:, 1]] = masked_dist

    return depth

def get_geometry(camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix):

    def create_frustum():
        iH, iW = [256, 704]
        fH, fW = [32, 88]
        dbound = [1.0, 60.0, 0.5]
        
        ds = (
            torch.arange(*dbound,
                         dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW))
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW,
                           dtype=torch.float).view(1, 1, fW).expand(D, fH, fW))
        ys = (
            torch.linspace(0, iH - 1, fH,
                           dtype=torch.float).view(1, fH, 1).expand(D, fH, fW))

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)
    
    intrins = camera_intrinsics[..., :3, :3]
    post_rots = img_aug_matrix[..., :3, :3]
    post_trans = img_aug_matrix[..., :3, 3]
    camera2lidar_rots = camera2lidar[..., :3, :3]
    camera2lidar_trans = camera2lidar[..., :3, 3]

    extra_rots = lidar_aug_matrix[..., :3, :3]
    extra_trans = lidar_aug_matrix[..., :3, 3]
    B, N, _ = camera2lidar_trans.shape

    # undo post-transformation
    # B x N x D x H x W x 3
    frustum = create_frustum().to(camera_intrinsics.device)
    points = frustum - post_trans.view(B, N, 1, 1, 1, 3)
    points = (
        torch.inverse(post_rots).view(B, N, 1, 1, 1, 3,
                                        3).matmul(points.unsqueeze(-1)))
    # cam_to_lidar
    points = torch.cat(
        (
            points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
            points[:, :, :, :, :, 2:3],
        ),
        5,
    )
    combine = camera2lidar_rots.matmul(torch.inverse(intrins))
    points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)


    points = (
        extra_rots.view(B, 1, 1, 1, 1, 3,
                        3).repeat(1, N, 1, 1, 1, 1, 1).matmul(
                            points.unsqueeze(-1)).squeeze(-1))\
                            
    points += extra_trans.view(B, 1, 1, 1, 1,
                                3).repeat(1, N, 1, 1, 1, 1)

    return points

def decode(heatmap_feat, score, rot, dim, center, height, vel, query_label):

    pc_range=[-54.0, -54.0]
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
    score_threshold=0.0
    out_size_factor=8
    voxel_size=[0.075, 0.075]

    heatmap = heatmap_feat.sigmoid()
    one_hot = F.one_hot(query_label.long(), num_classes=10).permute(1, 0)
    heatmap = heatmap * score * one_hot

    # class label
    final_preds = heatmap.max(0, keepdims=False).indices
    final_scores = heatmap.max(0, keepdims=False).values

    # change size to real world metric
    center[0, :] = center[0, :] * out_size_factor * voxel_size[0] + pc_range[0]
    center[1, :] = center[1, :] * out_size_factor * voxel_size[1] + pc_range[1]
    dim[0, :] = dim[0, :].exp()
    dim[1, :] = dim[1, :].exp()
    dim[2, :] = dim[2, :].exp()
    height = height - dim[2:3, :] * 0.5  # gravity center to bottom center
    rots, rotc = rot[0:1, :], rot[1:2, :]
    rot = torch.atan2(rots, rotc)

    final_box_preds = torch.cat([center, height, dim, rot, vel], dim=0).permute(1, 0)
    thresh_mask = final_scores > score_threshold

    post_center_range = torch.tensor(
        post_center_range, device=heatmap.device)
    mask = (final_box_preds[..., :3] >=
            post_center_range[:3]).all(1)
    mask &= (final_box_preds[..., :3] <=
                post_center_range[3:]).all(1)
    mask &= thresh_mask

    boxes3d = final_box_preds[mask]
    scores = final_scores[mask]
    labels = final_preds[mask]

    return boxes3d, scores, labels