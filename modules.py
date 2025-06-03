import torch
import torch.nn as nn

###################################

class ImgBackboneModel(nn.Module):
    def __init__(self, bevfusion_model):
        super().__init__()
        self.backbone = bevfusion_model.img_backbone
        self.neck     = bevfusion_model.img_neck

    def forward(self, img):
        B, N, C, H, W = img.shape
        x = img.view(B * N, C, H, W).contiguous()
        x = self.backbone(x)
        x = self.neck(x)
        if not isinstance(x, torch.Tensor):
            x = x[0]
        
        _, C2, H2, W2 = x.shape
        x = x.view(B, N, C2, H2, W2)

        return x  # image feature

###################################

class ViewTransformFeatureModel(nn.Module):
    def __init__(self, bevfusion_model):
        super().__init__()
        self.view_transform = bevfusion_model.view_transform

    def forward(self, feats, depth):
        out = self.view_transform.get_cam_feats(feats, depth)
        return out

###################################

class ViewTransformBEVPoolModel(nn.Module):
    def __init__(self, bevfusion_model):
        super().__init__()
        self.view_transform = bevfusion_model.view_transform

    def forward(self, geom, feats):
        out = self.view_transform.bev_pool(geom, feats)
        return out

###################################

class ViewTransformDownsampleModel(nn.Module):
    def __init__(self, bevfusion_model):
        super().__init__()
        self.view_transform = bevfusion_model.view_transform

    def forward(self, feats):
        out = self.view_transform.downsample(feats)
        return out

###################################

class VoxelizationModel(nn.Module):
    def __init__(self, bevfusion_model):
        super().__init__()
        self.voxelize = bevfusion_model.voxelize

    def forward(self, points):
        points = [point.float() for point in points]
        feats, coords, _ = self.voxelize(points)
        return feats, coords

###################################

class LidarBackboneModel(nn.Module):
    def __init__(self, bevfusion_model):
        super().__init__()
        self.pts_middle_encoder = bevfusion_model.pts_middle_encoder

    def forward(self, feats, coords, batch_size):
        bev_feat = self.pts_middle_encoder(feats, coords, batch_size)
        return bev_feat

###################################

class FusionModel(nn.Module):
    def __init__(self, bevfusion_model: nn.Module):
        super().__init__()
        self.fuser = bevfusion_model.fusion_layer
        self.backbone = bevfusion_model.pts_backbone
        self.neck     = bevfusion_model.pts_neck

    def forward(self, img_bev, pts_bev):
        x = self.fuser([img_bev, pts_bev])
        x = self.backbone(x)
        x = self.neck(x)
        if not isinstance(x, torch.Tensor):
            x = x[0]
        return x

###################################

class HeadModel(nn.Module):
    def __init__(self, bevfusion_model):
        super().__init__()
        self.bbox_head = bevfusion_model.bbox_head

    def forward(self, feats):
        out, query_label = self.bbox_head(feats, None)
        out = out[0][0]
        out = {k : out[k] for k in ['center', 'height', 'dim', 'rot', 'vel', 'heatmap', 'query_heatmap_score']}
        out["query_label"] = query_label[0]
        return out