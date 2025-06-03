import torch
import modules
from mmdet3d.apis import init_model
import warnings
import os

warnings.filterwarnings("ignore")
if not os.path.exists("onnx"):
    os.mkdir("onnx")
    
###################################

config     = "mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
checkpoint = "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
device     = "cuda:0"

model = init_model(config, checkpoint, device=device)
model.eval()

###################################

B, Ncam, C, H, W = 1, 6, 3, 256, 704
num_voxels = 15000
M = 32

dummy_imgs              = torch.randn(B, Ncam, C, H, W, device=device)
dummy_points            = torch.randn(B, num_voxels, 5, device=device)
dummy_img_features      = torch.randn(B, Ncam, 256, H // 8, W // 8, device=device)
dummy_depth             = torch.randn(B, Ncam, 1, H, W, device=device)
dummy_img_bev_feats1    = torch.randn(B, 80, 360, 360, device=device)
dummy_img_bev_feats     = torch.randn(B, 80, 180, 180, device=device)
dummy_pts_bev_feats     = torch.randn(B, 256, 180, 180, device=device)
dummy_fused_bev_feats   = torch.randn(B, 512, 180, 180, device=device)

###################################

img_backbone_model = modules.ImgBackboneModel(model)

torch.onnx.export(
    img_backbone_model,
    (dummy_imgs),
    "onnx/img_backbone.onnx",
    opset_version=13,
    input_names=["imgs"],
    output_names=["img_feats"],
    dynamic_axes={
        "imgs":      {0: "batch_size"},
        "img_feats": {0: "batch_size"},
    },
    do_constant_folding=False
)
print("✅ img_backbone.onnx saved")

###################################

viewtransform_feature_model = modules.ViewTransformFeatureModel(model)

torch.onnx.export(
    viewtransform_feature_model,
    (dummy_img_features, dummy_depth),
    "onnx/vtransform_feature.onnx",
    opset_version=13,
    input_names=["feats", "depth"],
    output_names=["bevpool_feats"],
    dynamic_axes={
        "feats":     {0: "batch_size"},
        "depth":     {0: "batch_size"},
        "bevpool_feats": {0: "batch_size"},
    },
    do_constant_folding=False
)
print("✅ vtransform_feature.onnx saved")

###################################

viewtransform_downsample_model = modules.ViewTransformDownsampleModel(model)

torch.onnx.export(
    viewtransform_downsample_model,
    (dummy_img_bev_feats1),
    "onnx/vtransform_downsample.onnx",
    opset_version=13,
    input_names=["feats"],
    output_names=["feats_downsampled"],
    dynamic_axes={
        "feats":                 {0: "batch_size"},
        "feats_downsampled":     {0: "batch_size"},
    },
    do_constant_folding=False
)
print("✅ vtransform_downsample.onnx saved")

###################################

fusion_model = modules.FusionModel(model)

torch.onnx.export(
    fusion_model,
    (dummy_img_bev_feats, dummy_pts_bev_feats),
    "onnx/fusion.onnx",
    input_names=['img_bev','pts_bev'],
    output_names=['fused_bev'],
    dynamic_axes={
      'img_bev':   {0:'batch'},
      'pts_bev':   {0:'batch'},
      'fused_bev': {0:'batch'}
    },
    opset_version=13,
)
print("✅ fusion.onnx saved")

###################################

head_model = modules.HeadModel(model)

torch.onnx.export(
    head_model,
    (dummy_fused_bev_feats),
    "onnx/head.onnx",
    input_names=['feats'],
    output_names=['reg', 'height', 'dim', 'rot', 'vel', 'heatmap', 'score', 'query_label'],
    dynamic_axes={
      'feats':   {0:'batch'},
      'reg':   {0:'batch'},
      'height': {0:'batch'},
      'dim':   {0:'batch'},
      'rot':   {0:'batch'},
      'vel': {0:'batch'},
      'heatmap':   {0:'batch'},
      'score':   {0:'batch'},
      'query_label':   {0:'batch'},
    },
    opset_version=13,
)
print("✅ head.onnx saved")