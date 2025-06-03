import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import spconv.pytorch as spconv
import time

from ops import bev_pool
from ops import Voxelization
import utils
import cuda_utils
from spconv_onnx import ONNXParser, ONNXEngine

class BEVFusionOnnxRuntimeModel(nn.Module):
    def __init__(self, batch_size, device):
        super().__init__()
        
        self._timing = {
            'img_backbone':      {'sum': 0.0, 'count': 0},
            'vtransform_feat':   {'sum': 0.0, 'count': 0},
            'vtransform_bevpool':{'sum': 0.0, 'count': 0},
            'vtransform_down':   {'sum': 0.0, 'count': 0},
            'voxelization':      {'sum': 0.0, 'count': 0},
            'lidar_backbone':    {'sum': 0.0, 'count': 0},
            'fusion':            {'sum': 0.0, 'count': 0},
            'head':              {'sum': 0.0, 'count': 0},
        }

        self.device = device
        
        self.session1 = ort.InferenceSession("onnx/img_backbone.onnx", providers=["CUDAExecutionProvider"])
        self.io1 = self.session1.io_binding()
        self.img_features = torch.empty(batch_size, 6, 256, 32, 88, device=self.device)
        self.io1.bind_output(name="img_feats", device_type=self.device, device_id=0, element_type=np.float32, shape=self.img_features.shape, buffer_ptr=self.img_features.data_ptr())

        self.session2 = ort.InferenceSession("onnx/vtransform_feature.onnx", providers=["CUDAExecutionProvider"])
        self.io2 = self.session2.io_binding()
        self.bevpool_features = torch.empty(batch_size, 6, 118, 32, 88, 80, device=self.device)
        self.io2.bind_input(name="feats", device_type=self.device, device_id=0, element_type=np.float32, shape=self.img_features.shape, buffer_ptr=self.img_features.data_ptr())
        self.io2.bind_output(name="bevpool_feats", device_type=self.device, device_id=0, element_type=np.float32, shape=self.bevpool_features.shape, buffer_ptr=self.bevpool_features.data_ptr())

        self.bevpool_layer = bev_pool

        self.session3 = ort.InferenceSession("onnx/vtransform_downsample.onnx", providers=["CUDAExecutionProvider"])
        self.io3 = self.session3.io_binding()
        self.img_bev_downsampled = torch.empty(batch_size, 80, 180, 180, device=self.device)
        self.io3.bind_output(name="feats_downsampled", device_type=self.device, device_id=0, element_type=np.float32, shape=self.img_bev_downsampled.shape, buffer_ptr=self.img_bev_downsampled.data_ptr())

        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            voxel_size=[0.075, 0.075, 0.2],
            max_voxels=[120000, 160000]
        )
        self.voxel_layer = Voxelization(**voxelize_cfg)

        self.lidar_backbone_parser = ONNXParser("onnx/lidar_backbone.onnx")
        self.lidar_backbone_model = ONNXEngine(self.lidar_backbone_parser)

        self.session4 = ort.InferenceSession("onnx/fusion.onnx", providers=["CUDAExecutionProvider"])
        self.io4 = self.session4.io_binding()
        self.fused_features = torch.empty(batch_size, 512, 180, 180, device=self.device)
        self.io4.bind_input(name="img_bev", device_type=self.device, device_id=0, element_type=np.float32, shape=self.img_bev_downsampled.shape, buffer_ptr=self.img_bev_downsampled.data_ptr())
        self.io4.bind_output(name="fused_bev", device_type=self.device, device_id=0, element_type=np.float32, shape=self.fused_features.shape, buffer_ptr=self.fused_features.data_ptr())
        
        self.session5 = ort.InferenceSession("onnx/head.onnx", providers=["CUDAExecutionProvider"])
        self.io5 = self.session5.io_binding()
        self.reg = torch.empty(batch_size, 2, 200, device=self.device)
        self.height = torch.empty(batch_size, 1, 200, device=self.device)
        self.dim = torch.empty(batch_size, 3, 200, device=self.device)
        self.rot = torch.empty(batch_size, 2, 200, device=self.device)
        self.vel = torch.empty(batch_size, 2, 200, device=self.device)
        self.heatmap = torch.empty(batch_size, 10, 200, device=self.device)
        self.score = torch.empty(batch_size, 10, 200, device=self.device)
        self.query_label = torch.empty(batch_size, 200, device=self.device)
        self.io5.bind_input(name="feats", device_type=self.device, device_id=0, element_type=np.float32, shape=self.fused_features.shape, buffer_ptr=self.fused_features.data_ptr())
        self.io5.bind_output(name="reg", device_type=self.device, device_id=0, element_type=np.float32, shape=self.reg.shape, buffer_ptr=self.reg.data_ptr())
        self.io5.bind_output(name="height", device_type=self.device, device_id=0, element_type=np.float32, shape=self.height.shape, buffer_ptr=self.height.data_ptr())
        self.io5.bind_output(name="dim", device_type=self.device, device_id=0, element_type=np.float32, shape=self.dim.shape, buffer_ptr=self.dim.data_ptr())
        self.io5.bind_output(name="rot", device_type=self.device, device_id=0, element_type=np.float32, shape=self.rot.shape, buffer_ptr=self.rot.data_ptr())
        self.io5.bind_output(name="vel", device_type=self.device, device_id=0, element_type=np.float32, shape=self.vel.shape, buffer_ptr=self.vel.data_ptr())
        self.io5.bind_output(name="heatmap", device_type=self.device, device_id=0, element_type=np.float32, shape=self.heatmap.shape, buffer_ptr=self.heatmap.data_ptr())
        self.io5.bind_output(name="score", device_type=self.device, device_id=0, element_type=np.float32, shape=self.score.shape, buffer_ptr=self.score.data_ptr())
        self.io5.bind_output(name="query_label", device_type=self.device, device_id=0, element_type=np.float32, shape=self.query_label.shape, buffer_ptr=self.query_label.data_ptr())

    def get_avg_latencies(self):
        """Return a dict of average latencies in milliseconds."""
        return {
            name: (info['sum']/info['count']*1000.0 if info['count']>0 else 0.0)
            for name, info in self._timing.items()
        }

    def forward(self, imgs, points, metas):

        t0 = time.perf_counter()
        self.io1.bind_input(name="imgs", device_type=self.device, device_id=0, element_type=np.float32, shape=imgs.shape, buffer_ptr=imgs.data_ptr())
        self.io1.synchronize_inputs()
        self.session1.run_with_iobinding(self.io1)
        elapsed = time.perf_counter() - t0

        self._timing['img_backbone']['sum']   += elapsed
        self._timing['img_backbone']['count'] += 1

        ###################################

        depth = utils.calculate_depth(imgs, points, metas["lidar2img"], metas["img_aug_matrix"], metas["lidar_aug_matrix"])

        t0 = time.perf_counter()
        self.io2.bind_input(name="depth", device_type=self.device, device_id=0, element_type=np.float32, shape=depth.shape, buffer_ptr=depth.data_ptr())
        self.io2.synchronize_inputs()
        self.session2.run_with_iobinding(self.io2)
        elapsed = time.perf_counter() - t0
        self._timing['vtransform_feat']['sum']   += elapsed
        self._timing['vtransform_feat']['count'] += 1

        ###################################

        geom = utils.get_geometry(metas["cam2img"], metas["cam2lidar"], metas["img_aug_matrix"], metas["lidar_aug_matrix"])

        t0 = time.perf_counter()
        img_bev_features = cuda_utils.bev_pool(geom, self.bevpool_features[:len(points)], self.bevpool_layer)
        elapsed = time.perf_counter() - t0
        self._timing['vtransform_bevpool']['sum']   += elapsed
        self._timing['vtransform_bevpool']['count'] += 1

        ###################################

        t0 = time.perf_counter()
        self.io3.bind_input(name="feats", device_type=self.device, device_id=0, element_type=np.float32, shape=img_bev_features.shape, buffer_ptr=img_bev_features.data_ptr())
        self.io3.synchronize_inputs()
        self.session3.run_with_iobinding(self.io3)
        elapsed = time.perf_counter() - t0
        self._timing['vtransform_down']['sum']   += elapsed
        self._timing['vtransform_down']['count'] += 1

        ###################################

        t0 = time.perf_counter()
        pts_features, pts_coords, _ = cuda_utils.voxelize(points, self.voxel_layer)
        elapsed = time.perf_counter() - t0
        self._timing['voxelization']['sum']   += elapsed
        self._timing['voxelization']['count'] += 1

        ###################################

        t0 = time.perf_counter()
        input_sp_tensor = spconv.SparseConvTensor(pts_features, pts_coords, [1440, 1440, 41], len(points))
        inputs = {self.lidar_backbone_parser.inputs[0]: input_sp_tensor}
        outputs = self.lidar_backbone_model.run(inputs)
        pts_bev_features = outputs[self.lidar_backbone_parser.outputs[0]]
        elapsed = time.perf_counter() - t0
        self._timing['lidar_backbone']['sum']   += elapsed
        self._timing['lidar_backbone']['count'] += 1

        ###################################

        t0 = time.perf_counter()
        self.io4.bind_input(name="pts_bev", device_type=self.device, device_id=0, element_type=np.float32, shape=pts_bev_features.shape, buffer_ptr=pts_bev_features.data_ptr())
        self.io4.synchronize_inputs()
        self.session4.run_with_iobinding(self.io4)
        elapsed = time.perf_counter() - t0
        self._timing['fusion']['sum']   += elapsed
        self._timing['fusion']['count'] += 1

        ###################################

        t0 = time.perf_counter()
        self.io5.synchronize_inputs()
        self.session5.run_with_iobinding(self.io5)
        elapsed = time.perf_counter() - t0
        self._timing['head']['sum']   += elapsed
        self._timing['head']['count'] += 1

        ###################################

        bboxes_3ds = list()
        scores_3ds = list()
        labels_3ds = list()

        for B in range(len(points)):
            bboxes_3d, scores_3d, labels_3d = utils.decode(
                self.heatmap[B],
                self.score[B],
                self.rot[B],
                self.dim[B],
                self.reg[B],
                self.height[B],
                self.vel[B],
                self.query_label[B],
            )
            bboxes_3ds.append(bboxes_3d)
            scores_3ds.append(scores_3d)
            labels_3ds.append(labels_3d)

        return bboxes_3ds, scores_3ds, labels_3ds

#########################################################

import tensorrt as trt

class BEVFusionTensorRTModel(nn.Module):
    def __init__(self, batch_size, device):
        super().__init__()
        
        self._timing = {
            'img_backbone':      {'sum': 0.0, 'count': 0},
            'vtransform_feat':   {'sum': 0.0, 'count': 0},
            'vtransform_bevpool':{'sum': 0.0, 'count': 0},
            'vtransform_down':   {'sum': 0.0, 'count': 0},
            'voxelization':      {'sum': 0.0, 'count': 0},
            'lidar_backbone':    {'sum': 0.0, 'count': 0},
            'fusion':            {'sum': 0.0, 'count': 0},
            'head':              {'sum': 0.0, 'count': 0},
        }

        self.device = device

        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(self.TRT_LOGGER)

        with open("onnx/img_backbone.engine", "rb") as f:
            self.engine1 = self.trt_runtime.deserialize_cuda_engine(f.read())
        self.context1 = self.engine1.create_execution_context()
        self.img_features = torch.empty(batch_size, 6, 256, 32, 88, device=self.device)

        self.bindingmap1 = {binding : i for i, binding in enumerate(self.engine1)}
        self.binding1 = [0] * len(self.bindingmap1)        
        self.binding1[self.bindingmap1["img_feats"]] = int(self.img_features.data_ptr())

        with open("onnx/vtransform_feature.engine", "rb") as f:
            self.engine2 = self.trt_runtime.deserialize_cuda_engine(f.read())
        self.context2 = self.engine2.create_execution_context()
        self.bevpool_features = torch.empty(batch_size, 6, 118, 32, 88, 80, device=self.device)

        self.bindingmap2 = {binding : i for i, binding in enumerate(self.engine2)}
        self.binding2 = [0] * len(self.bindingmap2)        
        self.binding2[self.bindingmap2["feats"]] = int(self.img_features.data_ptr())
        self.binding2[self.bindingmap2["bevpool_feats"]] = int(self.bevpool_features.data_ptr())

        self.bevpool_layer = bev_pool

        with open("onnx/vtransform_downsample.engine", "rb") as f:
            self.engine3 = self.trt_runtime.deserialize_cuda_engine(f.read())
        self.context3 = self.engine3.create_execution_context()
        self.img_bev_downsampled = torch.empty(batch_size, 80, 180, 180, device=self.device)

        self.bindingmap3 = {binding : i for i, binding in enumerate(self.engine3)}
        self.binding3 = [0] * len(self.bindingmap3)        
        self.binding3[self.bindingmap3["feats_downsampled"]] = int(self.img_bev_downsampled.data_ptr())

        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            voxel_size=[0.075, 0.075, 0.2],
            max_voxels=[120000, 160000]
        )
        self.voxel_layer = Voxelization(**voxelize_cfg)

        self.lidar_backbone_parser = ONNXParser("onnx/lidar_backbone.onnx")
        self.lidar_backbone_model = ONNXEngine(self.lidar_backbone_parser)

        with open("onnx/fusion.engine", "rb") as f:
            self.engine4 = self.trt_runtime.deserialize_cuda_engine(f.read())
        self.context4 = self.engine4.create_execution_context()
        self.fused_features = torch.empty(batch_size, 512, 180, 180, device=self.device)

        self.bindingmap4 = {binding : i for i, binding in enumerate(self.engine4)}
        self.binding4 = [0] * len(self.bindingmap4)        
        self.binding4[self.bindingmap4["img_bev"]] = int(self.img_bev_downsampled.data_ptr())
        self.binding4[self.bindingmap4["fused_bev"]] = int(self.fused_features.data_ptr())

        with open("onnx/head.engine", "rb") as f:
            self.engine5 = self.trt_runtime.deserialize_cuda_engine(f.read())
        self.context5 = self.engine5.create_execution_context()
        self.reg = torch.empty(batch_size, 2, 200, device=self.device)
        self.height = torch.empty(batch_size, 1, 200, device=self.device)
        self.dim = torch.empty(batch_size, 3, 200, device=self.device)
        self.rot = torch.empty(batch_size, 2, 200, device=self.device)
        self.vel = torch.empty(batch_size, 2, 200, device=self.device)
        self.heatmap = torch.empty(batch_size, 10, 200, device=self.device)
        self.score = torch.empty(batch_size, 10, 200, device=self.device)
        self.query_label = torch.empty(batch_size, 200, device=self.device)

        self.bindingmap5 = {binding : i for i, binding in enumerate(self.engine5)}
        self.binding5 = [0] * len(self.bindingmap5)        
        self.binding5[self.bindingmap5["feats"]] = int(self.fused_features.data_ptr())
        self.binding5[self.bindingmap5["reg"]] = int(self.reg.data_ptr())
        self.binding5[self.bindingmap5["height"]] = int(self.height.data_ptr())
        self.binding5[self.bindingmap5["dim"]] = int(self.dim.data_ptr())
        self.binding5[self.bindingmap5["rot"]] = int(self.rot.data_ptr())
        self.binding5[self.bindingmap5["vel"]] = int(self.vel.data_ptr())
        self.binding5[self.bindingmap5["heatmap"]] = int(self.heatmap.data_ptr())
        self.binding5[self.bindingmap5["score"]] = int(self.score.data_ptr())
        self.binding5[self.bindingmap5["query_label"]] = int(self.query_label.data_ptr())

    def get_avg_latencies(self):
        """Return a dict of average latencies in milliseconds."""
        return {
            name: (info['sum']/info['count']*1000.0 if info['count']>0 else 0.0)
            for name, info in self._timing.items()
        }

    def forward(self, imgs, points, metas):

        t0 = time.perf_counter()
        self.binding1[self.bindingmap1["imgs"]] = int(imgs.data_ptr())
        self.context1.execute_v2(self.binding1)
        elapsed = time.perf_counter() - t0
        self._timing['img_backbone']['sum']   += elapsed
        self._timing['img_backbone']['count'] += 1

        ###################################

        depth = utils.calculate_depth(imgs, points, metas["lidar2img"], metas["img_aug_matrix"], metas["lidar_aug_matrix"])

        t0 = time.perf_counter()
        self.binding2[self.bindingmap2["depth"]] = int(depth.data_ptr())
        self.context2.execute_v2(self.binding2)
        elapsed = time.perf_counter() - t0
        self._timing['vtransform_feat']['sum']   += elapsed
        self._timing['vtransform_feat']['count'] += 1

        ###################################

        geom = utils.get_geometry(metas["cam2img"], metas["cam2lidar"], metas["img_aug_matrix"], metas["lidar_aug_matrix"])

        t0 = time.perf_counter()
        img_bev_features = cuda_utils.bev_pool(geom, self.bevpool_features[:len(points)], self.bevpool_layer)
        elapsed = time.perf_counter() - t0
        self._timing['vtransform_bevpool']['sum']   += elapsed
        self._timing['vtransform_bevpool']['count'] += 1

        ###################################

        t0 = time.perf_counter()
        self.binding3[self.bindingmap3["feats"]] = int(img_bev_features.data_ptr())
        self.context3.execute_v2(self.binding3)
        elapsed = time.perf_counter() - t0
        self._timing['vtransform_down']['sum']   += elapsed
        self._timing['vtransform_down']['count'] += 1

        ###################################

        t0 = time.perf_counter()
        pts_features, pts_coords, _ = cuda_utils.voxelize(points, self.voxel_layer)
        elapsed = time.perf_counter() - t0
        self._timing['voxelization']['sum']   += elapsed
        self._timing['voxelization']['count'] += 1

        ###################################

        t0 = time.perf_counter()
        input_sp_tensor = spconv.SparseConvTensor(pts_features, pts_coords, [1440, 1440, 41], len(points))
        inputs = {self.lidar_backbone_parser.inputs[0]: input_sp_tensor}
        outputs = self.lidar_backbone_model.run(inputs)
        pts_bev_features = outputs[self.lidar_backbone_parser.outputs[0]]
        elapsed = time.perf_counter() - t0
        self._timing['lidar_backbone']['sum']   += elapsed
        self._timing['lidar_backbone']['count'] += 1

        ###################################

        t0 = time.perf_counter()
        self.binding4[self.bindingmap4["pts_bev"]] = int(pts_bev_features.data_ptr())
        self.context4.execute_v2(self.binding4)
        elapsed = time.perf_counter() - t0
        self._timing['fusion']['sum']   += elapsed
        self._timing['fusion']['count'] += 1

        ###################################

        t0 = time.perf_counter()
        self.context5.execute_v2(self.binding5)
        elapsed = time.perf_counter() - t0
        self._timing['head']['sum']   += elapsed
        self._timing['head']['count'] += 1

        ###################################

        bboxes_3ds = list()
        scores_3ds = list()
        labels_3ds = list()

        for B in range(len(points)):
            bboxes_3d, scores_3d, labels_3d = utils.decode(
                self.heatmap[B],
                self.score[B],
                self.rot[B],
                self.dim[B],
                self.reg[B],
                self.height[B],
                self.vel[B],
                self.query_label[B],
            )
            bboxes_3ds.append(bboxes_3d)
            scores_3ds.append(scores_3d)
            labels_3ds.append(labels_3d)

        return bboxes_3ds, scores_3ds, labels_3ds
