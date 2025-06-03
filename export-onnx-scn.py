import torch
import torch.nn as nn
import numpy as np

import onnx
import onnx.helper as helper
import spconv.pytorch as spconv

avoid_reuse_container = []
obj_to_tensor_id = {}
nodes = []
initializers = []
enable_trace = False

def register_node(fn):

    fnnames   = fn.split(".")
    fn_module = eval(".".join(fnnames[:-1]))
    fn_name   = fnnames[-1]
    oldfn = getattr(fn_module, fn_name)
    
    def make_hook(bind_fn):

        ilayer = 0
        def internal_forward(self, *args):
            global enable_trace

            if not enable_trace:
                return oldfn(self, *args)

            global avoid_reuse_container
            nonlocal ilayer

            # Use the enable_trace flag to avoid internal trace calls
            enable_trace = False
            y = oldfn(self, *args)
            bind_fn(self, ilayer, y, *args)
            enable_trace = True

            avoid_reuse_container.extend(list(args) + [y]) 
            ilayer += 1
            return y

        setattr(fn_module, fn_name, internal_forward)
    return make_hook

###################################

@register_node("torch.nn.BatchNorm1d.forward")
def symbolic_bn(self, ilayer, y, x):
    register_tensor(y)
    # print(f"   --> BatchNorm1d{ilayer} -> Input {get_tensor_id(x)}, Output {get_tensor_id(y)}")
    
    inputs = [get_tensor_id(x), 
              append_initializer(self.weight, f"bn{ilayer}.weight"), 
              append_initializer(self.bias, f"bn{ilayer}.bias"), 
              append_initializer(self.running_mean, f"bn{ilayer}.running_mean"), 
              append_initializer(self.running_var, f"bn{ilayer}.running_var")]
    
    nodes.append(
        helper.make_node(
            "BatchNormalization", inputs, [get_tensor_id(y)], f"bn{ilayer}",
            epsilon = self.eps, momentum = self.momentum
        )
    )

@register_node("torch.nn.ReLU.forward")
def symbolic_relu(self, ilayer, y, x):
    register_tensor(y)
    # print(f"   --> ReLU{ilayer} -> Input {get_tensor_id(x)}, Output {get_tensor_id(y)}")
    nodes.append(
        helper.make_node(
            "Relu", [get_tensor_id(x)], [get_tensor_id(y)], f"relu{ilayer}"
        )
    )

@register_node("torch.Tensor.__add__")
def symbolic_add(a, ilayer, y, b):
    register_tensor(y)
    # print(f"   --> Add{ilayer} -> Input {get_tensor_id(a)} + {get_tensor_id(b)}, Output {get_tensor_id(y)}")
    nodes.append(
        helper.make_node(
            "Add", [get_tensor_id(a), get_tensor_id(b)], [get_tensor_id(y)], f"add{ilayer}" 
        )
    )

@register_node("torch.Tensor.permute")
def node_permute(self, ilayer, y, *dims):
    register_tensor(y)
    # print(f"   --> Permute{ilayer}[{dims}][{list(y.shape)}] -> Input {get_tensor_id(self)}, Output {get_tensor_id(y)}")
    nodes.append(
        helper.make_node(
            "Permute", [get_tensor_id(self)], [get_tensor_id(y)], f"permute{ilayer}", perm=dims
        )
    )

@register_node("torch.Tensor.contiguous")
def symbolic_contiguous(self, ilayer, y):
    register_tensor(y)
    # print(f"   --> Contiguous{ilayer} -> Input {get_tensor_id(self)}, Output {get_tensor_id(y)}")
    nodes.append(
        helper.make_node(
            "Contiguous", [get_tensor_id(self)], [get_tensor_id(y)], f"contiguous{ilayer}"
        )
    )

@register_node("torch.Tensor.view")
def symbolic_view(self, ilayer, y, *shape):
    register_tensor(y)
    # print(f"   --> View{ilayer} -> Input {get_tensor_id(self)}, Output {get_tensor_id(y)}")
    
    inputs = [get_tensor_id(self), 
              append_initializer(torch.tensor([-1] + list(shape[1:]), dtype=torch.int64), f"view{ilayer}.shape")]
    
    nodes.append(
        helper.make_node(
            "View", inputs, [get_tensor_id(y)], f"view{ilayer}"
        )
    )

@register_node("spconv.SubMConv3d.forward")
def symbolic_submconv(self, ilayer, y, x):
    register_tensor(y)
    # print(f"   --> SubMConv3d{ilayer} -> Input {get_tensor_id(x)}, Output {get_tensor_id(y)}")

    inputs = [get_tensor_id(x),
              append_initializer(self.weight, f"submconv{ilayer}.weight")]
    if self.bias is not None:
        inputs.append(append_initializer(self.bias, f"submconv{ilayer}.bias"))

    nodes.append(
        helper.make_node(
            "SubMConv3d", inputs, [get_tensor_id(y)], f"submconv{ilayer}",
            kernel_shape=self.kernel_size, group=self.groups, pads=self.padding,
            dilations=self.dilation, strides=self.stride,   
        )
    )

@register_node("spconv.SparseConv3d.forward")
def symbolic_sparseconv(self, ilayer, y, x):
    register_tensor(y)
    # print(f"   --> SparseConv3d{ilayer} -> Input {get_tensor_id(x)}, Output {get_tensor_id(y)}")
    
    inputs = [get_tensor_id(x), 
              append_initializer(self.weight, f"sparseconv{ilayer}.weight")]
    if self.bias is not None:
        inputs.append(append_initializer(self.bias, f"sparseconv{ilayer}.bias"))

    nodes.append(
        helper.make_node(
            "SparseConv3d", inputs, [get_tensor_id(y)], f"sparseconv{ilayer}",
            kernel_shape=self.kernel_size, group=self.groups, pads=self.padding,
            dilations=self.dilation, strides=self.stride,
        )
    )

@register_node("spconv.SparseConvTensor.dense")
def symbolic_dense(self, ilayer, y):
    register_tensor(y)
    # print(f"   --> ToDense{ilayer}[{self.spatial_shape}][{list(y.size())}] -> Input {get_tensor_id(self)}, Output {get_tensor_id(y)}")
    nodes.append(
        helper.make_node(
            "ToDense", [get_tensor_id(self)], [get_tensor_id(y)], f"dense{ilayer}",
        )
    )

###################################

def __obj_to_id(obj):
    idd = id(obj)
    if isinstance(obj, spconv.SparseConvTensor):
        idd = id(obj.features)
    return idd

def register_tensor(obj):
    global obj_to_tensor_id
    obj_to_tensor_id[__obj_to_id(obj)] = str(len(obj_to_tensor_id))

def get_tensor_id(obj):
    idd = __obj_to_id(obj)
    assert idd in obj_to_tensor_id, "ops!!!😮 Cannot find the tensorid of this object. this means that some operators are not being traced. You need to confirm it."
    return obj_to_tensor_id[idd]

def append_initializer(value, name):
    initializers.append(
        helper.make_tensor(
            name=name,
            data_type=helper.TensorProto.DataType.FLOAT,
            dims=list(value.shape),
            vals=value.cpu().data.numpy().astype(np.float32).tobytes(),
            raw=True
        )
    )
    return name

def make_model_forward_hook(self):
    def impl(input_sp_tensor, **kwargs):
        # coors = coors.int()
        # input_sp_tensor = spconv.SparseConvTensor(
        #     voxel_features, coors, self.sparse_shape, batch_size
        # )

        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features
    return impl

###################################

def export_onnx(model : nn.Module, voxels, coors, batch_size, save_onnx):

    global avoid_reuse_container, obj_to_tensor_id, nodes, initializers, enable_trace
    avoid_reuse_container = []
    obj_to_tensor_id = {}
    nodes = []
    initializers = []

    # inplace = True일 경우 tracing이 안됨
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    model.forward = make_model_forward_hook(model)

    # print("Tracing model inference...")
    # print("> Do inference...")
    with torch.no_grad():
        input_sp_tensor = spconv.SparseConvTensor(
            voxels, coors, model.sparse_shape, batch_size
        )
        register_tensor(input_sp_tensor)
        enable_trace = True
        y = model(input_sp_tensor)
        enable_trace = False

    # print("Tracing done!")

    inputs = [
        helper.make_value_info(
            name=get_tensor_id(voxels),
            type_proto=helper.make_tensor_type_proto(
                elem_type=helper.TensorProto.DataType.FLOAT,
                shape=["num_points", voxels.size(1)]
            )
        )
    ]

    outputs = [
        helper.make_value_info(
            name=get_tensor_id(y),
            type_proto=helper.make_tensor_type_proto(
                elem_type=helper.TensorProto.DataType.FLOAT,
                shape=["batch_size"] + list(y.size())[1:]
            )
        )
    ]

    graph = helper.make_graph(
        name="lidar_backbone",
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializer=initializers
    )

    opset = [
        helper.make_operatorsetid("ai.onnx", 13)
    ]

    model = helper.make_model(graph, opset_imports=opset, producer_name="pytorch", producer_version="2.1.0")
    onnx.save_model(model, save_onnx)
    print(f"✅ {save_onnx} saved")

###################################

from torch.hub import load_state_dict_from_url

from mmdetection3d.projects.BEVFusion.bevfusion.sparse_encoder import BEVFusionSparseEncoder

device = "cuda"
model = BEVFusionSparseEncoder(in_channels=5,
        sparse_shape=[1440, 1440, 41],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock').eval().to(device)

ckpt = load_state_dict_from_url("https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth")

full_sd = ckpt['state_dict']
sparse_sd = {k.replace("pts_middle_encoder.", ""): v for k, v in full_sd.items() if k.startswith("pts_middle_encoder.")}
sparse_sd_fixed = {k: (v.permute(1, 2, 3, 4, 0).contiguous() if v.ndim == 5 else v) for k, v in sparse_sd.items()}
model.load_state_dict(sparse_sd_fixed)
# torch.save(model, "sparse_encoder.pth")

voxels = torch.zeros(1, 5).to(device)
coors  = torch.zeros(1, 4).int().to(device)

output = model(voxels, coors, 1)
export_onnx(model, voxels, coors, 1, "onnx/lidar_backbone.onnx")