import onnx
from onnxsim import simplify

#################################

model = onnx.load("onnx/img_backbone.onnx")

model_simplified, check = simplify(
    model,
    overwrite_input_shapes={
        "imgs": [1, 6, 3, 256, 704],
    }
)

onnx.save(model_simplified, "onnx/img_backbone_simp.onnx")

#################################

model = onnx.load("onnx/vtransform_feature.onnx")

model_simplified, check = simplify(
    model,
    overwrite_input_shapes={
        "feats": [1, 6, 256, 32, 88],
        "depth": [1, 6, 1, 256, 704]
    }
)

onnx.save(model_simplified, "onnx/vtransform_feature_simp.onnx")

# #################################

model = onnx.load("onnx/vtransform_downsample.onnx")

model_simplified, check = simplify(
    model,
    overwrite_input_shapes={
        "feats": [1, 80, 360, 360],
    }
)

onnx.save(model_simplified, "onnx/vtransform_downsample_simp.onnx")

# #################################

model = onnx.load("onnx/fusion.onnx")

model_simplified, check = simplify(
    model,
    overwrite_input_shapes={
        "img_bev": [1, 80, 180, 180],
        "pts_bev": [1, 256, 180, 180],
    }
)

onnx.save(model_simplified, "onnx/fusion_simp.onnx")

# #################################

model = onnx.load("onnx/head.onnx")

model_simplified, check = simplify(
    model,
    overwrite_input_shapes={
        "feats": [1, 512, 180, 180],
    }
)

import numpy as np
from onnx import numpy_helper

graph = model_simplified.graph

for init in graph.initializer:
    arr = numpy_helper.to_array(init)
    if arr.dtype == np.int64 and arr.size == 1:
        new_arr = np.array(arr, dtype=np.int32)
        new_init = numpy_helper.from_array(new_arr, name=init.name)
        graph.initializer.remove(init)
        graph.initializer.append(new_init)

onnx.save(model_simplified, "onnx/head_simp.onnx")
