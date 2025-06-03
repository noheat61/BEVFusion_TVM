import onnx
import torch
from onnx import numpy_helper, helper
import spconv.pytorch as spconv
import torch.nn.functional as F

class ONNXParser:
    def __init__(self, onnx_path: str):
        # Load ONNX model
        self.model = onnx.load(onnx_path)
        self.graph = self.model.graph
        
        # 1) Initializers → torch.Tensor
        self.initializer = {
            init.name: torch.tensor(numpy_helper.to_array(init))
            for init in self.graph.initializer
        }
        
        # 2) Input / Output names
        self.inputs  = [inp.name for inp in self.graph.input]
        self.outputs = [out.name for out in self.graph.output]
        
        # 3) Node list
        self.nodes   = list(self.graph.node)


class ONNXEngine:
    def __init__(self, parser: ONNXParser, custom_ops=None):
        self.parser = parser
        
        self.op_map = {
            "Add":               self._run_add,
            "Relu":              self._run_relu,
            "Permute":           self._run_transpose,
            "View":              self._run_view,
            "Contiguous":        self._run_contiguous,
            "ToDense":           self._run_to_dense,
            "BatchNormalization":self._run_batchnorm,
            "SubMConv3d":        self._run_submconv,
            "SparseConv3d":      self._run_sparseconv,
        }
        if custom_ops:
            self.op_map.update(custom_ops)

    def _run_add(self, a, b, **kw):
        return a + b

    def _run_relu(self, x, **kw):
        if isinstance(x, spconv.SparseConvTensor):
            return x.replace_feature(torch.relu(x.features))
        return torch.relu(x)

    def _run_transpose(self, x, perm, **kw):
        return x.permute(tuple(perm))

    def _run_view(self, x, shape, **kw):
        return x.view(tuple(shape.to(torch.int64).tolist()))

    def _run_contiguous(self, x, **kw):
        return x.contiguous()

    def _run_to_dense(self, x, **kw):
        return x.dense()

    def _run_batchnorm(self, x, scale, bias, mean, var, epsilon, momentum, **kw):
        if isinstance(x, spconv.SparseConvTensor):
            return x.replace_feature(F.batch_norm(x.features, running_mean=mean, running_var=var, weight=scale, bias=bias, training=False, momentum=momentum, eps=epsilon))
        return F.batch_norm(x, running_mean=mean, running_var=var, weight=scale, bias=bias, training=False, momentum=momentum, eps=epsilon)

    def _run_submconv(self, features, weight, bias=None, **kw):
        conv = spconv.SubMConv3d(
            in_channels=weight.shape[4],
            out_channels=weight.shape[0],
            kernel_size=tuple(kw["kernel_shape"]),
            stride=tuple(kw["strides"]),
            padding=tuple(kw["pads"]),
            dilation=tuple(kw["dilations"]),
            groups=kw["group"],
            bias=bias is not None).to("cuda")
        conv.weight = torch.nn.Parameter(weight)
        if bias is not None:
            conv.bias = torch.nn.Parameter(bias)
        return conv(features)

    def _run_sparseconv(self, features, weight, bias=None, **kw):
        conv = spconv.SparseConv3d(
            in_channels=weight.shape[4],
            out_channels=weight.shape[0],
            kernel_size=tuple(kw["kernel_shape"]),
            stride=tuple(kw["strides"]),
            padding=tuple(kw["pads"]),
            dilation=tuple(kw["dilations"]),
            groups=kw["group"],
            bias=bias is not None)
        conv.weight = torch.nn.Parameter(weight)
        if bias is not None:
            conv.bias = torch.nn.Parameter(bias)
        return conv(features)


    def run(self, input_data: dict):
        # 1) 메모리 초기화 (name → torch.Tensor)
        mem = {}
        for k, v in self.parser.initializer.items():
            mem[k] = v.to("cuda")
        mem.update(input_data)               # user inputs
        
        # 2) 노드 순차 실행
        for i, node in enumerate(self.parser.nodes):
            op_type = node.op_type
            func    = self.op_map.get(op_type)
            if func is None:
                raise RuntimeError(f"Unsupported op: {op_type}")
            
            # 입력 텐서들
            args = [mem[name] for name in node.input]
            
            # 속성 추출
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            
            # 실행
            result = func(*args, **attrs)
            
            # 결과 저장
            if isinstance(result, torch.Tensor) or isinstance(result, spconv.SparseConvTensor):
                mem[node.output[0]] = result
            else:
                # 여러 출력일 경우
                for out_name, out_tensor in zip(node.output, result):
                    mem[out_name] = out_tensor
        
        # 3) 최종 결과 반환
        return {name: mem[name] for name in self.parser.outputs}