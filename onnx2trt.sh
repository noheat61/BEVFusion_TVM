FP16_FLAG=""
if [[ "$1" == "--fp16" ]]; then
  FP16_FLAG="--fp16"
fi

trtexec --onnx=onnx/img_backbone.onnx --saveEngine=onnx/img_backbone.engine $FP16_FLAG && 
trtexec --onnx=onnx/vtransform_feature.onnx --saveEngine=onnx/vtransform_feature.engine $FP16_FLAG && 
trtexec --onnx=onnx/vtransform_downsample.onnx --saveEngine=onnx/vtransform_downsample.engine $FP16_FLAG && 
trtexec --onnx=onnx/fusion.onnx --saveEngine=onnx/fusion.engine $FP16_FLAG && 
trtexec --onnx=onnx/head.onnx --saveEngine=onnx/head.engine $FP16_FLAG
