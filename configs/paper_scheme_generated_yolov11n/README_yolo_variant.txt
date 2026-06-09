YOLO variant config folder: yolov11n

Model in configs: yolo11n.pt
Stage1 runs dir: ./runs_yolov11n/glcp_stage1_yolo_det
Stage2 runs dir: ./runs_yolov11n/glcp_stage2_yolo_det
Stage1 feature layers: [4, 6, 8]
Stage1 SPPF layer: 9
Stage2 visualization layers: [4, 6, 8, 9, 16, 19, 22]
Pre-head/proxy layers: [16, 19, 22]

Note: If you do not have pretrained weights for yolo11n.pt, replace model.yolo_model with the corresponding local architecture YAML. This will train from architecture initialization and is not strictly comparable to pretrained .pt initialization.
