YOLO variant config folder: yolov8n

Model in configs: yolov8n.pt
Stage1 runs dir: ./runs_yolov8n/glcp_stage1_yolo_det
Stage2 runs dir: ./runs_yolov8n/glcp_stage2_yolo_det
Stage1 feature layers: [4, 6, 8]
Stage1 SPPF layer: 9
Stage2 visualization layers: [4, 6, 8, 9, 15, 18, 21]
Pre-head/proxy layers: [15, 18, 21]

Note: If you do not have pretrained weights for yolov8n.pt, replace model.yolo_model with the corresponding local architecture YAML. This will train from architecture initialization and is not strictly comparable to pretrained .pt initialization.
