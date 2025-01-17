uint32 MAX_DETECTION_COUNT=20           # Fix capacity to enable shared-memory zero-copy between processes

zoo_msgs/Header header                  # Header timestamp should be acquisition time of image

uint32 detection_count

zoo_msgs/BoundingBox2D[20] bboxes       # size=MAX_DETECTION_COUNT
float32[60] world_positions             # size=MAX_DETECTION_COUNT*3
zoo_msgs/Tensor3b32m masks

