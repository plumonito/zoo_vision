uint32 MAX_DETECTION_COUNT=15           # Fix capacity to enable shared-memory zero-copy between processes

zoo_msgs/Header header                  # Header timestamp should be acquisition time of image

uint32 detection_count

uint32[15] track_ids                    # size=MAX_DETECTION_COUNT
uint32[15] identity_ids                 # size=MAX_DETECTION_COUNT
zoo_msgs/BoundingBox2D[15] bboxes       # size=MAX_DETECTION_COUNT
float32[45] world_positions             # size=MAX_DETECTION_COUNT*3
zoo_msgs/Tensor3b32m masks

uint64 processing_time_ns