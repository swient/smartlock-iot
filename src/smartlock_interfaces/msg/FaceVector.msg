# Face feature vector from vision subsystem
# 128-dimensional vector from face recognition model

builtin_interfaces/Time timestamp       # Detection timestamp
float32[] vector                        # 128-dimensional feature vector
float32 confidence                      # Detection confidence (0.0 - 1.0)
