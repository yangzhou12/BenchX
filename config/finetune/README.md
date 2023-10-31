## Loading Vision and Language Checkpoints

Before testing the model, load vision and language encoder weights separately:
1. `pretrained`: Path to local pretrained weights. Set to `"DEFAULT"` if using default backbone pre-trained weights, else train from scratch if `None`.
2. `prefix`: Prefix of weights of visual/language encoder.


## Setting Dataset