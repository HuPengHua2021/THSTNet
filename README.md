# A Two-Stage Hierarchical Spatiotemporal Fusion Network for Land Surface Temperature With Transformer

## Environment:
pip install -r requirements.txt

##Sample preparation
python tiff_to_numpy.py
python sample_cut.py

# Train
python train_stage_one.py
python train_stage_two.py

# Test
true_image_test.py

## Models
/stage_one
/stage_two
/base_models/swin_transformer.py

##Cite
@ARTICLE{10930886,
  author={Hu, Penghua and Pan, Xin and Yang, Yingbao and Dai, Yang and Chen, Yuncheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Two-Stage Hierarchical Spatiotemporal Fusion Network for Land Surface Temperature With Transformer}, 
  year={2025},
  volume={63},
  number={},
  pages={1-20},
  doi={10.1109/TGRS.2025.3552577}}


