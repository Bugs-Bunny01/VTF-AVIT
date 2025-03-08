our paper: VTF-AVIT: Enhanced Slip Detection via Token-Reorganized Multimodal Fusion

Abstract:
Accurate slip detection remains a critical challenge for dexterous robotic manipulation. This paper presents VTF-AVIT, a novel vision-tactile fusion framework that synergizes adaptive token pruning with Transformer architectures for real-time slip detection. Our core innovation lies in two aspects: (1) A dynamic token reorganization mechanism that automatically preserves task-critical features (e.g., shear deformation patterns in GelSight tactile images) while pruning redundant background tokens, reducing computational complexity from {O}(N^2) to {O}(KN) where K denotes the adaptive preservation ratio (optimally 0.6); (2) Hierarchical cross-modal attention layers that effectively fuse visual (RGB-D) and tactile (pressure/geometry) modalities through learned attention weights. Extensive experiments on a multimodal grasping dataset containing 12,800 samples demonstrate superior performance: achieves 86.75% detection accuracy with 12.83\,ms inference time per frame, reduces FLOPs by 41% compared to vanilla Vision Transformers (ViT), and maintains 89.3% slip compensation success rate on UR5 robotic platform. The proposed method shows 3.1% accuracy improvement over conventional Transformers and 5.7% over CNN-LSTM baselines. 
![image](https://github.com/Bugs-Bunny01/VTF-SLIP-TranSFormer/blob/main/V-T-fusion.png)
![image](https://github.com/Bugs-Bunny01/VTF-SLIP-TranSFormer/blob/main/token-visual.png)

Requirements:
* python3 >= 3.8.10                 
* numpy >= 1.21.1                
* pytorch  >= 1.9.0+cpu (cpu training&testing) or 1.9.0+cu102 (cuda training&testing)
* opencv >= 4.5.3
* yaml >= 5.4.1
* json >= 2.0.9
* matplotlib >= 3.4.2

Train & Test
1. We used a dataset from this paper: https://arxiv.org/abs/1802.10153, please download the slip detection data yourself.
2. python main.py
3. python test.py
