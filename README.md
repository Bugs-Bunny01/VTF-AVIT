our paper: Efficient Slip Detection in Robotic Grasping Using Accelerated TranSformers and Multimodal Sensor Fusion

Abstract:
We present introduces VTF-AVIT, an innovative TranSformer-based model that integrates tactile and visual sensors to enhance slip detection in robotic grasping.  By adopting a novel token reorganization strategy, VTF-AVIT substantially improves computational efficiency while enhancing accuracy.  The model excels in dynamic and uncertain environments, providing robust and reliable grasping strategies critical for real-time applications in autonomous robotics.  Experimental results demonstrate that VTF-AVIT outperforms traditional TranSformer and CNN-LSTM models in both accuracy and computational efficiency.  Furthermore, the fusion of tactile and visual data allows the system to effectively adapt to varying environmental conditions and object characteristics, thereby advancing the capabilities of robots in complex manipulation tasks. 
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
1. Download slip detection data，我们使用的数据集来自这篇论文https://arxiv.org/abs/1802.10153，请自行下载
2. python main.py
3. python test.py
