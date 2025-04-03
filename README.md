# VTF-AVIT
### Our paper has now been accepted by the BMSE journal.

## Title: 
Efficient Visual-Tactile Transformer with Token Reorganization for Robotic Slip Detection

## Abstract:
Transformer architectures have gained prominence in robotic slip detection owing to their capacity in capturing spatiotemporal correlations within multimodal sensory streams. Nevertheless, the conventional full-token self-attention mechanism inherent in standard Transformers introduces certain limitations, including quadratic computational complexity, data-hungry training requirements. To address these limitations, we propose a token reorganization scheme, leveraging class token attention saliency in the deep layers of the neural network to dynamically reorganize tokens. This approach selectively retains task-critical information in the visual-tactile modalities while efficiently compressing background tokens through adaptive averaging, thereby optimizing computational efficiency without compromising performance. This scheme reduces the computational complexity of salient self-attention. Based on this scheme, a visual-tactile fusion Transformer for robot slip detection is developed. Extensive experiments have shown that the proposed Transformer can achieve state-of-the-art performance, which will reduce the computational cost while increasing the detection accuracy to 86.75\% by enhancing the focus on deformation-sensitive features. This work provides an effective visual tactile perception solution for robust robot manipulation under real-world uncertainties. Our code and dataset are available at \url{https://github.com/Bugs-Bunny01/VTF-AVIT}.
![image](https://github.com/Bugs-Bunny01/VTF-SLIP-TranSFormer/blob/main/V-T-fusion.png)
![image](https://github.com/Bugs-Bunny01/VTF-SLIP-TranSFormer/blob/main/token-visual.png)

## Requirements:
* python3 >= 3.8.10                 
* numpy >= 1.21.1                
* pytorch  >= 1.9.0+cpu (cpu training&testing) or 1.9.0+cu102 (cuda training&testing)
* opencv >= 4.5.3
* yaml >= 5.4.1
* json >= 2.0.9
* matplotlib >= 3.4.2

## Train & Test
1. We used a dataset from this paper: https://arxiv.org/abs/1802.10153, please download the slip detection data yourself.
2. python main.py
3. python test.py
