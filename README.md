# Strip Attention for Image Restoration

Yuning Cui, Yi Tao, Luoxi Jing, Alois Knoll


 ## Abstract
As a long-standing task, image restoration aims to recover the latent sharp image from its degraded counterpart. In recent years, owing to the strong ability of self-attention in capturing longrange dependencies, Transformer based methods have achieved promising performance on multifarious image restoration tasks. However, the canonical self-attention leads to quadratic complexity with respect to input size, hindering its further applications in image restoration. In this paper, we propose a Strip Attention Network (SANet) for image restoration to integrate information in a more efficient and effective manner. Specifically, a strip attention unit is proposed to harvest the contextual information for each pixel from its adjacent pixels in the same row or column. By employing this operation in different directions, each location can perceive information from an expanded region. Furthermore, we apply various receptive fields in different feature groups to enhance representation learning. Incorporating these designs into a U-shaped backbone, our SANet performs favorably against state-of-the-art algorithms on several image restoration tasks. 

## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~


## Results
The resulting images can be downloaded [here](https://drive.google.com/drive/folders/1bz0ZQ1EWlhqRMVgXTYvb8TbYmVzf93zM?usp=sharing).



## Citation
If you find this project useful for your research, please consider citing:
~~~
@inproceedings{cui2023strip,
  title={Strip Attention for Image Restoration},
  author={Cui, Yuning and Tao, Yi and Jing, Luoxi and Knoll, Alois},
  booktitle={International Joint Conference on Artificial Intelligence, IJCAI},
  year={2023}
}
~~~
## Contact
Should you have any question, please contact Yuning Cui.
