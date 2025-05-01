

# PINNMamba

The official repo for PINNMamba. We highlight the advantages of using the State-Space Model in solving the physics partial differential equations.

[News]: The PINNMamba paper is accepted by ICML 2025. 

## Get Started

```shell
python reaction_pinnmamba.py --model PINNMamba --device 'cuda:0'
python wave_pinnmamba.py --model PINNMamba --device 'cuda:0'
python convection_pinnmamba.py --model PINNMamba --device 'cuda:0'
```


## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code base:

https://github.com/AdityaLab/pinnsformer

https://github.com/thuml/RoPINN

‘’‘
@article{xu2025sub,
  title={Sub-Sequential Physics-Informed Learning with State Space Model},
  author={Xu, Chenhui and Liu, Dancheng and Hu, Yuting and Li, Jiajie and Qin, Ruiyang and Zheng, Qingxiao and Xiong, Jinjun},
  journal={in International Conference on Machine Learning},
  year={2025}
}
’‘’
