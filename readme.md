# AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender

<div align="center">

[![EMNLP 2025](https://img.shields.io/badge/Venue-EMNLP--25-278ea5)](https://2025.emnlp.org/)
[![Status](https://img.shields.io/badge/Status-Accepted-success)](https://arxiv.org/abs/2504.09466)
[![Python](https://img.shields.io/badge/Python-3.9.0-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)

**The official implementation for the EMNLP 2025 paper**  
*"AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender"*

[📄 Paper](https://arxiv.org/abs/2504.09466) • [🚀 Code](https://github.com/MuyuenLP/AdaSteer)

</div>

---


## 🔧 Requirements

* Python 3.9.0
* PyTorch 2.4.1
* Transformers 4.46.3

### Installation

```bash
git clone https://github.com/MuyuenLP/AdaSteer
cd AdaSteer
pip install -e .
```

## 🚀 Quick Start

### 1. Vector Extraction

We employ a difference-in-means method for vector extraction. Here's an example of extracting the rejection vector for LLaMA-3.1-8B-Instruct:

```bash
bash scripts/llama31/RD.sh
```

### 2. Activation Steering

#### Fixed Steering Coefficient

Apply rejection vectors with a fixed steering coefficient λ to reject harmful inputs:

```bash
bash scripts/llama31/refusal.sh
```

#### Adaptive Steering (AdaSteer)

For dynamic steering, we modify the transformers library to implement adaptive activation steering coefficient calculation. See `./adasteer/models/For_Steering_LlamaModel_adasteer.py` for implementation details.

```bash
bash scripts/llama31/adasteer.sh
```


## 📚 Citation

If you find our work useful for your research, please kindly cite our paper:

```bibtex
@article{zhao2025adasteer,
  title={AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender},
  author={Zhao, Weixiang and Guo, Jiahe and Hu, Yulin and Deng, Yang and Zhang, An and Sui, Xingyu and Han, Xinyang and Zhao, Yanyan and Qin, Bing and Chua, Tat-Seng and others},
  journal={arXiv preprint arXiv:2504.09466},
  year={2025}
}
```

