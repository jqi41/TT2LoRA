## TT2LoRA: Enhancing Low-Rank Adaptation via Joint Tensor-Train Parameterization with Neural Tangent Kernel Guarantees
The implementation of Tensor-Train guided Low-Rank Adaptation 

Our codes include the experiments of TT2LoRA (tt2lora_qd.py), LoRA (lora_qd.py), and TT-LoRA (tt-lora_qd.py)

The simulations of  can be referred to our previous repo: https://github.com/jqi41/TTN-VQC and https://github.com/jqi41/Pretrained-TTN-VQC

### Installation 

The main dependencies include *pytorch* 

 ### 0. Downloading the dataset
```
git clone https://gitlab.com/QMAI/mlqe_2023_edx.git
```

### 1. Simulating experiments of quantum dot classification 

#### 1.1 Assessing TT2LoRA
python3 tt2lora_qd.py 

#### 1.2 Assessing LoRA
python lora_qd.py

### 1.3 Experimental TT-LoRA
python tt-lora_qd.py
