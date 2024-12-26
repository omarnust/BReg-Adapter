<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> ProKeR: A Kernel Perspective on Few-Shot Adaptation of Large Vision-Language Models </h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://zhengbo.wang/" target="_blank" style="text-decoration: none;">Yassir Bendou<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://liangjian.xyz/" target="_blank" style="text-decoration: none;">Amine Ouasfi<sup>2</sup></a>&nbsp;,&nbsp;
    <a href="https://tomsheng21.github.io/" target="_blank" style="text-decoration: none;">Vincent Gripon<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://sites.google.com/site/pinyuchenpage" target="_blank" style="text-decoration: none;">Adnane Boukhayma<sup>2</sup></a>&nbsp;,&nbsp;
	<br>
<sup>1</sup>IMT Atlantique&nbsp;&nbsp;&nbsp;
<sup>2</sup>INRIA&nbsp;&nbsp;&nbsp;
</p>

## Requirements
### Installation
Create a conda environment and install dependencies:
```
conda create -n h2b python=3.9
conda activate h2b

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

### Dataset
Follow [DATASET.md](DATASET.md) to install ImageNet and other datasets referring to [CoOp](https://github.com/KaiyangZhou/CoOp).

## Get Started
### Configs
The running configurations can be modified in `configs`. 

### Running
For few-shot classification:
```bash
   python main.py --method ProKeR --shots 1 2 4 8 16 --dataset caltech101 --augment-epoch 10
```

If GPU memory is saturated, consider using fewer data augmentations --augment-epoch

## Running Options
Multiple methods are implemented:
- ZeroShot (Zero-shot classifier) 
- TIP (Tip-Adapter)
- GDA 
- CLAP 
- ProKeR (ours)
- ProKeR_CLAP_joint (ours ProKeR + CLAP)

## Acknowledgement

This repo benefits from [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter), [CoOp,](https://github.com/KaiyangZhou/Dassl.pytorch) and [GDA](https://github.com/mrflogs/ICLR24). 

## Citation
```latex
```

## Contact

If you have any question, feel free to contact yassir.bendou@gmail.com.