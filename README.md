
## Table of Contents

- Install Python Libraries
- Download Dataset
- Run Experiments


## Install Python Libraries
In order to install requirements run:
```
pip install -r install requirements.txt
```

## Download Dataset
In order to download [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) make a folder ``data`` and then use the ``gdown`` library:
```
mkdir data/
cd data/
gdown https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45
tar -xvzf CUB_200_2011.tgz  
```
Unzip the downloaded file inside the folder and you're ready!

## Run Experiments
Run experiment on contrastive loss using Resnet50 on CUB  
- To run the clean experiment run:
```
cd code/
python baseline.py
```