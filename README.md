# Limited Data Rolling Bearing Fault Diagnosis with Few-shot Learning

 For the paper "Limited Data Rolling Bearing Fault Diagnosis with Few-shot Learning", as the mekhub.cn website needs to shut down for some reason, the origin code link (https://mekhub.cn/as/fault_diagnosis_with_few-shot_learning/) can't be open anymore. This is the alternate code repository of the paper. 
# [[Paper]](https://ieeexplore.ieee.org/abstract/document/8793060)

## To cite
```
@article{zhang2019limited,
  title={Limited Data Rolling Bearing Fault Diagnosis With Few-Shot Learning},
  author={Zhang, Ansi and Li, Shaobo and Cui, Yuxin and Yang, Wanli and Dong, Rongzhi and Hu, Jianjun},
  journal={IEEE Access},
  volume={7},
  pages={110895--110904},
  year={2019},
  publisher={IEEE}
}
```

## Structure

- Datasets: The Case Western Reserve University (CWRU) bearing fault diagnosis dataset.
- figures: Some figures useing in README.md.
- cwru.py: Define dataset load function which can auto download the data. But dataset link is not working now. You can download manually from [[Datasets.zip (229MB)]](https://1drv.ms/u/s!At5AiOeueyrEgyUY038Ln_SQ8SRo?e=Fp9o7P) or [提取码: 7uu9](https://pan.baidu.com/s/1WgJMPSDcipugR1Bh4KRadg).
- experimentAB.ipynb: Experiment A and B code in the paper
- experimentC.ipynb: Experiment C code in the paper 
- experimentD.ipynb: Experiment D code in the paper 
- metadata.txt: Be used in the cwru.py file.
- models.py: Define few-shot model and WDCNN model load functions.
- siamese.py: Define the init of few-shot input data, few-shot model training, and few-shot model testing functions.
- utils.py: Define some utility functions.
- tmp: Save trained models and test results. Data is not required to download for running the code，and my paper tmp files can download from [[tmp.zip (464MB)]](https://1drv.ms/u/s!At5AiOeueyrEgydZVNeJjbWHiw68?e=HmzK5f) or [提取码: htgw](https://pan.baidu.com/s/1k9xkejB-3YRqDunKA9AUsw).

## Usage
Launching Jupyter Notebook App.
```
jupyter notebook .
```
Then open experimentA-SVM.ipynb, experimentAB.ipynb, experimentC.ipynb, and experimentD.ipynb to see the detail of experiments.


## Intrudoction

Paper focuses on bearing fault diagnosis with limited training data. Recently deep learning based fault diagnosis methods have achieved promising results. However, most of these intelligent fault diagnosis methods have not addressed one of the major challenges **in real-world bearing fault diagnosis**: there are always one or more fault types with limited fault samples covering all different working conditions. This situation can arise in several situations: 

1. industry systems are not allowed to run into faulty states due to the consequences, especially for critical systems and failures; 
2. most electro-mechanical failures occur slowly and follow a degradation path such that failure degradation of a system might take months or even years, which makes it difficult to collect related datasets; 
3. working conditions of mechanical systems are very complicated and frequently change from time to time according to production requirements. It is unrealistic to collect and label enough training samples; 
4. especially in real-world applications, fault categories and working conditions are usually unbalanced. It is thus difficult to collect enough samples for every fault type under different working conditions. 
 
**How to develop effective algorithms that can deal with scarcity of samples for a portion of fault types thus becomes a critical problem in fault diagnosis community.**

In paper, we propose a deep neural network based few-shot learning approach for rolling bearing fault diagnosis with limited data. Our model is based on the siamese neural network, which learns by exploiting sample pairs of the same or different categories. Experimental results over the standard Case Western Reserve University (CWRU) bearing fault diagnosis benchmark dataset showed that our few-shot learning approach is more effective in fault diagnosis with limited data availability. When tested over different noise environments with minimal amount of training data, the performance of our few-shot learning model surpasses one of the baseline with reasonable noise level. When evaluated over test sets with new fault types or new working conditions, few-shot models work better than the baseline trained with all fault types. 

All our models and datasets in this study are open sourced and can be downloaded here.

Here are some figures in the paper, and the detail introduction of figures please see in paper：

### Few-shot learning general strategy
<img src=figures/few-shot.png width="50%">

### Flowchart of the few-shot learning based fault diagnosis
<img src=figures/framework1.png width="90%">

### Few-shot learning model based on CNNs (WDCNN)
<img src=figures/framework2.png width="50%">


## Env requirement
All codes and trained models were tested with Python3.6 and following versioned packages:

- tensorflow >= 1.12.0
- keras >= 2.2.4
- seaborn
- imblearn
