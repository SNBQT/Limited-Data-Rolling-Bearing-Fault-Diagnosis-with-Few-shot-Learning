# Limited Data Rolling Bearing Fault Diagnosis with Few-shot Learning 

This is the corresponding repository of paper Limited Data Rolling Bearing Fault Diagnosis with Few-shot Learning. [[Paper]](https://ieeexplore.ieee.org/abstract/document/8793060)


## [[Paper]](https://ieeexplore.ieee.org/abstract/document/8793060)

## [[Code]](https://mekhub.cn/as/fault_diagnosis_with_few-shot_learning/)

All our models and datasets in this study are open sourced and can be downloaded from https://mekhub.cn/as/fault_diagnosis_with_few-shot_learning/ 

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

## Intrudoction

Paper focuses on bearing fault diagnosis with limited training data. Recently deep learning based fault diagnosis methods have achieved promising results. However, most of these intelligent fault diagnosis methods have not addressed one of the major challenges **in real-world bearing fault diagnosis**: there are always one or more fault types with limited fault samples covering all different working conditions. This situation can arise in several situations: 

1. industry systems are not allowed to run into faulty states due to the consequences, especially for critical systems and failures; 
2. most electro-mechanical failures occur slowly and follow a degradation path such that failure degradation of a system might take months or even years, which makes it difficult to collect related datasets; 
3. working conditions of mechanical systems are very complicated and frequently change from time to time according to production requirements. It is unrealistic to collect and label enough training samples; 
4. especially in real-world applications, fault categories and working conditions are usually unbalanced. It is thus difficult to collect enough samples for every fault type under different working conditions. 
 
**How to develop effective algorithms that can deal with scarcity of samples for a portion of fault types thus becomes a critical problem in fault diagnosis community.**

In paper, we propose a deep neural network based few-shot learning approach for rolling bearing fault diagnosis with limited data. Our model is based on the siamese neural network, which learns by exploiting sample pairs of the same or different categories. Experimental results over the standard Case Western Reserve University (CWRU) bearing fault diagnosis benchmark dataset showed that our few-shot learning approach is more effective in fault diagnosis with limited data availability. When tested over different noise environments with minimal amount of training data, the performance of our few-shot learning model surpasses one of the baseline with reasonable noise level. When evaluated over test sets with new fault types or new working conditions, few-shot models work better than the baseline trained with all fault types. 

All our models and datasets in this study are open sourced and can be downloaded from https://mekhub.cn/as/fault_diagnosis_with_few-shot_learning/ 
