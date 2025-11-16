<div align="center">    

# Evolving and Regularizing Meta-Environment Learner for Fine-Grained Few-Shot Class-Incremental Learning
[Li-Jun Zhao](https://Legenddddd.github.io/), Zhen-Duo Chen, Yongxin Wang, Xin Luo, and Xin-Shun Xu

</div>

## Dataset Preparation Guidelines

We follow [PFR](https://github.com/zichengpan/PFR) setting to use the same data index_list for training.  
Please set up data by referring to [PFR](https://github.com/zichengpan/PFR).


## Scripts

```
  python train.py -project mel -gamma 0.1 -lr_base 0.2 -decay 0.0005 -epochs_base 400 -schedule Cosine -gpu '0' -temperature 16 -dataroot Datasets/ -complex_weight 0.5 -part_weight 0.5 -part_num 4 
```


## Acknowledgment
Our project references the codes in the following repos.

- [PFR](https://github.com/zichengpan/PFR)
