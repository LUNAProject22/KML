# KML: Knowledge Module Learning  
# PKR-QA: A Benchmark for Procedural Knowledge Reasoning

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)
![Dataset](https://img.shields.io/badge/dataset-PKR--QA-orange)
![Status](https://img.shields.io/badge/status-active-success)

---

## Overview

**Knowledge Module Learning (KML)** is a neurosymbolic framework that learns structured knowledge modules from relational data and performs procedural reasoning over multi-step tasks.

**PKR-QA** is the first benchmark for **Procedural Knowledge Reasoning**, combining instructional videos (COIN dataset), knowledge graphs, step predictions, and structured question-answer pairs.

This repository contains:

- KML source code  
- PKR-QA dataset (JSON format)  
- ProcedureVRL task/step predictions  
- Knowledge graph files  
- Training and evaluation scripts  

<img width="992" height="916" alt="KML" src="https://github.com/user-attachments/assets/f1ffcf19-28ba-4c56-930f-9447a41a8efb" />


---

## Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Dataset Structure](#dataset-structure)  
- [COIN Dataset](#coin-dataset)  
- [ProcedureVRL Predictions](#procedurevrl-predictions)
- [Installation](#Installation)
- [Citations](#citations)  
- [License](https://github.com/LUNAProject22/KML/blob/main/LICENSE)

---

## Dataset

Download and extract the PKR-QA dataset using:

```bash
tar -I zstd -xf pkr-qa.tar.zst
```

## PKR-QA includes:

- A knowledge graph (cointrain_kgv2.json)

- Step and task predictions from ProcedureVRL

- QA splits for training, validation, and testing

- Small sample splits for fast prototyping

## Dataset Structure

```pgsql
dataset/
├── cointrain_kgv2.json                     # Knowledge graph (KG)
├── QA_25Oct24_testing_pred.json            # ProcedureVRL predictions (test)
├── QA_25Oct24_validation_pred.json         # ProcedureVRL predictions (val)
└── s4_QADataset_12Feb2025/
    ├── testing.json    
    ├── train/
    │   └── training_small_100.json
    └── val/
        └── validation_small_50.json
```
<img width="1940" height="758" alt="PQR-QA" src="https://github.com/user-attachments/assets/646ad83b-0d9a-4544-998f-c26d132577b1" />

## COIN Dataset

Videos used in PKR-QA are from the COIN dataset:
https://coin-dataset.github.io/

```bibtex
@INPROCEEDINGS{
    title={COIN: A Large-scale Dataset for Comprehensive Instructional Video Analysis},
    author={Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, Jie Zhou},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2019}
}
```

## ProcedureVRL Predictions

PKR-QA uses ProcedureVRL for task and step predictions:
https://github.com/facebookresearch/ProcedureVRL

```bibtex
@inproceedings{zhong2023learning,
  title={Learning Procedure-aware Video Representation from Instructional Videos and Their Narrations},
  author={Zhong, Yiwu and Yu, Licheng and Bai, Yang and Li, Shangwen and Yan, Xueting and Li, Yin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14825--14835},
  year={2023}
}
```
## Installation

- Clone the repo
- Run setup.sh
- Need pytorch 2.8 or later.

## Train and testing

```bash
python KML_Main.py -s
```

## Citations
```bibtex
@article{nguyen2025neuro,
  title={Neuro Symbolic Knowledge Reasoning for Procedural Video Question Answering},
  author={Nguyen, Thanh-Son and Yang, Hong and Neoh, Tzeh Yuan and Zhang, Hao and Keat, Ee Yeo and Fernando, Basura},
  journal={arXiv preprint arXiv:2503.14957},
  year={2025}
}

@inproceedings{nguyen2025aaai,
  title={PKR-QA: A Benchmark for Procedural Knowledge Reasoning with Knowledge Module Learning},
  author={Nguyen, Thanh-Son and Yang, Hong and Neoh, Tzeh Yuan and Zhang, Hao and Keat, Ee Yeo and Fernando, Basura},
  booktitle={AAAI},
  year={2026}
}
```


## Acknowledgments
This research/project is supported by the National Research Foundation, Singapore, under its NRF Fellowship (Award\# NRF-NRFF14-2022-0001) and by funding allocation to Basura Fernando by the A*STAR under its SERC Central Research Fund (CRF), as well as its Centre for Frontier AI Research.



