# Cross-media Structured Common Space for Multimedia Event Extraction

Table of Contents
=================
  * [Overview](#overview)
  * [Requirements](#requirements)
  * [Data](#data)
  * [Quickstart](#quickstart)
  * [Citation](#citation)

## Overview
The code for paper [Cross-media Structured Common Space for Multimedia Event Extraction](http://blender.cs.illinois.edu/software/m2e2/).

<p align="center">
  <img src="./training-testing.png" alt="Photo" style="width="100%;"/>
</p>

## Requirements

You can install the environment using `requirements.txt` for each component.

```pip
pip install -r requirements.txt
```

## Data

### Situation Recognition (Visual Event Extraction Data)
We download situation recognition data from [imSitu](http://imsitu.org/) . Please find the preprocessed data in [PreprcessedSR](https://drive.google.com/drive/folders/1h0qwYWeGEoCx8m-zwH-XcoPSyffmrC-c?usp=sharing).

### ACE (Text Event Extraction Data)
We preprcoessed ACE following [JMEE](https://github.com/lx865712528/EMNLP2018-JMEE/tree/master). The preprocessing script is in `dataflow/preprocess_ace_JMEE.py`, and the sample data format is in [sample.json](https://github.com/lx865712528/EMNLP2018-JMEE/blob/master/ace-05-splits/sample.json). Due to license reason, the ACE 2005 dataset is only accessible to those with LDC2006T06 license, please drop me an email showing your possession of the license for the processed data.

### Voice of America Image-Caption Pairs
We crawled VOA image-captions to train the common space, including [images](http://pineapple.cs.columbia.edu:8007/voa_images.tgz), [captions](https://uofi.box.com/s/xtn9p6m8z5qtjbbi5tqrl45tn6apew4x). We preprocess the data including object detection, and parse text sentences. The preprocessed data is in [PreprocessedVOA](https://drive.google.com/drive/folders/1I9vMGIhWZpKqxQYip91eLoDRnrkqRxnt?usp=sharing).

### M2E2 (Multimedia Event Extraction Benchmark)

The images and text articles are in [m2e2_rawdata](https://drive.google.com/file/d/1xtFMjt_eYgeBts5rBomOWbPo7wV_mnhy/view?usp=sharing), and annotations are in [m2e2_annotation](http://blender.cs.illinois.edu/software/m2e2/m2e2_v0.1/m2e2_annotations.tgz).


## Quickstart

### Training

We have two variants to parse images into situation graph, one is parsing images to role-driven attention graph, and another is parsing images to object graphs.

(1) attention-graph based version
```bash
sh scripts/train/train_joint_att.sh 
```
(2) object-graph based version: 
```bash
sh scripts/train/train_joint_obj.sh 
```
Please specify the data paths `datadir`, `glovedir` in scripts. 


### Testing

(1) attention-graph based version
```
sh test_joint.sh
```
(2) object-graph based version: 
```bash
sh test_joint_object.sh
```

Please specify the data paths `datadir`, `glovedir`, and model paths `checkpoint_sr`, `checkpoint_sr_params`, `checkpoint_ee`, `checkpoint_ee_params` in scripts. 


## Citation

Manling Li, Alireza Zareian, Qi Zeng, Spencer Whitehead, Di Lu, Heng Ji, Shih-Fu Chang. 2020. Cross-media Structured Common Space for Multimedia Event Extraction. Proceedings of The 58th Annual Meeting of the Association for Computational Linguistics.
```
@inproceedings{li2020multimediaevent,
    title={Cross-media Structured Common Space for Multimedia Event Extraction},
    author={Manling Li and Alireza Zareian and Qi Zeng and Spencer Whitehead and Di Lu and Heng Ji and Shih-Fu Chang},
    booktitle={Proceedings of The 58th Annual Meeting of the Association for Computational Linguistics},
    year={2020}
```
