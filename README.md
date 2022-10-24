
# Automatic assessment for rhythmic pattern imitations

This repository contains code for automatic assessment for rhythmic pattern imitations. Data used in the machine experiments is available here: [**MAST Rhythm Data Set**](https://zenodo.org/record/2620357#.Y1aVhHVBwW0) with annotations available in [the version 1.1 of the data set](https://zenodo.org/record/7243752#.Y1aWCnVBwW0). Please refer to the paper below for a detailed description of the tools, experiments and experiment results.

```latex
@article{GuzelEtAl2022,
  title={Automatic Assessment of Student Rhythmic Pattern Imitation Performances},
  author={Köktürk-Güzel, Başak Esin and Büyük, Osman and Bozkurt, Baris and Baysal, Ozan},
  journal={Digital Signal Processing},
  volume={Under review},
  number={},
  pages={},
  year={},
  publisher={Elsevier}
}
```
and if you use this code please cite it as above. 

## Content

The repo contains the following folders:
*   'data' folder contains a copy of annotation files and manually corrected onsets for a part of the data. 
*   'data_processing' folder contains all preprocessing and feature extraction code that prepares tabular data for machine learning experiments. Simply run the dataProcessPipeline.py script to download audio data from Zenodo, perform all necessary conversions, preprocesses and feature extraction steps. This will place ML tabular data in the 'data' folder
*   'ML_experiments' folder contains code for machine learning experiments as well as results obtained via running that code

The Python libraries used are listed in the requrements.txt file 


# Authorship

The code shared in this repo are authored by:

Baris Bozkurt: Data preprocessing and feature extraction, 
Osman Büyük: Experiments with Siamese networks, 
Basak Esin Köktürk Güzel: Machine learning experiments


# Acknowledgment

This study is supported by TUBITAK with grant number 121E198 as a part of the Scientific and Technological Research Projects Funding Program (1001).

