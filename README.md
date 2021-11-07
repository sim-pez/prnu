# Camera Identification task using VDNet and VDID

- **Academic year:** 2020-2021
- **Project title:** Camera Identification task using VDNet and VDID
- **Students:** [Abdullah Chaudhry](https://github.com/chabdullah) and [Simone Pezzulla](https://github.com/daikon899)
- **CFUs:** 6

# Overview
As  explained  in  [5],  digital  images  and  video  continue  toreplace  their  analog  counterparts,  the  importance  of  reliable,inexpensive  and  fast  identification  of  digital  image  originwill  only  increase.  Reliable  identification  of  the  device  usedto  acquire  a  particular  digital  image  would  especially  proveuseful, for example, in the court for establishing the origin ofimages presented as evidence.

This project is based on the work of [2], the goal is to extractsensor’s fingerprint from the captured image and compare thisreference  with  other  query  fingerprints  in  order  to  performa  camera  identification  task.  As  suggested  in  [2],  they  usePhoto-Response  Non-Uniformity  (PRNU)  extracted  from  flatand  not  saturated  images  as  a  unique  fingerprint  of  digitalcamera and use Peak-correlation-to-correlation-ratio (PCE) forthe identification task. We use different algorithms in order toextract and compare digital camera fingerprints. In this reportwe extend the implementation provided in [2] by adding 2 new noise extraction methods: **VDNet**[1] and **VDID**[3]. 

# VDNet
VDNet uses a variational inference for non-iid  real-noise estimation and image denoising in a unique Bayesian Network. Specifically, an approximate posterior, parameterized by deep neural  networks,  is  presented  by  taking  the  intrinsic  clean image and noise variances as latent variables conditioned on the input noisy image.

# VDID
Variational Deep Image Denoise (VDID) is a bayesian framework that can handle blind scenarios based on the variational approximation of objective functions separating the complicated problem into simpler ones.
Its main characteristics are:
- It handles **both AWGN and real-world noise**
- Trained in an end-to-end scheme without any additional noise information
- Requires fewer parameters than state-of-the-art denoisers
    
The objective is formulated in terms of **maximum a posterior** (MAP) inference. An approximated form of the objective is calculated by introducing a latent variable based on variational Bayes which incorporates the underlying noisy image distribution.
Based on the latent space, VDID can focus on simpler subdistributions of the original problem.

# Dataset
For the experiments we used the VISION dataset providedby LESC laboratory [4]. It contains about 30 devices and foreach  device  there are flat and not saturated images from which the reference fingerprints are extracted. Each of them is then compared with 20 natural image query fingerprints.

# Repository structure
The respository is structured as follows:
- The ```Dataset/Videos/``` directory contains all user and trainer ```.mp4``` videos.
- ```Dataset/Frames/``` will contain all the frames extracted from the videos.

## Installation

```
git clone https://github.com/daikon899/PRNU
cd PRNU
pip3 install -r requirements.txt
```

## Usage
- Replace ```.txt``` files with your ```.mp4``` videos in ```Dataset/Videos/```. You can **ignore this step**, in this case pre-extracted skeletons will be used (located in  ```Dataset/Skeletons/```).

```
python3 Autoencoder.py <---optional
```


- Compute ROC
```
Usage: 
  computeROC.py [-h] [-space SPACE] [-metric METRIC] [-exer EXER] [-best_weights] [-silent]

This program compute ROC graph given a space, a metric and an exercise.

optional arguments:
  -h, --help      show this help message and exit
  -space SPACE    [r3n | latent(default) | gram]
  -metric METRIC  [custom-median-euclidean(default) | median-euclidean | euclidean]: specify it if space=r3n or space=latent
  -exer EXER      [armclap(default) | singlelunges | doublelunges | dumbbellcurl | pushup | squat]
  -best_weights   choose the best weights for exercise if space=latent
  -silent         suppress all prints and plots performed by computeROC.py

```

## Examples
### - Latent dimention (using best weights)
- Following this example you will get the ROC function using latent space, custom median  euclidean metric and the best weights to perform the classification task for the specified exercise. 
```
python3 computeROC.py -space latent -metric custom-median-euclidean -exer armclap -best_weights
```
#### Output
Output



# Project Documents
- For a detailed description of the experiments and results obtained refer to the [report](/docs/report.pdf).
- And also the [presentation](/docs/presentation.pdf)


# Tools and Techniques
The main tools used in this work:
- VDID: a bayesian framework for denoising that can handle blind scenarios.
- VDNet: uses a variational inference for non-iid  real-noise estimation and image denoising in a unique Bayesian Network.
- PyCharm is an integrated development environment used in computer programming, specifically for the Python language [8].
- Computersss.

# Bibliography
\[1\] https://github.com/zsyOAOA/VDNet

\[2\] http://dde.binghamton.edu/download/camera_fingerprin.

\[3\] https://github.com/JWSoh/VDIR

\[4\] https://lesc.dinfo.unifi.it/

\[5\] Jessica   Fridrich   Jan   Lukas   and   Miroslav   Goljan. "[Digital Camera Identification From Sensor Pattern Noise](http://ws2.binghamton.edu/fridrich/Research/double.pdf)", in: *IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY* In 2006.


# Acknowledgments
Elaborazione e Protezione delle Immagini - Computer Engineering Master Degree @[University of Florence](https://www.unifi.it/changelang-eng.html)
