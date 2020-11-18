# Computational Design of Cold Bent Glass Façades

Code and data for research paper K. Gavriil\*, R. Guseinov\*, J. Pérez, D. Pellis, P. Henderson, F. Rist, H. Pottmann, and B. Bickel. [Computational Design of Cold Bent Glass Façades](http://visualcomputing.ist.ac.at/publications/2020/CDoCBGF/). *ACM Trans. Graph.* (2020).

\* _Joint first authors_

![Thumb](/meta/thumb.jpg)

### Abstract

Cold bent glass is a promising and cost-efficient method for realizing doubly curved glass façades. They are produced by attaching planar glass sheets to curved frames and must keep the occurring stress within safe limits. However, it is very challenging to navigate the design space of cold bent glass panels because of the fragility of the material, which impedes the form finding for practically feasible and aesthetically pleasing cold bent glass façades. We propose an interactive, data-driven approach for designing cold bent glass façades that can be seamlessly integrated into a typical architectural design pipeline. Our method allows non-expert users to interactively edit a parametric surface while providing real-time feedback on the deformed shape and maximum stress of cold bent glass panels. The designs are automatically refined to minimize several fairness criteria, while maximal stresses are kept within glass limits. We achieve interactive frame rates by using a differentiable Mixture Density Network trained from more than a million simulations. Given a curved boundary, our regression model is capable of handling multistable configurations and accurately predicting the equilibrium shape of the panel and its corresponding maximal stress. We show that the predictions are highly accurate and validate our results with a physical realization of a cold bent glass surface.

## Contents

* Folder **cgb_model**: Code for model training and usage.
* Folder **cgb_sim_reader**: Code for reading and writing simulation data.
* Folder **data**: Examples of data files (panel id 1020022889). Default folder for project data.

## Instructions

1. Clone or download code to local folder ``cold-glass-acm``.
1. Download data:
   * For DNN usage download file **mdn_model.tar.gz** (TODO: IST Research Explorer link).
   * For DNN training download file **optimal_panels_data.tar.gz** (TODO: IST Research Explorer link).
   * To get original simulation data download file **sim_data.tar.gz** (TODO: link).
1. Unpack downloaded archives in folder ``cold-glass-acm/data``

### cgb_model

Install correct versions of required python packages in a new conda environment.
```bash
cd cgb_model
conda create -y python=3.7 --prefix ./envs
conda activate ./envs
conda install -y -c anaconda pip
pip install -r requirements.txt
```
To run on GPU, make sure you have compatible cuDNN and CUDA versions installed.

Execute ``python cgbmodel.py`` to see the usage instructions.

### cgb_sim_reader

Compile C++ code 64-bit to read [cereal](https://uscilab.github.io/cereal/) binary data format (note that you need to clone git submodules for compilation). This project has minimal demo functionality to provide data IO in C++ code.

Execute ``CgbSimReader`` to see the usage instructions.
