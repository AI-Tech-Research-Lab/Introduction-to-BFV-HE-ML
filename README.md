# Introduction-to-BFV-HE-ML

This repository contains the code used in the paper "Privacy-preserving deep learning with homomorphic encryption: an introduction".

Dependencies:
  - [PyTorch](https://pytorch.org/get-started/locally/)
  - [NumPy](https://numpy.org/)  
  - [Pyfhel](https://github.com/ibarrond/Pyfhel)

The repository is organized this way:
  - `BFV_theory` contains a Jupyter notebook (`BFV_theory.ipynb`) which introduces the reader to the BFV Homomorphic Encryption scheme, along with a (gentle) introduction to the main math concepts of the scheme, together with an implementation of the scheme made with Python, from scratch, using NumPy. This scheme construction should help the reader gain a deeper understanding of the scheme, before using more advanced and ready-to-use libraries (like SEAL, Pyfhel, etc.);
  - `HE-ML` contains a Jupyter notebook (`HE-ML.ipynb`) which introduces the reader to the concept of Homomorphic-Encryption enabled Machine-Learning. In the notebook a simple CNN, the LeNet-1, will be translated into a model able to work on encrypted data, along with an explanation on the different design choices, as well as on the encryption parameters setting;
  - `Experiments` contains the Python scripts used to obtain the experimental results shown in the paper. Running the scripts and looking at the corresponding file `.log`, it will be possible to reproduce and check the results.