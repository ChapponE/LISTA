Origins of unfolded networks and theory: Learning Iterative Soft Thresholding Algorithm (LISTA)
------------
Introduction
------------
This project aims to implement the first unfolded neural network described in "Learning fast approximations of sparse coding" and 3 variants of it introduced in the article "Theoretical Linear Convergence of Unfolded ISTA and its Practical Weights and Thresholds" https://arxiv.org/abs/1808.10038 which gives theoritical guarantees of convergence.

Installation
------------
detailed in requierements.txt file

Generating Data
------------
To generate datasets and apply blur (with customizable parameters including size of blur kernel, and the sigma's of blur and noise):
--> python data/generate.py

Train models: 
------------
To train models (LISTA, LISTA_CP, LISTA_SS, LISTA_CPSS)
python src/train/train_lista.py

- Test trained models:
------------
python src/test/plot_loss_psnr.py
python src/test/theorem_1.py
python src/test/discussion.py
python src/test/theorem_2_3.py
