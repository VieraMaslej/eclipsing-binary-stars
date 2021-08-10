# Automatic classification of eclipsing binary stars using deep learning methods

Viera Maslej Krešňáková, Michal Čokina, Peter Butka, Štefan Parimucha

### Abstract

In the last couple of decades, tremendous progress has been achieved in developing robotic telescopes and, as a result, sky surveys (both terrestrial and space) have become the source of a substantial amount of new observation data. These data contain a lot of information about binary stars, hidden in their light-curves. With the huge amount of gathered astronomical data, it is not reasonable to expect their manual processing and analysis. Therefore, in this paper, we focus on the automatic classification of eclipsing binary stars using deep learning methods. Our classifier provides a tool for the categorization of light curves of binary stars into two classes: detached and over-contact. We used the ELISa software to obtain synthetic data, which we then used for the  training of the classifier. For testing purposes, we collected 100 light curves of observed binary stars, in order to test a number of classifiers. We tested semi-detached eclipsing binary stars as detached. The best-performing classifier combines bidirectional Long Short-Term Memory (LSTM) and a one-dimensional convolutional neural network, which achieved 98% accuracy on the test set. Omitting semi-detached eclipsing binary stars, we could obtain 100% accuracy in classification. 

### Full text is avaiable on https://doi.org/10.1016/j.ascom.2021.100488

### ELISa software 
ELISa software available on https://github.com/mikecokina/elisa

Dataset of synthetic data from ELISa software: https://mega.nz/file/jYxVWQgK#FVfqyz57jNGxHOPG6XI3PXyohjLsJDZ_9lDqHJ-AMkg

Observation data: observed_lc.csv

### BibTeX:

@article{COKINA2021100488,<br/>
author = {{\v{C}}okina, M and Maslej-Kre{\v{s}}ň{\'{a}}kov{\'{a}}, V and Butka, P and Parimucha, {\v{S}}},<br/>
doi = {https://doi.org/10.1016/j.ascom.2021.100488},<br/>
issn = {2213-1337},<br/>
journal = {Astronomy and Computing},<br/>
keywords = {Classification,Deep learning,Eclipsing binary stars,Light curves},<br/>
pages = {100488},<br/>
title = {{Automatic classification of eclipsing binary stars using deep learning methods}},<br/>
url = {https://www.sciencedirect.com/science/article/pii/S2213133721000421},<br/>
year = {2021}<br/>
}
