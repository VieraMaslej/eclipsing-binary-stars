# Eclipsing-binary-stars-classification
## Automatic classification of eclipsing binary stars using deep learning methods

Viera Maslej Krešňáková, Michal Čokina, Peter Butka, Štefan Parimucha

### Abstract

In the last couple of decades, tremendous progress has been achieved in developing robotic telescopes and, as a result, sky surveys (both terrestrial and space) have become the source of a substantial amount of new observation data. These data contain a lot of information about binary stars, hidden in their light-curves. With the huge amount of gathered astronomical data, it is not reasonable to expect their manual processing and analysis. Therefore, in this paper, we focus on the automatic classification of eclipsing binary stars using deep learning methods. Our classifier provides a tool for the categorization of light curves of binary stars into two classes: detached and over-contact. We used the ELISa software to obtain synthetic data, which we then used for the  training of the classifier. For testing purposes, we collected 100 light curves of observed binary stars, in order to test a number of classifiers. We tested semi-detached eclipsing binary stars as detached. The best-performing classifier combines bidirectional Long Short-Term Memory (LSTM) and a one-dimensional convolutional neural network, which achieved 98% accuracy on the test set. Omitting semi-detached eclipsing binary stars, we could obtain 100% accuracy in classification. 

### ELISa software 
ELISa software available on https://github.com/mikecokina/elisa

Dataset of synthetic data from ELISa software: https://mega.nz/file/jYxVWQgK#FVfqyz57jNGxHOPG6XI3PXyohjLsJDZ_9lDqHJ-AMkg
