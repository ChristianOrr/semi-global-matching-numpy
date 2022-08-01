# Semi-Global Matching Numpy

## Introduction
This is a reimplementation of the Semi-Global Matching algorithm using numpy. A python script and jupyter notebook is provided implementing SGM in the same way. The notebook provides visualizations and explanations for each step in the process. If you're new to depth estimation/SGM, its recommended to read the notebook first.

## SGM Brief Description
SGM is a popular classic depth estimation algorithm, known for having good accuracy for its speed. Its widely used in resource constrained  situations.

## Requirements
All you need to run the SGM algorithm are a pair of rectified stereo images. Stereo pair examples from the Middlebury dataset are provided for you (cones, figures and teddy). If you would like to review the algorithms accuracy, you will also need groundtruth disparities. 

If you would like to use your own stereo images and your stereo camera doesn't provide rectified stereo pairs, then you can calibrate your stereo camera using the methods shown in this repo: [stereo-camera-calibration](https://github.com/ChristianOrr/stereo-camera-calibration).

The python packages needed are shown in the requirements.txt. I've listed the versions of the packages I used, but it will probably also work with other versions.






## References
* [Previous Implementation in Python](https://github.com/beaupreda/semi-global-matching), by David-Alexandre Beaupre.
* [(SGM) Stereo Processing by Semi-Global Matching and Mutual Information](https://core.ac.uk/download/pdf/11134866.pdf), by Heiko Hirschmuller.
* [(Census) Nonparametric Local Transforms for Computing Visual Correspondence](http://www.cs.cornell.edu/~rdz/Papers/ZW-ECCV94.pdf), by R Zabih and J Woodfill.