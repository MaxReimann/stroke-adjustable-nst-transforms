# Flexible Stroke Control in Fast Style Transfer

This repository contains the PyTorch code for our paper "Controlling Strokes in Fast Neural Style Transfer using Content Transforms", TVJC 2022 (to appear).

Make sure to pull the repository with git-lfs to retrieve the models.

To run the adjustable model run the notebook `notebooks/adjustable.ipynb`, the contained interactive widget can be used to adjusts model variants, styles, and settings.
To test reversible content transformations, run the notebook `notebooks/reversible_warping.ipynb`.

The introduced adjustable nst network can be trained using `python adjustable_upscaleNst train`, it requires the ms_coco dataset, the args class contains possible configuration choices.
The code in `adaptiveStrokeNet.py` is a pytorch re-implementation of "Stroke Controllable Fast Style Transfer", Jing et al., 2018.


