# Flexible Stroke Control in Fast Style Transfer

This repository contains the PyTorch code for our paper "Controlling Strokes in Fast Neural Style Transfer using Content Transforms", TVCJ 2022 (to appear).

Make sure to pull the repository with git-lfs to retrieve the models.

## Adjustable NST network
To run the adjustable model run the notebook `notebooks/adjustable.ipynb`, the contained interactive widget can be used to pick model variants, styles, and adjust stroke settings.

## Reversible Content Transformations
To test reversible content transformations, run the notebook `notebooks/reversible_warping.ipynb`, or use `apply_reversible.py` to create animated GIFs using content transformations, such as swirl, rotation or warping. Furthermore `reversible_edit/warp_gui.py` contains a GUI for adjusting strokes using reversible local deformations (i.e. thin spline warping).

## Training
The introduced adjustable nst network can be trained using `python adjustable_upscaleNst train`, it requires the ms_coco dataset, the args class contains possible configuration choices.
The code in `adaptiveStrokeNet.py` is a pytorch re-implementation of "Stroke Controllable Fast Style Transfer", Jing et al., 2018.


