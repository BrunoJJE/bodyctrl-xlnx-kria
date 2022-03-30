# 'bodyctrl' for the Kria board

This is a git repository for the project described here :

https://www.hackster.io/brunojje/human-pose-estimation-as-game-controller-4de02b


This project implement blazepose (human pose estimation) on the Kria board.

The detected poses are converted to commands sent to a server through ethernet. See the following repo for usage information and the associated server that must be run on the PC playing the game to be controlled.

https://github.com/BrunoJJE/bodyctrl-xlnx-pc


# Usage

You can edit the 'use_delegate' option in the python script to test with or without the TVM delegate of the blazepose landmark inference.

The script ('compile_tflite.py') used to generate the TVM delegate lib is included in the repo.

