# Assignment 3
## Econ 8310 - Business Forecasting

For homework assignment 3, you will work with our baseball pitch data (available in Canvas).

- You must create a custom data loader as described in the first week of neural network lectures to load the baseball videos [2 points]
- You must create a working and trained neural network (any network focused on the baseball pitch videos will do) using only pytorch [2 points]
- You must store your weights and create an import script so that I can evaluate your model without training it [2 points]

Submit your forked repository URL on Canvas! :) I'll be manually grading this assignment.

Some checks you can make on your own:
- Can your custom loader import a new video or set of videos?
- Does your script train a neural network on the assigned data?
- Did your script save your model?
- Do you have separate code to import your model for use after training?





# Answer Assignment 3 — ECON 8310

This repository contains my implementation for Assignment 3, including:
- A custom PyTorch dataset (dataset.py)
- A neural network model (model.py)
- Training script (train.py)
- Inference script (inference.py)

Download the dataset from the instructor’s OneDrive (Canvas) and place it in:  
`data/Project Frames/`

Run training with `python train.py` and inference with `python inference.py`.

