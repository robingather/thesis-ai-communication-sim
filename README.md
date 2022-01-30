### Predator-prey simulation
This is the simulation used to answer the research questions of my thesis. You can run it for yourself, although you will need to install the PyTorch and PyGame packages.

You can edit hyperparamters in 'constants.py.' During the process of making my thesis, I generated a ton of models. The final ones can be found in the model folder. Note that not all of them my work as they require some additional statistics or lack additional statistics. You can load them by changing the MODEL_NAME constant in 'constants.py.'

Warning: changing the WORLD_SIZE or BLOCK_SIZE hyperparameter may have unintended consequences for the rendering. I settled on these hyperparameters in the last month of thesis writing to run all experiments and did not make the rendering methods general for changes in these sizes. Change them at your own risk.

This simulation used a snake-ai simulation made by Patrick Loeber as a jumping-off point. The original MIT license is included in this repository. I heavily, heavily modified the simulation for my own purposes.

Robin Gather