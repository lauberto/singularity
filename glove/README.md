# Glove
This container is used to train Glove embeddings on a corpus for the CAT&kittens project. 

## Requirements
 * Singularity (>=3.9.0) - [Installation](https://sylabs.io/guides/3.0/user-guide/installation.html)

## Build the image
Building an image requires root privileges. So it is often more convenient to build it on your local machine, and then copy it to your server with `scp`.
Refer to [this tutorial](https://github.com/bdusell/singularity-tutorial) for more info.  
`sudo singularity build version-1.sif version-1.def`

## Exec training
When you exec the train script inside the container you should bind the host directories where you store text data (your corpus), trained models and logs, to the 
inside-container directories, like so:  
`singularity exec --bind ./path/to/host/data:/container/data,./path/to/host/models:/models,./path/to/host/log:/container/log version-1.sif python3 train.py`