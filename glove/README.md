# Glove
This container is used to train Glove embeddings on a corpus for the CAT&kittens project. 

## Requirements
 * Singularity (>=3.9.0) - [Installation](https://sylabs.io/guides/3.0/user-guide/installation.html)

## Bind to data
`singularity exec --bind /path/to/data:/mnt my_container.sif`