# Glove
This container is used to train a Glove embeddings on a corpus. 

## Requirements
 * Singularity (>=3.9.0) - How to install: [here](https://sylabs.io/guides/3.0/user-guide/installation.html)

## Data and model directory tree
Data and models should be store in the following way  

```
.
└--data
|   |
|   └--preprocessed
|       |
|       └--full_domains
|           |
|           └--lemmas
|
└---models
    |
    └--glove

```


## Bind to data
`singularity exec --bind /path/to/data:/mnt my_container.sif`