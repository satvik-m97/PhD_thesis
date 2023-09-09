# PhD_thesis
All my PhD related work 
I will frequently update this file, but I am yet to work out the details
## Current theme
I am trying to generate mock cosmological data, using outputs from the known code CLASS[https://github.com/lesgourg/class_public]. The idea is to get the statistical quantities such as the Matter Power spectra and the associated Halo Mass Function and work with it

## Requirements
To make life simple, I have already produced and stored the files that are needed to produce mock realizations. It is work in progress.

### Docker
To use an environment where the code has been tested with all the denendencies, one can use the dockerfile present in the [Dockerfile](Docker/satvik/docker/). The command to build the docker image is 
```bash
docker build -t image_name:tag 
```
To run the said image, you can run it with
```bash
docker run -it image_name:tag /bin/bash
```
## Using
As of now, the repository should be cloned and then one can access the jupyter notebooks that does the realization

## Dependencies
Since all the files required are already present in the repository, installing python with the standard packages is enough. 
Eventually there might be a requirememnt for specific packages. Work in progress

### Important note

To merge with the main branch, please convert jupyter notebooks to python scripts using the command 
```bash
jupyter nbconvert jupyter_notebook.ipynb --to python --output-dir /path/to/Python_scripts/

