# Entropic Dual Averaging
The repository accompanies the publication:
> Szymon Łagosz, Paweł Wachel and Przemysław Śliwiński, _A Dual Averaging Algorithm for On-line Modelling of Infinite Memory 
Nonlinear Systems_, IEEE Transactions on Automatic Control, to appear, https://ieeexplore.ieee.org/abstract/document/9965613

## Requirements
The code was tested with Python 3.8.9 on Ubuntu 16.04.7 64bit.

Install required python packages using:
 ```
 pip install -r requirements.txt
 ```

## Usage
The script `main.py` offers a command-line interface that facilitates running experiments described in the paper.
You can run it either by setting up your own python environment or, as we recommend, in a docker container.

To speed up computations, our repository contains prebaked partial results. If you would like to reproduce experiments
from the ground up, choose option `Delete partial results` in the CLI, prior to running any experiment.

### Running experiments in Docker
First, make sure that Docker is installed on your machine. Then, in the directory containing the code run:
```
docker build -t eda .
```
to build an image. Alternatively, you can pull it by running
```
docker pull szym4n/eda
```

In order to run the CLI, type
```
docker run -it --name eda_container eda 
```

To copy generated plots from the Docker container to your disk, run
```
docker cp eda_container:/opt/app/plots ./eda_plots
```
