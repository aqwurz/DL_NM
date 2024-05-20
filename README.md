# DL-NM
Combining Deep Learning with Neuromodulated Neural Networks

## Contents of this repository

All listed items are in the `dfnn` directory.

- `data` directory:
    - `data/gifs`: All recordings of the best agents per configuration, for Section 5.3.2.
    - `data/rl_evaluation`: Metric data from all agents, for Section 5.3.1.
    - `data/tensorboard`: Raw data from the experiments, in Tensorboard format.
    - `data/good_population.pickle`: A population of 400 diffusion-based neuromodulatory (NM) networks, evolved for 500 generations. The best of these networks was used for the experiments.
- Implementation of diffusion-based NM network:
    - `main.py`: The script that runs the evolution of the networks. Run `python main.py --help` for usage information.
    - `environment.py`: A class implementing the environment for the foraging task.
    - `network.py`: The network class.
    - `pnsga.py`: Functions implementing PNSGA.
    - `postevolution.py`: A script that performs post-evolution analysis, as described in Velez et al.'s paper.
    - `setup.py`: A script for compiling the Cython files.
    - `test_suite.py`: Functions implementing unit tests of the implementation. Run using Pytest.
    - `test_utils.pyx`: Functions implementing unit tests of Cython-based components. Dependency of `test_suite.py`. **NOTE: `test_update_weights` is broken**
    - `utils.pyx`: A Cython library implementing network functionality.
    - `utils.pxd`: Headers for `utils.pyx`.
- Implementation of main experiments:
    - `train.py`: The script that trains the agents on the environment. Run `python train.py --help` for usage information.
    - `examine.py`: A script for recording agents. Also contains a function for logging evaluation metrics.
    - `tst_environment.py`: The class implementing the environment for the experiments.
- Data visualization scripts:
    - `plot.py`: Compares fitness per generation of network evolution with data from the original experiment. Used to create Figure 3.6.
    - `plot_evaluation.py`: Creates box plots of evaluation metrics. Used to create Figures A.1 through A.8.
    - `plot_final.py`: Script for creating training reward plots or statistical significance plots. Used to create Figures 5.1 and 5.2, as well as 5.3 and 5.4. Note that due to a change in implementation, the significance plots generated by this script are different to the ones in the thesis.

## Installation instructions

``` sh
$ cd /path/to/dir
$ git clone https://github.com/aqwurz/DL_NM.git
$ cd DL_NM/dfnn
$ python setup.py build_ext --inplace
```

## Dependencies

- Python >= 3.10
- PyTorch >= 2.1.2
- Stable Baselines 3 >= 2.1.0
- [Fork of Minigrid (implements custom tiles)](https://github.com/aqwurz/Minigrid "Minigrid fork") 
- Gymnasium >= 0.29.1
- Cython >= 3.0
- Numba >= 0.59.0
- Pythran >= 0.15.0
- Numpy >= 1.26.1
- tqdm >= 4.66.2
- ImageIO >= 2.34.1
- Matplotlib >= 3.8
- Pandas >= 1.5.3
- SciPy >= 1.13.0
- Seaborn >= 0.12.2
- tbparse >= 0.0.8


