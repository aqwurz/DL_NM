import numpy as np
from network import *
from main import *

layer_config = [
    [(i-2, 0) for i in range(5)],
    [(i-5.5, 1) for i in range(12)], 
    [(i-3.5, 2) for i in range(8)], 
    [(i-2.5, 3) for i in range(6)], 
    [(i-0.5, 4) for i in range(2)]
]
source_config = [(-3,2), (3,2)]

individual = {'network': Network(layer_config, source_config)}
print(individual['network'].connection_cost())
