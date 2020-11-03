import logging
import pickle
import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import multiprocessing
from plotly.offline import plot
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import CoolProp.CoolProp as cp
from CoolProp.HumidAirProp import HAPropsSI as ha
# from paths import *
# from util import *
# from hp_simulation import HeatPumpModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('in module %(name)s, in func %(funcName)s, '
                              '%(levelname)-8s: [%(filename)s:%(lineno)d] %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
if not len(logger.handlers):
    logger.addHandler(stream_handler)
    logger.propagate = False


def optimize():
    t_db, phi = 20, 0.35
    cold = 1_000



if __name__ == '__main__':
    optimize()
