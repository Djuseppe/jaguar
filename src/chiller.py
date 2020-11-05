import logging
import pickle
import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import multiprocessing
from plotly.offline import plot
import plotly.graph_objs as go
from matplotlib import pyplot as plt
# from paths import *
# from util import *
# from hp_simulation import HeatPumpModel
from utils import plot_df

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('in module %(name)s, in func %(funcName)s, '
                              '%(levelname)-8s: [%(filename)s:%(lineno)d] %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
if not len(logger.handlers):
    logger.addHandler(stream_handler)
    logger.propagate = False


class Chiller:
    def __init__(self):
        self.eff_22 = self.read_csv(os.path.join('..', 'data', 'csv', 'chiller_eff_22.csv'))
        self.eff_25 = self.read_csv(os.path.join('..', 'data', 'csv', 'chiller_eff_25.csv'))
        self.eff_30 = self.read_csv(os.path.join('..', 'data', 'csv', 'chiller_eff_30.csv'))
        self.ch = pd.concat([self.eff_22, self.eff_25, self.eff_30], axis=0)
        self.heat_eva_max = 6_600

    @staticmethod
    def read_csv(f_name):
        eff = pd.read_csv(f_name, skiprows=2, index_col=[0]).drop(['Unnamed: 6'], axis=1)
        eff.columns = [
            'part_load', 'heat_fr', 'w_el', 'eer', 'eva_fr',
            'te_in', 'te_out', 'dp_e', 'con_fr', 'tc_in', 'tc_out', 'dp_c'
        ]
        eff.part_load *= 1e-2
        return eff

    @staticmethod
    def find_closest(array: np.array, value):
        ind = (np.abs(array - value)).argmin()
        return ind, array[ind]

    def find_closest_values(self, array, value, limits=(660, 6600)):
        array = np.asarray(array)
        array.sort()
        closest_ind, closest_val = self.find_closest(array, value)
        if limits[0] <= value <= limits[1]:
            if closest_val < value:
                lower_val = closest_val
                higher_val = array[closest_ind + 1]
            elif closest_val > value:
                higher_val = closest_val
                lower_val = array[closest_ind - 1]
            else:
                lower_val = higher_val = closest_val
        else:
            higher_val = lower_val = closest_val
        return higher_val, lower_val

    def interpolate_df(self, q, plot_bool=False):
        q_high, q_low = self.find_closest_values(self.eff_30.heat_fr.values, q, limits=(660, 6600))

        def func(x, k, b):
            return k * x + b

        if q_high != q_low:
            # logger.info(f'q_low = {q_low}, q = {q}, q_high = {q_high}')
            df_l, df_h = self.ch.loc[self.ch.heat_fr == q_low, :], self.ch.loc[self.ch.heat_fr == q_high, :]

            coeffs_low = self.fit_func(df_l.tc_in, df_l.w_el, func)
            coeffs_high = self.fit_func(df_h.tc_in, df_h.w_el, func)

            k_new = self._interpolate(q, q_low, q_high, coeffs_low[0], coeffs_high[0])
            b_new = self._interpolate(q, q_low, q_high, coeffs_low[1], coeffs_high[1])
            if plot_bool:
                plt.plot(df_l.tc_in, func(df_l.tc_in, k_new, b_new), label='opt')
                plt.plot(df_l.tc_in, df_l.w_el, label='low data')
                plt.plot(df_h.tc_in, df_h.w_el, label='high data')
                plt.legend()
                plt.show()
        else:
            df = self.ch.loc[self.ch.heat_fr == q_low, :]
            k_new, b_new = self.fit_func(df.tc_in, df.w_el, func)
        return k_new, b_new

    def get_curve(self, q, tc_in, t_limits=(18.33, 40)):
        coeffs = self.interpolate_df(q)
        w_el = coeffs[0] * tc_in + coeffs[1]
        eer = q / w_el
        return w_el, eer

    @staticmethod
    def _interpolate(x, x1, x2, y1, y2):
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    @staticmethod
    def fit_func(x_data, y_data, func):
        coeffs_opt, _ = curve_fit(func, x_data, y_data)
        return coeffs_opt

    def get_family_curves(self, q):
        q = q if q <= self.heat_eva_max else self.heat_eva_max
        res = list()
        for t in np.arange(18, 36, 2):
            w_el, eer = self.get_curve(q, t)
            res.append((w_el, eer, t, q, w_el * (eer + 1)))
        return pd.DataFrame(data=res, columns=['w_el', 'eer', 'tc_in', 'hr_eva', 'hr_con'])


def main():
    ch = Chiller()
    q = 3600
    res = list()

    # ch.get_curve(q, 30)
    for t in np.arange(18, 36, 2):
        w_el, eer = ch.get_curve(q, t)
        res.append((w_el, eer, t))
    df = pd.DataFrame(data=res, columns=['w_el', 'eer', 'tc_in'])
    print(df)
    # res = ch.find_closest_values(ch.eff_30.heat_fr.values, q)
    # print(res)


if __name__ == '__main__':
    main()
