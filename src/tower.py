import logging
import pickle
import os
import json
import time
import warnings
import pickle
import multiprocessing as mlp
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import multiprocessing
from plotly.offline import plot
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import CoolProp.CoolProp as cp
from CoolProp.HumidAirProp import HAPropsSI as ha
# from paths import *
# from util import *
# from hp_simulation import HeatPumpModel
from utils import plot_df

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('in module %(name)s, in func %(funcName)s, '
                              '%(levelname)-8s: [%(filename)s:%(lineno)d] %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
if not len(logger.handlers):
    logger.addHandler(stream_handler)
    logger.propagate = False


class Writer:
    def __init__(self, f_name):
        self.f_name = f_name

    def write(self, data):
        with open(self.f_name, 'wb') as f:
            pickle.dump(data, f)


class Air:
    def __init__(self, t_db, phi):
        self.t_o = 273.15
        self.p_o = 1e5
        # self.t_wb = t_wb
        # self.t_db = t_db
        self.phi = phi
        self.temp = t_db + self.t_o

    def get_wet_bulb(self):
        return ha('T', 'P', self.p_o, 'H', self.get_enthalpy(), 'R', 1) - self.t_o

    def get_enthalpy(self):
        return ha('H', 'P', self.p_o, 'T', self.temp, 'R', self.phi)


class DataInterpolator:
    @staticmethod
    def find_closest(array: np.array, value):
        ind = (np.abs(array - value)).argmin()
        return ind, array[ind]

    @staticmethod
    def _interpolate(x, x1, x2, y1, y2):
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    def find_closest_values(self, array, value):
        array = np.asarray(array)
        array.sort()
        limits = min(array), max(array)
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

    def interpolate_df(self, df, col_name, q):
        q_high, q_low = self.find_closest_values(df[col_name].values, q)

        if q_high != q_low:
            # logger.info(f'q_low = {q_low}, q = {q}, q_high = {q_high}')
            df_l, df_h = (df.loc[df[col_name] == q_low, :],
                          df.loc[df[col_name] == q_high, :])
            df_res = pd.DataFrame(data=np.zeros(shape=(1, df.shape[1])), columns=df.columns)
            for c in df_res.columns:
                df_res[c] = self._interpolate(q, q_low, q_high, df_l[c].values, df_h[c].values)
        else:
            df_res = df.loc[df[col_name] == q_low, :]
        return df_res


class CoolingTower(DataInterpolator):
    def __init__(self, tw_in, tw_out, ta_dry, phi, fr_a, fr_w, heat_fr, col_name='heat_fr', fan_input=4 * 45,
                 pump_input=2 * 4, fr_wet_wat=56.8, ta_wet=None, m=0.8, error=0.001):
        super().__init__()
        self.tw_in = tw_in
        self.phi = phi
        self.tw_out = tw_out
        self.ta_wet = ta_wet
        self.ta_dry = ta_dry
        self.fr_a = fr_a
        self.fr_w = fr_w
        self.fr_wet_wat = fr_wet_wat
        self.heat_fr = heat_fr
        self.m = m
        self.error = error

        self.cp_wat = 4.192
        self.cp_air = 1.005
        self.ntu, self.characteristics =\
            self.tower_iteration(self.fr_a, self.fr_w, self.tw_in, self.tw_out, t_db=self.ta_dry, phi=self.phi)
        self.fan_input = fan_input
        self.pump_input = pump_input

    def tower_iteration(self, fr_a, fr_w, tw_in, tw_out, t_db, phi, m=0.8):
        l_to_g = fr_w / fr_a

        coefs = np.array([0.1, 0.4, 0.6, 0.9])

        t_range = tw_in - tw_out
        tw_arr = tw_out + t_range * coefs

        hw = np.array([Air(t, 1).get_enthalpy() / 1e3 for t in tw_arr])

        ha_in = Air(t_db, phi).get_enthalpy() / 1e3
        h_a = ha_in + coefs * self.cp_wat * l_to_g * t_range

        h_sums = 1 / (hw - h_a)
        ntu = h_sums.sum() / 4 * t_range

        characteristics = ntu * l_to_g ** m
        return ntu, characteristics

    def get_water_min(self, tw_in, fr):
        return tw_in - self.heat_fr / (self.cp_wat * fr)

    def get_el_power(self, air_fr, wet_bool=True):
        power = self.fan_input * np.power(air_fr / self.fr_a, 3)
        if wet_bool:
            # power += self.pump_input * np.power(air_fr / self.fr_a, 3)
            power += self.pump_input
        return power

    def get_water_consumption(self, tw_in, tw_out):
        return self.fr_wet_wat * (tw_in - tw_out) * 0.024453684  # * 1.15

    def calculate(self, fr_air, fr_wat, t_db, phi, tw_in):
        # logger.info(f't_wb = {Air(t_db, phi).get_wet_bulb():.1f} C')
        e = 100
        counter = 0
        step = 0.5
        t_start = Air(t_db, phi).get_wet_bulb()
        res_arr = list()
        while e > self.error:
            t = t_start + step * counter
            _, c = self.tower_iteration(fr_a=fr_air, fr_w=fr_wat, tw_in=tw_in, tw_out=t, t_db=t_db, phi=phi)
            e = abs(self.characteristics - c)
            counter += 1
            res_arr.append((t, c, e))
            if counter >= 1_000 or t >= 50:
                # logger.debug('out of counter')
                break
        res = pd.DataFrame(res_arr, columns=['temp', 'characteristics', 'error'])
        # t_wat_out_min = self.get_water_min(fr_wat)
        tw_out = res.loc[res.error == res.error.min(), 'temp'].values[0]
        heat_fr = (tw_in - tw_out) * fr_wat * self.cp_wat
        # logger.info(f'power = {power:.0f} kW')
        return (tw_out,
                self.get_el_power(fr_air),
                self.get_water_consumption(tw_in, tw_out),
                res,
                heat_fr)

    def get_tower_curve(self, t_db, phi, fr_w, fr_a, tw_in):
        result = list()
        for part in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            temp, p, wat, _, heat_fr = self.calculate(fr_air=fr_a * part, fr_wat=fr_w, t_db=t_db, phi=phi, tw_in=tw_in)
            result.append((temp, p, wat, part, heat_fr))
        result = np.array(result)
        df = pd.DataFrame(result, columns=['tw_out', 'power_el', 'wat', 'part_load', 'heat_fr'])
        self.interpolate_results(df)
        return df
        # df.to_csv(os.path.join('..', 'data', 'linalg', 'lin_data.csv'))

    @staticmethod
    def get_indexes_to_interpolate(df, col_name='heat_fr'):
        dfi = df.diff().fillna(0)
        indexes = list()
        for ind, row in dfi.iloc[:-1, :].iterrows():
            if np.abs(row[col_name] - dfi.loc[ind + 1, col_name]) <= 500:
                indexes.append(ind)
        indexes = np.array(indexes)
        return indexes

    @staticmethod
    def fit_func(x_data, y_data, func):
        coeffs_opt, _ = curve_fit(func, x_data, y_data)
        return coeffs_opt

    @staticmethod
    def func(x, a, b):
        return a * x + b

    def get_line_coeffs(self, ser, indexes, plot_bool=False):
        df = pd.DataFrame(ser, index=np.arange(ser.shape[0]))
        df.columns = ['series']
        df['part_load'] = np.arange(0.1, 1.01, 0.1)
        # df.loc[~df.index.isin(indexes), 'series'] = np.nan

        x_data = df.loc[df.index.isin(indexes), 'part_load'].values
        y_data = df.loc[df.index.isin(indexes), 'series'].values
        try:
            coeffs = self.fit_func(x_data, y_data, self.func)
        except Exception as e:
            coeffs = None
            logger.error(e)
        if plot_bool:
            plt.plot(x_data, self.func(x_data, *coeffs), label='opt')
            plt.plot(x_data, y_data, label='data')
            plt.legend()
            plt.show()
        return coeffs

    def get_line(self, x_data, coeffs):
        line = self.func(x_data, *coeffs)
        return line

    def interpolate_results(self, df):
        indexes = self.get_indexes_to_interpolate(df, col_name='heat_fr')
        if indexes.any() and indexes.shape[0] > 1:
            x_data = df.loc[:, 'part_load'].values
            cols_to_iterate = [c for c in df.columns if 'power_el' not in c]
            for col in cols_to_iterate:
                ser = df.loc[:, [col]]
                # if ser.iloc[0, 0] is None:
                #     print()
                coeffs = self.get_line_coeffs(ser, indexes)
                if coeffs is not None:
                    df.loc[:, col] = self.get_line(x_data, coeffs)
                else:
                    df = None
        return df


def main():
    # m = 0.8
    l1, l2, l3 = np.array([155.5, 102, 52])
    g = 124.6 * 1.2
    ct = CoolingTower(tw_in=36, tw_out=30, ta_dry=37.5, phi=0.35, fr_a=g, fr_w=l1, heat_fr=3530)

    # res = ct.calculate(fr_air=g, fr_wat=l1, t_db=37.5, phi=0.35, tw_in=28)
    res, p, wat, df, heat_fr = ct.calculate(fr_air=g*0.6, fr_wat=l1, t_db=16, phi=0.35, tw_in=25)
    print(res)
    print(p, ' kW')
    print(wat, ' kg/s')
    df.error.plot()
    plt.show()


def calc():
    l1, l2, l3 = np.array([155.5, 102, 52])
    g = 124.6 * 1.2
    ct = CoolingTower(tw_in=36, tw_out=30, ta_dry=37.5, phi=0.35, fr_a=g, fr_w=l1, heat_fr=3530)
    t_db, phi = 30, 0.7
    logger.info('t_wb = {:.1f} C'.format(Air(t_db, phi).get_wet_bulb()))
    result = list()
    for part in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(part)
        temp, p, wat, _, heat_fr = ct.calculate(fr_air=g * part, fr_wat=l1, t_db=t_db, phi=phi, tw_in=27)
        result.append((temp, p, wat, part, heat_fr))
    result = np.array(result)
    df = pd.DataFrame(result, columns=['tw_out', 'power_el', 'wat', 'part_load', 'heat_fr'])
    ct.interpolate_results(df)
    print(df.head())
    print()
    plot_df(df)
    df.to_csv(os.path.join('..', 'data', 'linalg', 'lin_data.csv'))


def curve():
    fr_wat = 155.5
    fr_air = 124.6 * 1.2
    ct = CoolingTower(tw_in=36, tw_out=30, ta_dry=37.5, phi=0.35, fr_a=fr_air, fr_w=fr_wat, heat_fr=3530)
    t_db, phi = 20, 0.35
    df = ct.get_tower_curve(t_db, phi, fr_wat, fr_air, tw_in=20)
    plot_df(df)


def get_curve_family(ct: CoolingTower, t_db, phi, t_in_range=None):
    if t_in_range is None:
        t_in_range = range(20, 41, 2)
    df_d = dict()
    for t in t_in_range:
        df = ct.get_tower_curve(t_db, phi, ct.fr_w, ct.fr_a, tw_in=t)
        # df.name = f'tw_in_{t:.0f}'
        df_d[t] = df
    return df_d


def simulate_ct(ta_range: np.array, name: str, ta_step: int = 2, phi_step: float = 0.1):
    fr_wat = 155.5
    fr_air = 124.6 * 1.2

    # t_db, phi = 20, 0.35
    ct = CoolingTower(tw_in=36, tw_out=30, ta_dry=37.5, phi=0.35, fr_a=fr_air, fr_w=fr_wat, heat_fr=3530)

    def round_f(x):
        return round(x, 2)

    res = dict()
    for ta in ta_range:  # np.arange(10, 31, ta_step)
        phi_d = dict()
        tower_range = None if ta <= 20 else range(ta, 41, 2)
        for _phi in np.arange(0.2, 1.01, phi_step):  # np.arange(0.2, 1.01, phi_step)
            phi = round(_phi, 2)
            logger.info(f"@ {time.strftime('%d.%m.%Y %H:%M:%S')}: \t Calculating step for phi = {phi} and ta = {ta}")
            t_m, phi_m = (ta + ta_step) / 2, (phi + phi_step) / 2
            ct_curve = get_curve_family(ct, t_m, phi_m, tower_range)
            phi_d[(round_f(phi), round_f(phi + phi_step))] = ct_curve
        res[(ta, ta + ta_step)] = phi_d
    Writer(os.path.join('..', 'data', f'tower_{name}.pickle')).write(res)
    return res


def interpolate():

    interpolator = DataInterpolator()
    res = interpolator.interpolate_df(
        q=1300,
        df=pd.read_csv(os.path.join('..', 'data', 'linalg', 'tower_char.csv'), index_col=[0]),
        col_name='heat_fr'
    )
    print(res)


if __name__ == '__main__':
    start_time = time.time()
    # calc()
    main()
    # curve()

    # res = dict()
    # ta_step = 2
    # for ta in np.arange(0, 34, ta_step):
    #     phi_d = dict()
    #     phi_step = 0.05
    #     for phi in np.arange(0.2, 1.0, phi_step):
    #         round_f = lambda x: round(x, 2)
    #         t_m, phi_m = (ta + ta_step) / 2, (phi + phi_step) / 2
    #         curve = get_curve_family(ct, t_m, phi_m)
    #         phi_d[(round_f(phi), round_f(phi + phi_step))] = np.random.rand()
    #     res[(ta, ta + ta_step)] = phi_d

    # d = simulate_ct()
    # print(d.keys())
    # processes_list = list()
    # proc_1 = mlp.Process(
    #     target=simulate_ct,
    #     args=(np.arange(10, 21, 2), 'range_1')
    #     # args=([15], 'range_1')
    # )
    # processes_list.append(proc_1)
    # proc_1.start()
    #
    # # second range
    # proc_2 = mlp.Process(
    #     target=simulate_ct,
    #     args=(np.arange(20, 31, 2), 'range_2')
    #     # args=([25], 'range_2')
    # )
    # processes_list.append(proc_2)
    # proc_2.start()
    #
    # # run processes
    # for process in processes_list:
    #     process.join()
    #
    # # simulate_ct(ta_range=np.array([15]), name='test')
    # # interpolate()
    logger.info(f'Elapsed {time.time() - start_time :.2f} sec')
