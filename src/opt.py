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
# import CoolProp.CoolProp as cp
# from CoolProp.HumidAirProp import HAPropsSI as ha
# from paths import *
from utils import plot_df
# from hp_simulation import HeatPumpModel
from tower import DataInterpolator
from chiller import Chiller

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('in module %(name)s, in func %(funcName)s, '
                              '%(levelname)-8s: [%(filename)s:%(lineno)d] %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
if not len(logger.handlers):
    logger.addHandler(stream_handler)
    logger.propagate = False


def range_mapper(keys_array, value):
    if not isinstance(keys_array, list):
        keys_array = list(keys_array)
    if value < keys_array[0][0]:
        return keys_array[0]
    elif value > keys_array[-1][1]:
        return keys_array[-1]
    else:
        for k in keys_array:
            if k[0] <= value <= k[1]:
                return k


class WeatherReader:
    def __init__(self, weather_file=os.path.join('..', 'data', 'csv', 'amb.csv')):
        self.weather_file = weather_file
        self.data = self.read()

    def read(self, trim_bool=False):
        df = pd.read_csv(
            self.weather_file, index_col=[0], parse_dates=[0],
            date_parser=lambda t: pd.to_datetime(t, format='%d.%m.%Y %H:%M:%S')
        ).drop(['Unnamed: 3'], axis=1)
        df.columns = ['rel_hum', 't_amb']
        # df.index.name = 'time'
        df.rel_hum *= 1e-2
        if trim_bool:
            df = df.loc[(df.index >= datetime(2019, 1, 1)) & (df.index <= datetime(2019, 12, 31, 23, 59)), :]
        return df


class TowerReader:
    def __init__(self, tower_files=None):
        folder = os.path.join('..', 'data')
        if tower_files is None:
            _tower_files = tower_files if tower_files is not None else ['tower_range_1.pickle', 'tower_range_2.pickle']
            self.tower_files = [os.path.join(folder, f) for f in _tower_files]
        else:
            self.tower_files = os.path.join(folder, tower_files)

    @staticmethod
    def read_pickle(file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        return data

    def read_list(self):
        res_d = dict()
        for i, f in enumerate(self.tower_files):
            res_d[i] = self.read_pickle(f)
        return res_d

    def read(self):
        return self.read_pickle(self.tower_files)

    @staticmethod
    def write_pickle(data, f_name):
        with open(f_name, 'wb') as f:
            pickle.dump(data, f)


class Load:
    def __init__(self, load_file=os.path.join('..', 'data', 'cooling_load.csv')):
        self.load_file = load_file if os.path.exists(load_file) else None

    def read(self, min_threshold=500):
        df = pd.read_csv(self.load_file, index_col=[0], parse_dates=[0])
        df = df.resample('H').mean()
        df.loc[df.load <= min_threshold, 'load'] = 0
        return df


class DataReader:
    def __init__(self, f_path=None):
        self.f_path = f_path if f_path is not None else os.path.join('..', 'results', 'results_total.csv')

    def read(self):
        df = pd.read_csv(self.f_path, index_col=[0], parse_dates=[0])
        df.rename(mapper={
            'chiller_el': "w_el_new",
            'tc_in': "tc_in_new",
            'eer': "eer_new",
            'power_tot': "power_tot_new"
        }, axis=1, inplace=True)
        return df.drop(['t_tower_old', 'old_consumption', 'tower_wat'], axis=1)


def simulate_chillers(write_bool=False):
    df_sim = pd.read_csv(os.path.join('..', 'data', 'chillers_load_temp.csv'), index_col=[0], parse_dates=[0])
    chiller = Chiller()
    interpolator = DataInterpolator()
    df_sim.loc[(df_sim.load > 0) & (df_sim.tc_in_mean <= 18), 'tc_in_mean'] = 18

    # df_sim = df_sim.iloc[:72, :]
    res = list()
    for i, (ind, row) in enumerate(df_sim.iterrows()):
        chiller_char = chiller.get_family_curves(row.load)
        df = interpolator.interpolate_df(chiller_char, 'tc_in', row.tc_in_mean)
        df.index = [ind]
        if row.load > 0:
            ch_coeff = row.load / chiller_char.loc[chiller_char.index[0], 'hr_eva']
        else:
            ch_coeff = 1
        if ch_coeff != 1 or row.load > 6_600:
            logger.info(f'ch_coeff = {ch_coeff:.3f} !')
        df.loc[:, ['w_el', 'hr_con', 'hr_eva']] *= ch_coeff

        res.append(df)
    # print()
    dfr = pd.concat(res, axis=0)
    dfr.loc[dfr.hr_eva < 1, ['w_el', 'hr_con', 'eer']] = 0
    if write_bool:
        dfr.to_csv(os.path.join('..', 'data', 'chillers_sim.csv'))
    return dfr


def grid_search(tower_char: dict, chiller_char: pd.DataFrame, load: float):
    loss_dict = dict()
    hr_con = chiller_char.loc[chiller_char.index[0], 'hr_con']
    for i, (temp, df) in enumerate(tower_char.items()):
        tower_ind = np.argmin(np.abs(df.heat_fr - hr_con))
        df_sel = df.loc[df.index == tower_ind, :]
        chiller_ind = np.argmin(np.abs(chiller_char.tc_in - df_sel.tw_out.values[0]))
        ch_sel = chiller_char.loc[chiller_char.index == chiller_ind, :]

        if load > 0:
            ch_coeff = load / chiller_char.loc[chiller_char.index[0], 'hr_eva']
        else:
            ch_coeff = 1
        if ch_coeff != 1 or load > 6_600:
            logger.info(f'ch_coeff = {ch_coeff:.3f} !')
        # tower_coeff = ch_sel.hr_con.values[0] /
        loss_dict[i] = {
            'tower_ind': tower_ind,
            'tower_el': df_sel.power_el.values[0] * ch_coeff,
            'tower_wat': df_sel.wat.values[0] * ch_coeff,
            'chiller_ind': chiller_ind,
            'chiller_el': ch_sel.w_el.values[0] * ch_coeff,
            'tc_in': ch_sel.tc_in.values[0],
            'eer': chiller_char.eer.values[0]
        }

    value, key = 1e5, None
    for k, v in loss_dict.items():
        power = v.get('tower_el') + v.get('chiller_el')
        if power <= value:
            value = power
            key = k
    return pd.DataFrame(loss_dict.get(key), index=[0])
    # return loss_dict.get(key)


def opt_given_temp(write_bool=False, name='some_name.csv'):
    data = DataReader(os.path.join('..', 'results', 'config_new.csv')).read()

    weather = WeatherReader().read()

    tower = TowerReader('tower_total_new.pickle').read()

    chiller = Chiller()

    load = Load().read()

    interpolator = DataInterpolator()

    # data_mod = data.drop(data[~data.index.isin(weather.index)].index, axis=0)
    load_mod = load.drop(load[~load.index.isin(weather.index)].index, axis=0)
    df_sim = pd.concat([load_mod, weather.loc[weather.index.isin(load_mod.index), :], data.tc_in_old], axis=1)
    df_sim.loc[(df_sim.load > 0) & (df_sim.tc_in_old <= 18), 'tc_in_old'] = 18
    df_sim['iteration'] = np.arange(0, df_sim.shape[0], 1)
    df_sim = df_sim.assign(
        tower_el=0.0,
        tower_wat=0.0,
        chiller_el=0.0,
        eer=0.0
    )
    # cols_to_add = ['tower_el', 'tower_wat', 'chiller_el', 'eer']

    # df_sim = df_sim.loc[(df_sim.index >= datetime(2020, 8, 10, 15)) & (df_sim.index <= datetime(2020, 12, 30)), :]
    for ind, row in df_sim.iterrows():
        logger.info(f'Iteration # {int(row.iteration)} \t load = {row.load}')
        t_key = range_mapper(tower.keys(), row.t_amb)
        phi_key = range_mapper(tower.get(t_key).keys(), row.rel_hum)
        tower_char = tower.get(t_key).get(phi_key)
        chiller_char = chiller.get_family_curves(row.load)

        # simulate chiller.
        df_chill = interpolator.interpolate_df(chiller_char, 'tc_in', row.tc_in_old)
        df_chill.index = [ind]
        if row.load > 0:
            ch_coeff = row.load / chiller_char.loc[chiller_char.index[0], 'hr_eva']
        else:
            ch_coeff = 1
        if ch_coeff != 1 or row.load > 6_600:
            logger.info(f'ch_coeff = {ch_coeff:.3f} !')

        df_chill.loc[:, ['w_el', 'hr_con', 'hr_eva']] *= ch_coeff

        # simulate tower.
        hr_con = df_chill.loc[:, 'hr_con'].values[0]
        tower_temps = np.array(list(tower_char.keys()))
        tower_temp_ind = np.argmin(np.abs(tower_temps - row.tc_in_old))
        tower_temp = tower_temps[tower_temp_ind]
        df_tower = tower_char.get(tower_temp)
        df_tower_int = interpolator.interpolate_df(df_tower, 'heat_fr', hr_con)
        tower_coeff = hr_con / df_tower_int.heat_fr.values[0]
        df_tower_int.loc[:, ['power_el', 'wat', 'part_load', 'heat_fr']] *= tower_coeff

        # update df_sim with chiller results
        df_sim.loc[ind, ['chiller_el', 'eer']] = df_chill.loc[:, ['w_el', 'eer']].values[0]
        # update df_sim with tower results
        df_sim.loc[ind, ['tower_el', 'tower_wat']] = df_tower_int.loc[:, ['power_el', 'wat']].values[0]

    # save results
    if write_bool:
        df_sim.to_csv(os.path.join('..', 'results', name))
    # print()


def opt(write_bool=False, name='some_name.csv'):
    weather = WeatherReader().read()

    tower = TowerReader('tower_total_new.pickle').read()

    chiller = Chiller()

    load = Load().read()
    # print()

    load_mod = load.drop(load[~load.index.isin(weather.index)].index, axis=0)
    df_sim = pd.concat([load_mod, weather.loc[weather.index.isin(load_mod.index), :]], axis=1)
    # df_sim = load.copy()
    df_sim['iteration'] = np.arange(0, df_sim.shape[0], 1)
    df_sim = df_sim.assign(
        tower_el=0.0,
        tower_wat=0.0,
        chiller_el=0.0,
        tc_in=0.0,
        eer=0.0
    )
    cols_to_add = ['tower_el', 'tower_wat', 'chiller_el', 'tc_in', 'eer']

    df_sim = df_sim.loc[(df_sim.index >= datetime(2020, 8, 10, 15)) & (df_sim.index <= datetime(2020, 12, 30)), :]
    for ind, row in df_sim.iterrows():
        logger.info(f'Iteration # {int(row.iteration)} \t load = {row.load}')
        t_key = range_mapper(tower.keys(), row.t_amb)
        phi_key = range_mapper(tower.get(t_key).keys(), row.rel_hum)
        tower_char = tower.get(t_key).get(phi_key)
        chiller_char = chiller.get_family_curves(row.load)

        df_iter = grid_search(tower_char, chiller_char, row.load)
        df_iter.rename({df_iter.index[0]: ind}, axis=0, inplace=True)
        # update df_sim with results
        df_sim.loc[[ind], cols_to_add] = df_iter.loc[:, cols_to_add]

    # save results
    if write_bool:
        df_sim.to_csv(os.path.join('..', 'results', name))
    # print()


def validate_data():
    def interpolate_row(_df_, row):
        row_list = [row] if not isinstance(row, list) else row
        for row in row_list:
            _df_.loc[_df_.index == row, :] = (
                    (_df_.loc[_df_.index == row - 1, :].values +
                     _df_.loc[_df_.index == row + 1, :].values) / 2)

    def interpolate_df(dfc, t_prev, t_next):
        # dfc = data.copy()
        for c in dfc.columns:
            dfc.loc[:, c] = (
                    (tower.get(k).get(phi).get(t_prev)[c].values + tower.get(k).get(phi).get(t_next)[c].values) / 2)
        return dfc

    tower = TowerReader('tower_total_new.pickle').read()
    for k, v in tower.items():
        for phi, d in v.items():
            for temp, df in d.items():
                std = df.diff().heat_fr.std()
                if abs(std) >= 0.1:
                    print(df)
                    print(std)
                    print('Keys are:')
                    print()
                    print('-' * 80)
    # print()
    # s = 0
    # for j in range(2):
    #     s += j
    #     print(s, j)


if __name__ == '__main__':
    start_time = time.time()
    # opt(True)
    opt_given_temp(False, 'opt_given_temp.csv')
    # simulate_chillers(True)
    # validate_data()
    logger.info(f'Elapsed {time.time() - start_time :.2f} sec')
