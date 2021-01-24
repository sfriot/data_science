# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:21:57 2019

@author: Sylvain Friot

Content: time series analysis and predictions - SARIMAX
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import logging as lg
import sys
import warnings

import modules_perso.sf_graphiques as sfg


"""
à écrire
"""
class TimeSerie:
    
    def __init__(self, data, time_index, serie_values, model_type='add'):
        try:
            if data[[time_index]].select_dtypes(include='datetime').shape[1] == 0:
                raise TypeError("Type Error : time_index must be of datetime type.")
            if (model_type != 'add') & (model_type != 'mul'):
                raise ValueError("Value Error : model_type must be 'add' or 'mul'.")
        except (TypeError, ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()))
        else:
            self.df = data.copy()
            self.time_index = time_index
            self.serie_values = serie_values
            self.model_type = model_type
            self.n_observations = len(data)
            
    def decomp_ma_auto(self, model_type=None, season_freq=None, centered_ma=True, estim_all=True):
        if model_type is None:
            model_type = self.model_type
        if estim_all:
            extrapolate_trend = 'freq'
        else:
            extrapolate_trend = 0
        decomp = sm.tsa.seasonal_decompose(self.df[self.serie_values], model=model_type, freq=season_freq, two_sided=centered_ma, extrapolate_trend=extrapolate_trend)
        if model_type == 'add':
            df_decomp = pd.DataFrame({self.serie_values: decomp.observed.values, "trend": decomp.trend.values, "seasonal": decomp.seasonal.values, "resid": decomp.resid.values, "cvs":decomp.observed.values - decomp.seasonal.values}, index=self.df[self.time_index])
        else:
            df_decomp = pd.DataFrame({self.serie_values: decomp.observed.values, "trend": decomp.trend.values, "seasonal": decomp.seasonal.values, "resid": decomp.resid.values, "cvs":decomp.observed.values / decomp.seasonal.values}, index=self.df[self.time_index])
        return df_decomp
    
    def decomp_ma_x11(self, model_type=None):
        if model_type is None:
            model_type = self.model_type
        decomp = pd.DataFrame({"y": self.df[self.serie_values].values}, index=self.df[self.time_index])
        decomp["tt1"] = self.__get_ma2(2, 12, decomp.y)        # estimation 1 tendance
        if self.model_type == 'add':
            decomp["rt1"] = decomp.y - decomp.tt1                   # estimation 1 reste saison + résidu
        else:
            decomp["rt1"] = 100 * decomp.y / decomp.tt1                   # estimation 1 reste saison + résidu
        for m in np.arange(1,13):
            m3x3 = decomp[decomp.index.month == m].rt1.dropna()
            if m == 1:
                interdf = self.__get_x11_ma2(3, 3, m3x3)
            else:
                interdf = pd.concat([interdf, self.__get_x11_ma2(3, 3, m3x3)], ignore_index=False)
        decomp["st1"] = interdf.copy()
        if self.model_type == 'add':
            decomp["stnorm1"] = decomp.st1.dropna() - self.__get_x11_ma2(2, 12, decomp.st1.dropna())
        else:
            decomp["stnorm1"] = 100 * decomp.st1.dropna() / self.__get_x11_ma2(2, 12, decomp.st1.dropna())
        for m in np.arange(6):  # les 6 premiers mois sont recopiés sur les premiers mois équivalents renseignés (idem pour les 6 derniers mois)
            decomp.iloc[m,4] = decomp.iloc[m+12,4]
            decomp.iloc[-m-1,4] = decomp.iloc[-m-13,4]
        if self.model_type == 'add':
            decomp["cvs1"] = decomp.y - decomp.stnorm1
        else:
            decomp["cvs1"] = 100 * decomp.y / decomp.stnorm1
        
        intertt = self.__get_henderson(13, decomp.cvs1)
        if self.model_type == 'add':
            interrt = decomp.cvs1 - intertt
            c = np.abs(intertt - intertt.shift(1)).mean()
            i = np.abs(interrt - interrt.shift(1)).mean()
        else:
            interrt = 100 * decomp.cvs1 / intertt
            c = np.abs(intertt / intertt.shift(1) - 1).mean()
            i = np.abs(interrt / interrt.shift(1) - 1).mean()
        if i / c < 1.0:
            decomp["tt2"] = self.__get_x11_henderson(9, decomp.cvs1)
        else:
            decomp["tt2"] = self.__get_x11_henderson(13, decomp.cvs1)
        if self.model_type == 'add':
            decomp["rt2"] = decomp.y - decomp.tt2                   # estimation 2 reste saison + résidu
        else:
            decomp["rt2"] = 100 * decomp.y / decomp.tt2                   # estimation 2 reste saison + résidu
        for m in np.arange(1,13):
            m3x5 = decomp[decomp.index.month == m].rt2.dropna()
            if m == 1:
                interdf = self.__get_x11_ma2(3, 5, m3x5)
            else:
                interdf = pd.concat([interdf, self.__get_x11_ma2(3, 5, m3x5)], ignore_index=False)
        decomp["st2"] = interdf.copy()
        if self.model_type == 'add':
            decomp["stnorm2"] = decomp.st2.dropna() - self.__get_x11_ma2(2, 12, decomp.st2.dropna())
        else:
            decomp["stnorm2"] = 100 * decomp.st2.dropna() / self.__get_x11_ma2(2, 12, decomp.st2.dropna())
        if self.model_type == 'add':
            decomp["cvs2"] = decomp.y - decomp.stnorm2
            decomp["resid"] = decomp.rt2 - decomp.stnorm2
        else:
            decomp["cvs2"] = 100 * decomp.y / decomp.stnorm2
            decomp["resid"] = 100 * decomp.rt2 / decomp.stnorm2
        df_decomp = pd.DataFrame({self.serie_values: decomp.y.values, "trend": decomp.tt2.values, "seasonal": decomp.stnorm2.values, "resid": decomp.resid.values, "cvs": decomp.cvs2.values}, index=self.df[self.time_index])
        return df_decomp
    
    """
    désaisonnalisation avec régression linéaire
    on cherche y(t) = a + b.t + c(s)*indicatrice(saison s)   ! y doit être une série temporelle de type additif
    modèle y(t) = somme(i)[alpha*Tt(i)] + somme(j)[beta(j)*St(j)] 
    -> b = alpha ; a = moyenne(beta(j)) ; c(s) = beta(j) - a
    """
    def decomp_lr_season(self, season_freq, return_coefs=False):
        if self.model_type == 'add':
            y = self.df[self.serie_values].values
        else:   # on passe la série en additif pour décomposer avec la régression linéaire
            y = np.log(self.df[self.serie_values]).values
        lr_df = pd.DataFrame({"y": y, "t": np.arange(self.n_observations) + 1} , index=self.df[self.time_index])
        sm_formula = "y ~ t"
        for i in range(season_freq):
            su = np.zeros(season_freq)
            su[i] = 1
            s = np.tile(su, self.n_observations//season_freq)
            s = np.append(s, su[:self.n_observations%season_freq])
            lr_df["s{}".format(i+1)] = s
            sm_formula = "{} + s{}".format(sm_formula, i+1)
        sm_formula = "{} - 1".format(sm_formula)
        regmodel = smf.ols(formula=sm_formula, data=lr_df).fit()
        a = regmodel.params[1:].mean()
        b = regmodel.params[0]
        c = regmodel.params[1:] - a
        my_trend = a + (b * lr_df.t)
        my_season = np.zeros(self.n_observations)
        for i in range(season_freq):
            my_season += (c[i] * lr_df["s{}".format(i+1)])
        my_resid = lr_df.y - my_trend - my_season
        df_decomp = pd.DataFrame({self.serie_values: lr_df.y, "trend": my_trend, "seasonal": my_season, "resid": my_resid, "cvs": lr_df.y - my_season}, index=self.df[self.time_index])
        if self.model_type != 'add':   # on corrige les résultats par l'exponentiel car modèle multiplicatif
            df_decomp = np.exp(df_decomp)
            a = np.exp(a)
            b = np.exp(b)
            c = np.exp(c)
        if return_coefs:
            return a, b, c
        return df_decomp
    
    def detrend_lr_auto(self, trend_polynom_order=1, season_freq=0):
        if self.model_type == 'add':
            interdata = pd.DataFrame({"y": self.df[self.serie_values].values}, index=self.df[self.time_index])
        else:   # on passe la série en additif pour décomposer avec la régression linéaire
            interdata = pd.DataFrame({"y": np.log(self.df[self.serie_values]).values}, index=self.df[self.time_index])
        interdata["reste"] = sm.tsa.detrend(interdata.y, order=trend_polynom_order)
        interdata["trend"] = interdata.y - interdata.reste
        if season_freq > 1:
            nb = len(interdata) // season_freq
            interserie = [interdata.iloc[[s + i*season_freq for i in range(nb)]].reste.mean() for s in np.arange(season_freq)]
            interdata["season"] = [interserie[i % season_freq] for i in np.arange(len(interdata))]
            interdata["resid"] = interdata.reste - interdata.season
            interdata["cvs"] = interdata.y - interdata.season
            df_decomp = pd.DataFrame({self.serie_values: interdata.y, "trend": interdata.trend, "seasonal": interdata.season, "resid": interdata.resid, "cvs":interdata.cvs}, index=interdata.index)
        else:
            df_decomp = pd.DataFrame({self.serie_values: interdata.y, "trend": interdata.trend, "resid": interdata.reste}, index=interdata.index)
        if self.model_type != 'add':   # on corrige les résultats par l'exponentiel car modèle multiplicatif
            df_decomp = np.exp(df_decomp)
        return df_decomp
    
    '''
    forced_color = None -> couleur par défaut ; integer -> < 0 toutes couleurs alternativement ; >= 0 toujours la même couleur i  de la liste
    '''
    def tsplot(self, df_decomp=None, title=None, figsize=(12,9), is_mono=False, forced_color=None):
        if forced_color is None:
            forced_color = 0;
        if forced_color == 'all':
            forced_color = -1;
        if df_decomp is None:
            if title is None:
                title = "Graphique de la série temporelle {}".format(self.serie_values)
            tsgraph = sfg.MyGraph(title, nblin=1, nbcol=1, is_mono=is_mono, figsize=figsize)
            if forced_color < 0:
                forced_color = 0;
            tsgraph.add_plot(self.df[self.time_index], self.df[self.serie_values], label="", color=tsgraph.liste_couleurs[forced_color%len(tsgraph.liste_couleurs)], with_grid='x')
            tsgraph.set_axe('y', label="Valeurs de {}".format(self.serie_values), tick_dash=True)
            tsgraph.set_axe('x', label="Temps")
        else:
            if title is None:
                title = "Décomposition de la série temporelle {}".format(self.serie_values)
            nblin = df_decomp.shape[1]
            tsgraph = sfg.MyGraph(title, nblin=nblin, nbcol=1, is_mono=is_mono, figsize=figsize)
            for lin in range(nblin):
                if df_decomp.columns[lin] == self.serie_values:
                    labely = "Valeurs"
                elif df_decomp.columns[lin] == "trend":
                    labely = "Tendance"
                elif df_decomp.columns[lin] == "seasonal":
                    labely = "Saison"
                elif df_decomp.columns[lin] == "resid":
                    labely = "Résidu"
                elif df_decomp.columns[lin] == "cvs":
                    labely = "CVS"
                else:
                    labely="NameError"
                if forced_color >= 0:
                    tsgraph.add_plot(df_decomp.index, df_decomp[df_decomp.columns[lin]], label="", color=tsgraph.liste_couleurs[forced_color%len(tsgraph.liste_couleurs)], with_grid='x', multi_index=lin+1)
                else:
                    tsgraph.add_plot(df_decomp.index, df_decomp[df_decomp.columns[lin]], label="", color=tsgraph.liste_couleurs[lin%len(tsgraph.liste_couleurs)], with_grid='x', multi_index=lin+1)
                tsgraph.set_axe('y', label=labely, tick_dash=True, multi_index=lin+1)
                if lin < nblin - 1:
                    tsgraph.set_axe('x', tick_labels=[], multi_index=lin+1)
                else:
                    tsgraph.set_axe('x', label="Temps", multi_index=lin+1)
        return tsgraph.fig, tsgraph.ax
    
    def plot_cvs(self, df_decomp, title=None, figsize=(12,8), is_mono=False, forced_color=None):
        if forced_color is None:
            forced_color = 0;
        if title is None:
            title = "Graphique de la série temporelle {} corrigée des variations saisonnières".format(self.serie_values)
        tsgraph = sfg.MyGraph(title, nblin=1, nbcol=1, is_mono=is_mono, figsize=figsize)
        tsgraph.add_plot(df_decomp.index, df_decomp[self.serie_values], label="Valeurs observées", color=tsgraph.liste_couleurs[forced_color%len(tsgraph.liste_couleurs)])
        tsgraph.add_plot(df_decomp.index, df_decomp.cvs, label="Valeurs CVS", color=tsgraph.liste_couleurs[(forced_color+1)%len(tsgraph.liste_couleurs)], with_grid='x', legend=True)
        tsgraph.set_axe('y', label="Valeurs de {}".format(self.serie_values), tick_dash=True)
        tsgraph.set_axe('x', label="Temps")
        return tsgraph.fig, tsgraph.ax
    
    '''
    trend_type et seasonal_type = None, 'add' ou 'mul'
    seasonal_freq = integer
    data_freq = str pour indiquer fréquence des données :
        B	business day frequency
        D	calendar day frequency
        W	weekly frequency
        M	month end frequency
        MS	month start frequency
        SM	semi-month end frequency (15th and end of month)
        SMS	semi-month start frequency (1st and 15th)
        Q	quarter end frequency
        QS	quarter start frequency
        A, Y	year end frequency
        AS, YS	year start frequency
        H	hourly frequency
        T, min	minutely frequency
        S	secondly frequency
    nb_forecast = integer -> nombre de prévisions out of sample à partir de la dernière donnée disponible
    deb_predict et fin_predict -> integer : indices en base 0 d'estimation des données de l'échantillon et/ou de prévisions out of sample
    si nb_forecast, deb_predict et fin_predict tous à None -> estimations pour toutes les valeurs de l'échantillon
    '''
    def forecast_lissage_exp(self, data_freq, trend_type=None, season_type=None, season_freq=None, nb_forecast=None, deb_predict=None, fin_predict=None):
        lissexp = sm.tsa.ExponentialSmoothing(self.df[self.serie_values].values, trend=trend_type, seasonal=season_type, seasonal_periods=season_freq).fit()
        if nb_forecast is not None:
            df_forecast = pd.DataFrame({"t": pd.date_range(self.df.iloc[-1][self.time_index], periods=nb_forecast+1, freq=data_freq)[1:]})
            df_forecast["forecast"] = lissexp.forecast(nb_forecast)
        else:
            if deb_predict is None:
                deb_predict = 0
            if fin_predict is None:
                fin_predict = len(self.df)
            if deb_predict > fin_predict:
                temp = deb_predict
                deb_predict = fin_predict
                fin_predict = temp
            df_forecast = pd.DataFrame({"t": pd.date_range(self.df.iloc[0][self.time_index], periods=fin_predict, freq=data_freq)[deb_predict:]})
            df_forecast["forecast"] = lissexp.predict(deb_predict, fin_predict-1)
        return df_forecast
    
    def plot_forecast(self, df_forecast, figsize=(12,8), is_mono=False, forced_color=None):
        if forced_color is None:
            forced_color = 0;
        tsgraph = sfg.MyGraph("Estimation par lissage exponentiel de la série temporelle {}".format(self.serie_values), nblin=1, nbcol=1, is_mono=is_mono, figsize=figsize)
        tsgraph.add_plot(self.df[self.time_index], self.df[self.serie_values], label="Valeurs observées", color=tsgraph.liste_couleurs[forced_color%len(tsgraph.liste_couleurs)])
        tsgraph.add_plot(df_forecast.t, df_forecast.forecast, label="Prévisions", color=tsgraph.liste_couleurs[(forced_color+1)%len(tsgraph.liste_couleurs)], with_grid='x', legend=True)
        tsgraph.set_axe('y', label="Valeurs de {}".format(self.serie_values), tick_dash=True)
        tsgraph.set_axe('x', label="Temps")
        return tsgraph.fig, tsgraph.ax
    
    def __get_ma2(self, m2, m1, my_serie, extrapol=False):
        if ((m1 % 2 == 0) != (m2 % 2 == 0)) | (m1 < m2):
            return pd.Series(np.nan)
        first_serie = my_serie.rolling(window=m1, center=True).mean()
        vide_m1 = m1 // 2
        if m1 % 2 == 0:
            first_serie = first_serie.shift(-1)
            vide_m1 = vide_m1 - 1
        if extrapol:
            for v1 in range(0, vide_m1, 1):
                first_serie.iloc[vide_m1-v1-1] = my_serie.iloc[0:m1-v1-1].mean()
            if m1 % 2 == 0:          # il faut réajuster les indices vides en fin de série si m1 et m2 sont pairs
                vide_m1 = vide_m1 + 1
            for v1 in range(0, vide_m1, 1):
                first_serie.iloc[-(vide_m1-v1)]  = my_serie.iloc[-(m1-v1-1):].mean()
        if m2 > 1:
            second_serie = first_serie.rolling(window=m2, center=True).mean()
            if extrapol:
                vide_m2 = m2 // 2
                for v2 in range(0, vide_m2, 1):
                    second_serie.iloc[vide_m2-v2-1] = first_serie.iloc[0:m2-v2-1].mean()
                if m1 % 2 == 0:     # il faut réajuster les indices vides en fin de série si m1 et m2 sont pairs
                    vide_m2 = vide_m2 - 1
                if vide_m2 > 0:
                    for v2 in range(0, vide_m2, 1):
                        second_serie.iloc[-(vide_m2-v2)] = first_serie.iloc[-(m2-v2-1):].mean()
        else:
            second_serie = first_serie.copy()
        return second_serie
    
    '''
        def __get_ma2(self, m2, m1, my_serie, extrapol=False):
        if ((m1 % 2 == 0) != (m2 % 2 == 0)) | (m1 < m2):
            return pd.Series(np.nan)
        first_serie = my_serie.rolling(window=m1, center=True).mean()
        vide_m1 = m1 // 2
        if m1 % 2 == 0:
            first_serie = first_serie.shift(-1)
            vide_m1 = vide_m1 - 1
        if m2 > 1:
            second_serie = first_serie.rolling(window=m2, center=True).mean()
        else:
            second_serie = first_serie.copy()
        if extrapol == False:
            return second_serie
        #extrapolation personnelle -> on recopie première moyenne mobile centrée puis on calcule une moyenne mobile avec les données dispo
        vide_m2 = m2 // 2
        if vide_m2 > 0:
            for v2 in range(0, vide_m2, 1):
                second_serie.iloc[vide_m1+vide_m2-v2-1] = first_serie.iloc[vide_m1+vide_m2-v2-1]
        for v1 in range(0, vide_m1, 1):
            second_serie.iloc[vide_m1-v1-1] = my_serie.iloc[0:m1-v1-1].mean()
        if m1 % 2 == 0:     # il faut réajuster les indices vides en fin de série si m1 et m2 sont pairs
            vide_m1 = vide_m1 + 1
            vide_m2 = vide_m2 - 1
        if vide_m2 > 0:
            for v2 in range(0, vide_m2, 1):
                second_serie.iloc[-(vide_m1+vide_m2-v2)] = first_serie.iloc[-(vide_m1+vide_m2-v2)]
        for v1 in range(0, vide_m1, 1):
            second_serie.iloc[-(vide_m1-v1)]  = my_serie.iloc[-(m1-v1-1):].mean()
        return second_serie
    '''
    
    def __weightav(self, wts):
        def g(x):
            return np.average(x, weights=wts)
        return g
    
    def __get_henderson(self, n, my_serie):
        if n == 5:
            myw = [-21, 84, 160, 84, -21]
        elif n == 7:
            myw = [-42, 42, 210, 295, 210, 42, -42]
        elif n == 9:
            myw = [-99, -24, 288, 648, 805, 648, 288, -24, -99]
        elif n == 13:
            myw = [-325, -468, 0, 1100, 2475, 3600, 4032, 3600, 2475, 1100, 0, -468, -325]
        elif n == 23:
            myw = [-17250, -44022, -63250, -58575, -19950, 54150, 156978, 275400, 392700, 491700, 557700, 580853, 
                   557700, 491700, 392700, 275400, 156978, 54150, -19950, -58575, -63250, -44022, -17250]
        else:
            myw = np.nan
        return my_serie.rolling(window=len(myw), center=True).apply(self.__weightav(myw), raw=True)
    
    def __get_x11_ma2(self, m2, m1, my_serie):
        interserie = self.__get_ma2(m2, m1, my_serie)
        if (m2 == 3) & (m1== 3):
            interserie[0] = ((5 * my_serie[2]) + (11 * my_serie[1]) + (11 * my_serie[0])) / 27
            interserie[1] = ((3 * my_serie[3]) + (7 * my_serie[2]) + (10 * my_serie[1]) + (7 * my_serie[0])) / 27
            interserie[-2] = ((3 * my_serie[-4]) + (7 * my_serie[-3]) + (10 * my_serie[-2]) + (7 * my_serie[-1])) / 27
            interserie[-1] = ((5 * my_serie[-3]) + (11 * my_serie[-2]) + (11 * my_serie[-1])) / 27
        if (m2 == 3) & (m1== 5):
            interserie[0] = ((9 * my_serie[3]) + (17 * my_serie[2]) + (17 * my_serie[1]) + (17 * my_serie[0])) / 60
            interserie[1] = ((4 * my_serie[4]) + (11 * my_serie[3]) + (15 * my_serie[2]) + (15 * my_serie[1]) + (15 * my_serie[0])) / 60
            interserie[2] = ((4 * my_serie[5]) + (8 * my_serie[4]) + (13 * my_serie[3]) + (13 * my_serie[2]) + (13 * my_serie[1]) + (9 * my_serie[0])) / 60
            interserie[-3] = ((4 * my_serie[-6]) + (8 * my_serie[-5]) + (13 * my_serie[-4]) + (13 * my_serie[-3]) + (13 * my_serie[-2]) + (9 * my_serie[-1])) / 60
            interserie[-2] = ((4 * my_serie[-5]) + (11 * my_serie[-4]) + (15 * my_serie[-3]) + (15 * my_serie[-2]) + (15 * my_serie[-1])) / 60
            interserie[-1] = ((9 * my_serie[-4]) + (17 * my_serie[-3]) + (17 * my_serie[-2]) + (17 * my_serie[-1])) / 60
        if (m2 == 3) & (m1== 9):
            interserie[0] = ((52 * my_serie[5]) + (115 * my_serie[4]) + (177 * my_serie[3]) + (202 * my_serie[2]) + (227 * my_serie[1]) + (252 * my_serie[0])) / 1026
            interserie[1] = ((29 * my_serie[6]) + (94 * my_serie[5]) + (148 * my_serie[4]) + (164 * my_serie[3]) + (181 * my_serie[2]) + (197 * my_serie[1]) + (213 * my_serie[0])) / 1026
            interserie[2] = ((33 * my_serie[7]) + (81 * my_serie[6]) + (136 * my_serie[5]) + (136 * my_serie[4]) + (147 * my_serie[3]) + (158 * my_serie[2]) + (167 * my_serie[1]) + (177 * my_serie[0])) / 1026
            interserie[3] = ((35 * my_serie[8]) + (77 * my_serie[7]) + (116 * my_serie[6]) + (120 * my_serie[5]) + (126 * my_serie[4]) + (131 * my_serie[3]) + (135 * my_serie[2]) + (141 * my_serie[1]) + (145 * my_serie[0])) / 1026
            interserie[4] = ((35 * my_serie[9]) + (75 * my_serie[8]) + (114 * my_serie[7]) + (116 * my_serie[6]) + (117 * my_serie[5]) + (119 * my_serie[4]) + (120 * my_serie[3]) + (121 * my_serie[2]) + (123 * my_serie[1]) + (86 * my_serie[0])) / 1026
            interserie[-5] = ((35 * my_serie[-10]) + (75 * my_serie[-9]) + (114 * my_serie[-8]) + (116 * my_serie[-7]) + (117 * my_serie[-6]) + (119 * my_serie[-5]) + (120 * my_serie[-4]) + (121 * my_serie[-3]) + (123 * my_serie[-2]) + (86 * my_serie[-1])) / 1026
            interserie[-4] = ((35 * my_serie[-9]) + (77 * my_serie[-8]) + (116 * my_serie[-7]) + (120 * my_serie[-6]) + (126 * my_serie[-5]) + (131 * my_serie[-4]) + (135 * my_serie[-3]) + (141 * my_serie[-2]) + (145 * my_serie[-1])) / 1026
            interserie[-3] = ((33 * my_serie[-8]) + (81 * my_serie[-7]) + (136 * my_serie[-6]) + (136 * my_serie[-5]) + (147 * my_serie[-4]) + (158 * my_serie[-3]) + (167 * my_serie[-2]) + (177 * my_serie[-1])) / 1026
            interserie[-2] = ((29 * my_serie[-7]) + (94 * my_serie[-6]) + (148 * my_serie[-5]) + (164 * my_serie[-4]) + (181 * my_serie[-3]) + (197 * my_serie[-2]) + (213 * my_serie[-1])) / 1026
            interserie[-1] = ((52 * my_serie[-6]) + (115 * my_serie[-5]) + (177 * my_serie[-4]) + (202 * my_serie[-3]) + (227 * my_serie[-2]) + (252 * my_serie[-1])) / 1026
        if (m2 == 2) & (m1== 12):
            interserie[0] = interserie[6]
            interserie[1] = interserie[6]
            interserie[2] = interserie[6]
            interserie[3] = interserie[6]
            interserie[4] = interserie[6]
            interserie[5] = interserie[6]
            interserie[-6] = interserie[-7]
            interserie[-5] = interserie[-7]
            interserie[-4] = interserie[-7]
            interserie[-3] = interserie[-7]
            interserie[-2] = interserie[-7]
            interserie[-1] = interserie[-7]
        return interserie
        
    def __get_x11_henderson(self, n, my_serie):
        interserie = self.__get_henderson(n, my_serie)
        if n == 5:
            interserie[0] = (-0.18357 * my_serie[2]) + (0.36713 * my_serie[1]) + (0.81643 * my_serie[0])
            interserie[1] = (-0.03671 * my_serie[3]) + (0.29371 * my_serie[2]) + (0.52273 * my_serie[1]) + (0.22028 * my_serie[0])
            interserie[-2] = (-0.03671 * my_serie[-4]) + (0.29371 * my_serie[-3]) + (0.52273 * my_serie[-2]) + (0.22028 * my_serie[-1])
            interserie[-1] = (-0.18357 * my_serie[-3]) + (0.36713 * my_serie[-2]) + (0.81643 * my_serie[-1])
        if n == 7:
            interserie[0] = (-0.03379 * my_serie[3]) + (0.11601 * my_serie[2]) + (0.38329 * my_serie[1]) + (0.53449 * my_serie[0])
            interserie[1] = (-0.05421 * my_serie[4]) + (0.06101 * my_serie[3]) + (0.29371 * my_serie[2]) + (0.41032 * my_serie[1]) + (0.28917 * my_serie[0])
            interserie[2] = (-0.05314 * my_serie[5]) + (0.05818 * my_serie[4]) + (0.28699 * my_serie[3]) + (0.39972 * my_serie[2]) + (0.27468 * my_serie[1]) + (0.03356 * my_serie[0])
            interserie[-3] = (-0.05314 * my_serie[-6]) + (0.05818 * my_serie[-5]) + (0.28699 * my_serie[-4]) + (0.39972 * my_serie[-3]) + (0.27468 * my_serie[-2]) + (0.03356 * my_serie[-1])
            interserie[-2] = (-0.05421 * my_serie[-5]) + (0.06101 * my_serie[-4]) + (0.29371 * my_serie[-3]) + (0.41032 * my_serie[-2]) + (0.28917 * my_serie[-1])
            interserie[-1] = (-0.03379 * my_serie[-4]) + (0.11601 * my_serie[-3]) + (0.38329 * my_serie[-2]) + (0.53449 * my_serie[-1])
        if n == 9:
            interserie[0] = (-0.15554 * my_serie[4]) + (-0.03384 * my_serie[3]) + (0.18536 * my_serie[2]) + (0.42429 * my_serie[1]) + (0.57972 * my_serie[0])
            interserie[1] = (-0.04941 * my_serie[5]) + (-0.01056 * my_serie[4]) + (0.12578 * my_serie[3]) + (0.28187 * my_serie[2]) + (0.35445 * my_serie[1]) + (0.29786 * my_serie[0])
            interserie[2] = (-0.02262 * my_serie[6]) + (-0.00021 * my_serie[5]) + (0.11969 * my_serie[4]) + (0.25933 * my_serie[3]) + (0.31547 * my_serie[2]) + (0.24244 * my_serie[1]) + (0.08590 * my_serie[0])
            interserie[3] = (-0.03082 * my_serie[7]) + (-0.00426 * my_serie[6]) + (0.11980 * my_serie[5]) + (0.26361 * my_serie[4]) + (0.32391 * my_serie[3]) + (0.25504 * my_serie[2]) + (0.10267 * my_serie[1]) + (-0.02995 * my_serie[0])
            interserie[-4] = (-0.03082 * my_serie[-8]) + (-0.00426 * my_serie[-7]) + (0.11980 * my_serie[-6]) + (0.26361 * my_serie[-5]) + (0.32391 * my_serie[-4]) + (0.25504 * my_serie[-3]) + (0.10267 * my_serie[-2]) + (-0.02995 * my_serie[-1])
            interserie[-3] = (-0.02262 * my_serie[-7]) + (-0.00021 * my_serie[-6]) + (0.11969 * my_serie[-5]) + (0.25933 * my_serie[-4]) + (0.31547 * my_serie[-3]) + (0.24244 * my_serie[-2]) + (0.08590 * my_serie[-1])
            interserie[-2] = (-0.04941 * my_serie[-6]) + (-0.01056 * my_serie[-5]) + (0.12578 * my_serie[-4]) + (0.28187 * my_serie[-3]) + (0.35445 * my_serie[-2]) + (0.29786 * my_serie[-1])
            interserie[-1] = (-0.15554 * my_serie[-5]) + (-0.03384 * my_serie[-4]) + (0.18536 * my_serie[-3]) + (0.42429 * my_serie[-2]) + (0.57972 * my_serie[-1])
        if n == 13:
            interserie[0] = (-0.09186 * my_serie[6]) + (-0.05811 * my_serie[5]) + (0.01202 * my_serie[4]) + (0.11977 * my_serie[3]) + (0.24390 * my_serie[2]) + (0.35315 * my_serie[1]) + (0.42113 * my_serie[0])
            interserie[1] = (-0.04271 * my_serie[7]) + (-0.03863 * my_serie[6]) + (0.00182 * my_serie[5]) + (0.07990 * my_serie[4]) + (0.17436 * my_serie[3]) + (0.25392 * my_serie[2]) + (0.29223 * my_serie[1]) + (0.27910 * my_serie[0])
            interserie[2] = (-0.01603 * my_serie[8]) + (-0.02487 * my_serie[7]) + (0.00267 * my_serie[6]) + (0.06784 * my_serie[5]) + (0.14939 * my_serie[4]) + (0.21605 * my_serie[3]) + (0.24144 * my_serie[2]) + (0.21540 * my_serie[1]) + (0.14810 * my_serie[0])
            interserie[3] = (-0.00813 * my_serie[9]) + (-0.02019 * my_serie[8]) + (0.00413 * my_serie[7]) + (0.06608 * my_serie[6]) + (0.14441 * my_serie[5]) + (0.20784 * my_serie[4]) + (0.23002 * my_serie[3]) + (0.20076 * my_serie[2]) + (0.13024 * my_serie[1]) + (0.04483 * my_serie[0])
            interserie[4] = (-0.01099 * my_serie[10]) + (-0.02204 * my_serie[9]) + (0.00330 * my_serie[8]) + (0.06626 * my_serie[7]) + (0.14559 * my_serie[6]) + (0.21004 * my_serie[5]) + (0.23324 * my_serie[4]) + (0.20498 * my_serie[3]) + (0.13547 * my_serie[2]) + (0.05108 * my_serie[1]) + (-0.01694 * my_serie[0])
            interserie[5] = (-0.01643 * my_serie[11]) + (-0.02577 * my_serie[10]) + (0.00127 * my_serie[9]) + (0.06594 * my_serie[8]) + (0.14698 * my_serie[7]) + (0.21314 * my_serie[6]) + (0.23803 * my_serie[5]) + (0.21149 * my_serie[4]) + (0.14368 * my_serie[3]) + (0.06099 * my_serie[2]) + (-0.00532 * my_serie[1]) + (-0.03401 * my_serie[0])
            interserie[-6] = (-0.01643 * my_serie[-12]) + (-0.02577 * my_serie[-11]) + (0.00127 * my_serie[-10]) + (0.06594 * my_serie[-9]) + (0.14698 * my_serie[-8]) + (0.21314 * my_serie[-7]) + (0.23803 * my_serie[-6]) + (0.21149 * my_serie[-5]) + (0.14368 * my_serie[-4]) + (0.06099 * my_serie[-3]) + (-0.00532 * my_serie[-2]) + (-0.03401 * my_serie[-1])
            interserie[-5] = (-0.01099 * my_serie[-11]) + (-0.02204 * my_serie[-10]) + (0.00330 * my_serie[-9]) + (0.06626 * my_serie[-8]) + (0.14559 * my_serie[-7]) + (0.21004 * my_serie[-6]) + (0.23324 * my_serie[-5]) + (0.20498 * my_serie[-4]) + (0.13547 * my_serie[-3]) + (0.05108 * my_serie[-2]) + (-0.01694 * my_serie[-1])
            interserie[-4] = (-0.00813 * my_serie[-10]) + (-0.02019 * my_serie[-9]) + (0.00413 * my_serie[-8]) + (0.06608 * my_serie[-7]) + (0.14441 * my_serie[-6]) + (0.20784 * my_serie[-5]) + (0.23002 * my_serie[-4]) + (0.20076 * my_serie[-3]) + (0.13024 * my_serie[-2]) + (0.04483 * my_serie[-1])
            interserie[-3] = (-0.01603 * my_serie[-9]) + (-0.02487 * my_serie[-8]) + (0.00267 * my_serie[-7]) + (0.06784 * my_serie[-6]) + (0.14939 * my_serie[-5]) + (0.21605 * my_serie[-4]) + (0.24144 * my_serie[-3]) + (0.21540 * my_serie[-2]) + (0.14810 * my_serie[-1])
            interserie[-2] = (-0.04271 * my_serie[-8]) + (-0.03863 * my_serie[-7]) + (0.00182 * my_serie[-6]) + (0.07990 * my_serie[-5]) + (0.17436 * my_serie[-4]) + (0.25392 * my_serie[-3]) + (0.29223 * my_serie[-2]) + (0.27910 * my_serie[-1])
            interserie[-1] = (-0.09186 * my_serie[-7]) + (-0.05811 * my_serie[-6]) + (0.01202 * my_serie[-5]) + (0.11977 * my_serie[-4]) + (0.24390 * my_serie[-3]) + (0.35315 * my_serie[-2]) + (0.42113 * my_serie[-1])
        if n == 23:
            interserie[0] = (-0.07689 * my_serie[11]) + (-0.06385 * my_serie[10]) + (-0.04893 * my_serie[9]) + (-0.02808 * my_serie[8]) + (0.00119 * my_serie[7]) + (0.03925 * my_serie[6]) + (0.08444 * my_serie[5]) + (0.13350 * my_serie[4]) + (0.18228 * my_serie[3]) + (0.22652 * my_serie[2]) + (0.26258 * my_serie[1]) + (0.28801 * my_serie[0])
            interserie[1] = (-0.04520 * my_serie[12]) + (-0.04130 * my_serie[11]) + (-0.03554 * my_serie[10]) + (-0.02385 * my_serie[9]) + (-0.00373 * my_serie[8]) + (0.02518 * my_serie[7]) + (0.06121 * my_serie[6]) + (0.10112 * my_serie[5]) + (0.14074 * my_serie[4]) + (0.17583 * my_serie[3]) + (0.20273 * my_serie[2]) + (0.21901 * my_serie[1]) + (0.22380 * my_serie[0])
            interserie[2] = (-0.02293 * my_serie[13]) + (-0.02486 * my_serie[12]) + (-0.02491 * my_serie[11]) + (-0.01904 * my_serie[10]) + (-0.00475 * my_serie[9]) + (0.01834 * my_serie[8]) + (0.04856 * my_serie[7]) + (0.08264 * my_serie[6]) + (0.11644 * my_serie[5]) + (0.14571 * my_serie[4]) + (0.16679 * my_serie[3]) + (0.17724 * my_serie[2]) + (0.17622 * my_serie[1]) + (0.16456 * my_serie[0])
            interserie[3] = (-0.00861 * my_serie[14]) + (-0.01396 * my_serie[13]) + (-0.01744 * my_serie[12]) + (-0.01500 * my_serie[11]) + (-0.00413 * my_serie[10]) + (0.01554 * my_serie[9]) + (0.04233 * my_serie[8]) + (0.07299 * my_serie[7]) + (0.10337 * my_serie[6]) + (0.12921 * my_serie[5]) + (0.14687 * my_serie[4]) + (0.15390 * my_serie[3]) + (0.14945 * my_serie[2]) + (0.13437 * my_serie[1]) + (0.11111 * my_serie[0])
            interserie[4] = (-0.00065 * my_serie[15]) + (-0.00776 * my_serie[14]) + (-0.01300 * my_serie[13]) + (-0.01230 * my_serie[12]) + (-0.00319 * my_serie[11]) + (0.01472 * my_serie[10]) + (0.03976 * my_serie[9]) + (0.06866 * my_serie[8]) + (0.09729 * my_serie[7]) + (0.12137 * my_serie[6]) + (0.13728 * my_serie[5]) + (0.14255 * my_serie[4]) + (0.13634 * my_serie[3]) + (0.11951 * my_serie[2]) + (0.09449 * my_serie[1]) + (0.06493 * my_serie[0])
            interserie[5] = (0.00258 * my_serie[16]) + (-0.00519 * my_serie[15]) + (-0.01109 * my_serie[14]) + (-0.01106 * my_serie[13]) + (-0.00261 * my_serie[12]) + (0.01464 * my_serie[11]) + (0.03902 * my_serie[10]) + (0.06726 * my_serie[9]) + (0.09522 * my_serie[8]) + (0.11865 * my_serie[7]) + (0.13389 * my_serie[6]) + (0.13850 * my_serie[5]) + (0.13163 * my_serie[4]) + (0.11413 * my_serie[3]) + (0.08845 * my_serie[2]) + (0.05823 * my_serie[1]) + (0.02773 * my_serie[0])
            interserie[6] = (0.00268 * my_serie[17]) + (-0.00511 * my_serie[16]) + (-0.01103 * my_serie[15]) + (-0.01101 * my_serie[14]) + (-0.00258 * my_serie[13]) + (0.01465 * my_serie[12]) + (0.03900 * my_serie[11]) + (0.06723 * my_serie[10]) + (0.09517 * my_serie[9]) + (0.11858 * my_serie[8]) + (0.13380 * my_serie[7]) + (0.13839 * my_serie[6]) + (0.13150 * my_serie[5]) + (0.11399 * my_serie[4]) + (0.08829 * my_serie[3]) + (0.05805 * my_serie[2]) + (0.02753 * my_serie[1]) + (0.00088 * my_serie[0])
            interserie[7] = (0.00108 * my_serie[18]) + (-0.00642 * my_serie[17]) + (-0.01205 * my_serie[16]) + (-0.01175 * my_serie[15]) + (-0.00303 * my_serie[14]) + (0.01448 * my_serie[13]) + (0.03913 * my_serie[12]) + (0.06764 * my_serie[11]) + (0.09587 * my_serie[10]) + (0.11956 * my_serie[9]) + (0.13507 * my_serie[8]) + (0.13995 * my_serie[7]) + (0.13334 * my_serie[6]) + (0.11611 * my_serie[5]) + (0.09070 * my_serie[4]) + (0.06075 * my_serie[3]) + (0.03052 * my_serie[2]) + (0.00415 * my_serie[1]) + (-0.01509 * my_serie[0])
            interserie[8] = (-0.00103 * my_serie[19]) + (-0.00817 * my_serie[18]) + (-0.01344 * my_serie[17]) + (-0.01279 * my_serie[16]) + (-0.00372 * my_serie[15]) + (0.01416 * my_serie[14]) + (0.03916 * my_serie[13]) + (0.06802 * my_serie[12]) + (0.09661 * my_serie[11]) + (0.12066 * my_serie[10]) + (0.13652 * my_serie[9]) + (0.14176 * my_serie[8]) + (0.13551 * my_serie[7]) + (0.11864 * my_serie[6]) + (0.09358 * my_serie[5]) + (0.06398 * my_serie[4]) + (0.03411 * my_serie[3]) + (0.00810 * my_serie[2]) + (-0.01078 * my_serie[1]) + (-0.02087 * my_serie[0])
            interserie[9] = (-0.00282 * my_serie[20]) + (-0.00968 * my_serie[19]) + (-0.01467 * my_serie[18]) + (-0.01372 * my_serie[17]) + (-0.00436 * my_serie[16]) + (0.01380 * my_serie[15]) + (0.03908 * my_serie[14]) + (0.06823 * my_serie[13]) + (0.09711 * my_serie[12]) + (0.12144 * my_serie[11]) + (0.13759 * my_serie[10]) + (0.14312 * my_serie[9]) + (0.13716 * my_serie[8]) + (0.12057 * my_serie[7]) + (0.09580 * my_serie[6]) + (0.06649 * my_serie[5]) + (0.03690 * my_serie[4]) + (0.01118 * my_serie[3]) + (-0.00742 * my_serie[2]) + (-0.01721 * my_serie[1]) + (-0.01859 * my_serie[0])
            interserie[10] = (-0.00390 * my_serie[21]) + (-0.01059 * my_serie[20]) + (-0.01542 * my_serie[19]) + (-0.01431 * my_serie[18]) + (-0.00479 * my_serie[17]) + (0.01354 * my_serie[16]) + (0.03898 * my_serie[15]) + (0.06830 * my_serie[14]) + (0.09734 * my_serie[13]) + (0.12184 * my_serie[12]) + (0.13815 * my_serie[11]) + (0.14384 * my_serie[10]) + (0.13804 * my_serie[9]) + (0.12162 * my_serie[8]) + (0.09701 * my_serie[7]) + (0.06786 * my_serie[6]) + (0.03844 * my_serie[5]) + (0.01288 * my_serie[4]) + (-0.00555 * my_serie[3]) + (-0.01519 * my_serie[2]) + (-0.01640 * my_serie[1]) + (-0.01169 * my_serie[0])
            interserie[-11] = (-0.00390 * my_serie[-22]) + (-0.01059 * my_serie[-21]) + (-0.01542 * my_serie[-20]) + (-0.01431 * my_serie[-19]) + (-0.00479 * my_serie[-18]) + (0.01354 * my_serie[-17]) + (0.03898 * my_serie[-16]) + (0.06830 * my_serie[-15]) + (0.09734 * my_serie[-14]) + (0.12184 * my_serie[-13]) + (0.13815 * my_serie[-12]) + (0.14384 * my_serie[-11]) + (0.13804 * my_serie[-10]) + (0.12162 * my_serie[-9]) + (0.09701 * my_serie[-8]) + (0.06786 * my_serie[-7]) + (0.03844 * my_serie[-6]) + (0.01288 * my_serie[-5]) + (-0.00555 * my_serie[-4]) + (-0.01519 * my_serie[-3]) + (-0.01640 * my_serie[-2]) + (-0.01169 * my_serie[-1])
            interserie[-10] = (-0.00282 * my_serie[-21]) + (-0.00968 * my_serie[-20]) + (-0.01467 * my_serie[-19]) + (-0.01372 * my_serie[-18]) + (-0.00436 * my_serie[-17]) + (0.01380 * my_serie[-16]) + (0.03908 * my_serie[-15]) + (0.06823 * my_serie[-14]) + (0.09711 * my_serie[-13]) + (0.12144 * my_serie[-12]) + (0.13759 * my_serie[-11]) + (0.14312 * my_serie[-10]) + (0.13716 * my_serie[-9]) + (0.12057 * my_serie[-8]) + (0.09580 * my_serie[-7]) + (0.06649 * my_serie[-6]) + (0.03690 * my_serie[-5]) + (0.01118 * my_serie[-4]) + (-0.00742 * my_serie[-3]) + (-0.01721 * my_serie[-2]) + (-0.01859 * my_serie[-1])
            interserie[-9] = (-0.00103 * my_serie[-20]) + (-0.00817 * my_serie[-19]) + (-0.01344 * my_serie[-18]) + (-0.01279 * my_serie[-17]) + (-0.00372 * my_serie[-16]) + (0.01416 * my_serie[-15]) + (0.03916 * my_serie[-14]) + (0.06802 * my_serie[-13]) + (0.09661 * my_serie[-12]) + (0.12066 * my_serie[-11]) + (0.13652 * my_serie[-10]) + (0.14176 * my_serie[-9]) + (0.13551 * my_serie[-8]) + (0.11864 * my_serie[-7]) + (0.09358 * my_serie[-6]) + (0.06398 * my_serie[-5]) + (0.03411 * my_serie[-4]) + (0.00810 * my_serie[-3]) + (-0.01078 * my_serie[-2]) + (-0.02087 * my_serie[-1])
            interserie[-8] = (0.00108 * my_serie[-19]) + (-0.00642 * my_serie[-18]) + (-0.01205 * my_serie[-17]) + (-0.01175 * my_serie[-16]) + (-0.00303 * my_serie[-15]) + (0.01448 * my_serie[-14]) + (0.03913 * my_serie[-13]) + (0.06764 * my_serie[-12]) + (0.09587 * my_serie[-11]) + (0.11956 * my_serie[-10]) + (0.13507 * my_serie[-9]) + (0.13995 * my_serie[-8]) + (0.13334 * my_serie[-7]) + (0.11611 * my_serie[-6]) + (0.09070 * my_serie[-5]) + (0.06075 * my_serie[-4]) + (0.03052 * my_serie[-3]) + (0.00415 * my_serie[-2]) + (-0.01509 * my_serie[-1])
            interserie[-7] = (0.00268 * my_serie[-18]) + (-0.00511 * my_serie[-17]) + (-0.01103 * my_serie[-16]) + (-0.01101 * my_serie[-15]) + (-0.00258 * my_serie[-14]) + (0.01465 * my_serie[-13]) + (0.03900 * my_serie[-12]) + (0.06723 * my_serie[-11]) + (0.09517 * my_serie[-10]) + (0.11858 * my_serie[-9]) + (0.13380 * my_serie[-8]) + (0.13839 * my_serie[-7]) + (0.13150 * my_serie[-6]) + (0.11399 * my_serie[-5]) + (0.08829 * my_serie[-4]) + (0.05805 * my_serie[-3]) + (0.02753 * my_serie[-2]) + (0.00088 * my_serie[-1])
            interserie[-6] = (0.00258 * my_serie[-17]) + (-0.00519 * my_serie[-16]) + (-0.01109 * my_serie[-15]) + (-0.01106 * my_serie[-14]) + (-0.00261 * my_serie[-13]) + (0.01464 * my_serie[-12]) + (0.03902 * my_serie[-11]) + (0.06726 * my_serie[-10]) + (0.09522 * my_serie[-9]) + (0.11865 * my_serie[-8]) + (0.13389 * my_serie[-7]) + (0.13850 * my_serie[-6]) + (0.13163 * my_serie[-5]) + (0.11413 * my_serie[-4]) + (0.08845 * my_serie[-3]) + (0.05823 * my_serie[-2]) + (0.02773 * my_serie[-1])
            interserie[-5] = (-0.00065 * my_serie[-16]) + (-0.00776 * my_serie[-15]) + (-0.01300 * my_serie[-14]) + (-0.01230 * my_serie[-13]) + (-0.00319 * my_serie[-12]) + (0.01472 * my_serie[-11]) + (0.03976 * my_serie[-10]) + (0.06866 * my_serie[-9]) + (0.09729 * my_serie[-8]) + (0.12137 * my_serie[-7]) + (0.13728 * my_serie[-6]) + (0.14255 * my_serie[-5]) + (0.13634 * my_serie[-4]) + (0.11951 * my_serie[-3]) + (0.09449 * my_serie[-2]) + (0.06493 * my_serie[-1])
            interserie[-4] = (-0.00861 * my_serie[-15]) + (-0.01396 * my_serie[-14]) + (-0.01744 * my_serie[-13]) + (-0.01500 * my_serie[-12]) + (-0.00413 * my_serie[-11]) + (0.01554 * my_serie[-10]) + (0.04233 * my_serie[-9]) + (0.07299 * my_serie[-8]) + (0.10337 * my_serie[-7]) + (0.12921 * my_serie[-6]) + (0.14687 * my_serie[-5]) + (0.15390 * my_serie[-4]) + (0.14945 * my_serie[-3]) + (0.13437 * my_serie[-2]) + (0.11111 * my_serie[-1])
            interserie[-3] = (-0.02293 * my_serie[-14]) + (-0.02486 * my_serie[-13]) + (-0.02491 * my_serie[-12]) + (-0.01904 * my_serie[-11]) + (-0.00475 * my_serie[-10]) + (0.01834 * my_serie[-9]) + (0.04856 * my_serie[-8]) + (0.08264 * my_serie[-7]) + (0.11644 * my_serie[-6]) + (0.14571 * my_serie[-5]) + (0.16679 * my_serie[-4]) + (0.17724 * my_serie[-3]) + (0.17622 * my_serie[-2]) + (0.16456 * my_serie[-1])
            interserie[-2] = (-0.04520 * my_serie[-13]) + (-0.04130 * my_serie[-12]) + (-0.03554 * my_serie[-11]) + (-0.02385 * my_serie[-10]) + (-0.00373 * my_serie[-9]) + (0.02518 * my_serie[-8]) + (0.06121 * my_serie[-7]) + (0.10112 * my_serie[-6]) + (0.14074 * my_serie[-5]) + (0.17583 * my_serie[-4]) + (0.20273 * my_serie[-3]) + (0.21901 * my_serie[-2]) + (0.22380 * my_serie[-1])
            interserie[-1] = (-0.07689 * my_serie[-12]) + (-0.06385 * my_serie[-11]) + (-0.04893 * my_serie[-10]) + (-0.02808 * my_serie[-9]) + (0.00119 * my_serie[-8]) + (0.03925 * my_serie[-7]) + (0.08444 * my_serie[-6]) + (0.13350 * my_serie[-5]) + (0.18228 * my_serie[-4]) + (0.22652 * my_serie[-3]) + (0.26258 * my_serie[-2]) + (0.28801 * my_serie[-1])
        return interserie


class Sarimax:
    
    def __init__(self, data, time_index, serie_values, exogenes=None):
        try:
            if data[[time_index]].select_dtypes(include='datetime').shape[1] == 0:
                raise TypeError("Type Error : time_index must be of datetime type.")
        except (TypeError, ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()))
        else:
            self.df = data.copy()
            self.time_index = time_index
            self.serie_values = serie_values
            self.exogenes = exogenes
            self.n_observations = len(data)
            self.core_diff = 0
            self.core_ar = 0
            self.core_ma = 0
            self.season_diff = 0
            self.season_ar = 0
            self.season_ma = 0
            self.season_freq = 0
            self.model = None
            self.residuals = None
            self.calcul_sarimax()
            
    def set_arima_coef(self, ar=0, diff=0, ma=0, with_calculation=True):
        self.core_diff = diff
        self.core_ar = ar
        self.core_ma = ma
        if with_calculation:
            self.calcul_sarimax()
        
    def set_season_coef(self, ar=0, diff=0, ma=0, freq=0, with_calculation=True):
        self.season_diff = diff
        self.season_ar = ar
        self.season_ma = ma
        self.season_freq = freq
        if with_calculation:
            self.calcul_sarimax()

    def calcul_sarimax(self, method='lbfgs', maxiter=50, data_split=None):
        if data_split is None:
            my_serie = self.df[self.serie_values]
        else:
            my_serie = self.df[self.serie_values][:data_split]
        if self.exogenes is None:
            if self.season_freq < 0:
                self.model = sm.tsa.statespace.SARIMAX(my_serie, order=(self.core_ar, self.core_diff, self.core_ma)).fit(method=method, maxiter=maxiter)
            else:
                self.model = sm.tsa.statespace.SARIMAX(my_serie, order=(self.core_ar, self.core_diff, self.core_ma), seasonal_order=(self.season_ar, self.season_diff, self.season_ma, self.season_freq)).fit(method=method, maxiter=maxiter)
        else:
            if self.season_freq < 0:
                self.model = sm.tsa.statespace.SARIMAX(my_serie, exog=self.df[self.exogenes], order=(self.core_ar, self.core_diff, self.core_ma)).fit(method=method, maxiter=maxiter)
            else:
                self.model = sm.tsa.statespace.SARIMAX(my_serie, exog=self.df[self.exogenes], order=(self.core_ar, self.core_diff, self.core_ma), seasonal_order=(self.season_ar, self.season_diff, self.season_ma, self.season_freq)).fit(method=method, maxiter=maxiter)
        self.residuals = self.model.resid
    
    def check_autocorrelations(self, max_diff=2, lags=40, season_freq=0):
        df_diff = self.df[[self.serie_values]].copy()
        if season_freq <= 0:
            title = "Vérification de la stationnarité et de la saisonnalité"
        else:
            title = "Vérification de la stationnarité et de la saisonnalité\nFréquence de saisonnalité = {}".format(season_freq)
        if max_diff == 0:
            hauteur_fig = 6
        else:
            hauteur_fig = (max_diff+1)*3
        checkgraph = sfg.MyGraph(title, nblin=max_diff+1, nbcol=2, figsize=(12,hauteur_fig))
        checkgraph.graph_sarimax_autocorrelations(df_diff, max_diff=max_diff, lags=lags, season_freq=season_freq)
        return checkgraph.fig, checkgraph.ax
        
    '''
    serie = 'raw', 'arima', 'season'
    '''
    def check_rawserie(self, centered=False):
        title = "Analyse de la série temporelle {}".format(self.serie_values)
        return self.plot_sarimax(self.df[self.serie_values], self.df[self.time_index], title, centered=centered)
    
    def check_arima(self, core_diff=None, centered=False):
        if core_diff is None:
            core_diff = self.core_diff
        df_diff = self.df[[self.serie_values]].copy()
        df_diff["arima"] = np.append(np.zeros(core_diff), np.diff(df_diff[df_diff.columns[0]].values, n=core_diff))
        title = "Analyse de la série temporelle {}\nARIMA - Différence = {}".format(self.serie_values, core_diff)
        return self.plot_sarimax(df_diff.iloc[core_diff:].arima, self.df[self.time_index][core_diff:], title, centered=centered)
    
    def check_season(self, season_freq=None, season_diff=None, centered=True):
        if season_freq is None:
            season_freq = self.season_freq
        if season_diff is None:
            season_diff = self.season_diff
        df_diff = self.df[[self.serie_values]].copy()
        df_diff["interdiff"] = np.append(np.zeros(season_diff), np.diff(df_diff[df_diff.columns[0]].values, n=season_diff))
        df_diff["sarima"] = df_diff.interdiff.diff(season_freq)
        title = "Analyse de la série temporelle {}\nSaisonnalité - Fréquence = {} - Différence = {}".format(self.serie_values, season_freq, season_diff)
        return self.plot_sarimax(df_diff.iloc[season_freq+season_diff:].sarima, self.df[self.time_index][season_freq+season_diff:], title, centered=centered)
    
    def check_residuals(self, decal_deb=None, train_size=None):
        if decal_deb is None:
            if self.season_freq < 0:
                decal_deb = self.core_diff
            else:
                decal_deb = self.core_diff + self.season_freq
            decal_deb = max(decal_deb, 1)
        if self.season_freq < 0:
            txt_model = "ARIMA={},{},{} - Pas de saisonnalité".format(self.core_ar, self.core_diff, self.core_ma)
        else:
            txt_model = "ARIMA={},{},{} - Saisonnalité={} - Saisonnalité ARIMA={},{},{}".format(self.core_ar, self.core_diff, self.core_ma, self.season_freq, self.season_ar, self.season_diff, self.season_ma)
        if train_size is None:
            return self.plot_sarimax(self.residuals[decal_deb:], self.df[self.time_index][decal_deb:], "Analyse des résidus du modèle SARIMA\n{}".format(txt_model), centered=True)
        return self.plot_sarimax(self.residuals[decal_deb:train_size], self.df[self.time_index][decal_deb:train_size], "Analyse des résidus du modèle SARIMA\n{}".format(txt_model), centered=True)
    
    def plot_sarimax(self, my_serie, my_index, title, centered=False):
        checkgraph = sfg.MyGraph(title, nblin=3, nbcol=2, figsize=(12,9), gridspec=True)
        checkgraph.graph_sarimax_analysis(my_serie, my_index, centered=centered)
        return checkgraph.fig, checkgraph.ax
    
    def check_results(self, alpha=0.05, train_size=None):
        if self.season_freq < 0:
            decal_deb = self.core_diff
        else:
            decal_deb = self.core_diff + self.season_freq
        lags = [None,6,12,18,24,30,36]
        verif_params = True
        for param in range(len(self.model.pvalues)):
            if (self.model.pvalues.index[param] != 'sigma2') & (self.model.pvalues[param] > alpha):
                verif_params = False
        print(self.model.summary())
        if verif_params:
            print("\033[1m"+"Les paramètres du modèle sont significatifs au seuil {:.0%}".format(alpha)+"\033[0m")
        else:
            print("\033[1m"+"Modèle rejeté : les paramètres du modèle ne sont pas significatifs au seuil {:.0%}".format(alpha)+"\033[0m")
        print("")    
        print("\033[1m"+"Test de blancheur des résidus au seuil {:.0%}".format(alpha)+"\033[0m")
        print("   H0: les résidus suivent un bruit blanc")
        print("   H1: les résidus ne suivent pas un bruit blanc")
        df_blancheur = pd.DataFrame({"retard":["all",6,12,18,24,30,36]})
        df_blancheur["p_value"] = [np.round(self.model.test_serial_correlation('ljungbox', lags=l)[0,1,-1], 4) for l in lags]
        df_blancheur["Accept H0"] = df_blancheur.p_value >= alpha
        print(df_blancheur.set_index("retard").T)
        print("")
        print("\033[1m"+"Test de normalité des résidus au seuil {:.0%}".format(alpha)+"\033[0m")
        print("   H0: les résidus suivent une loi normale")
        print("   H1: les résidus ne suivent pas une loi normale")
        df_normalite = pd.DataFrame(index=["Jarque-Bera","Shapiro"], columns=["p_value","Accept H0"])
        df_normalite.iloc[0,0] = np.round(self.model.test_normality('jarquebera')[0,1], 4)
        df_normalite.iloc[1,0] = np.round(st.shapiro(self.residuals[decal_deb:])[1], 4)
        df_normalite["Accept H0"] = df_normalite.p_value >= alpha
        print(df_normalite.T)
        print("")
        print("\033[1m"+"Test d'homoscédasticité des résidus au seuil {:.0%}".format(alpha)+"\033[0m")
        df_hs = pd.DataFrame(index=["H0","H1","p_value","Accept H0"], columns=["Two-sided","Increasing","Decreasing"])
        df_hs.iloc[0] = ["Constant variance","Not increasing variance","Not decreasing variance"]
        df_hs.iloc[1] = ["Variance changes","Increasing variance","Decreasing variance"]
        df_hs.iloc[2] = [np.round(self.model.test_heteroskedasticity('breakvar',alternative=alt)[0,1], 4) for alt in df_hs.columns]
        df_hs.iloc[3] = [df_hs.iloc[2,col] >= alpha for col in range(len(df_hs.columns))]
        print(df_hs)
        print("")
        print("\033[1m"+"Primo évaluation du modèle"+"\033[0m")
        df_eval = self.calcul_eval(max_loc=train_size)
        print(df_eval)
        
    def evaluate_model(self, data_freq, test_percent=0.2, return_forecast=False):
        train_size = np.int_(np.round(len(self.df) * (1.0 - test_percent)))
        test_size = len(self.df) - train_size
        print("Longueur de la période d'entraînement : {}".format(train_size))
        print("Longueur de la période de test : {}".format(test_size))
        print()
        print("\033[1m"+"\033[94m"+"Vérification du modèle sur la période d'entraînement"+"\033[0m"+"\033[0m")
        self.calcul_sarimax(data_split=train_size)
        self.check_results(train_size=train_size)
        print("")
        print("")
        print("\033[1m"+"\033[94m"+"Vérification des estimations sur la période de test"+"\033[0m"+"\033[0m")
        df_forecast = self.calcul_forecast(data_freq=data_freq, deb_predict=train_size)
        df_eval = self.calcul_eval(df_forecast)
        print(df_eval)
        print("")
        self.plot_forecast(df_forecast, title="Vérification des estimations du modèle sur la période de test", only_forecast=True)
        if return_forecast:
            return df_forecast
        
    def calcul_eval(self, df_forecast=None, max_loc=None):
        if df_forecast is None:
            if max_loc is None:
                max_loc = len(self.df)
            df_forecast = pd.DataFrame({"t": self.df.iloc[:max_loc][self.time_index], "observed": self.df.iloc[:max_loc][self.serie_values]})
            predict = self.model.get_prediction()
            df_forecast["forecast"] = predict.predicted_mean.values
            df_forecast["ci_min"] = predict.conf_int().iloc[:,0].values
            df_forecast["ci_max"] = predict.conf_int().iloc[:,1].values
        rmse = np.sqrt(np.mean((df_forecast.observed - df_forecast.forecast)**2))
        mape = np.mean(np.abs(1 - df_forecast.forecast / df_forecast.observed))
        corr = np.corrcoef(df_forecast.forecast, df_forecast.observed)[0,1]
        mins = np.amin(np.hstack([df_forecast.forecast[:,None], df_forecast.observed[:,None]]), axis=1)
        maxs = np.amax(np.hstack([df_forecast.forecast[:,None], df_forecast.observed[:,None]]), axis=1)
        minmax = 1 - np.mean(mins/maxs)
        return pd.DataFrame({"RMSE": rmse, "MAPE":mape, "Corrélation":corr, "Min-Max Error":minmax}, index=["Mesures"])
        
    def calcul_forecast(self, data_freq, nb_forecast=None, deb_predict=None, fin_predict=None, np_op=None):
        if nb_forecast is not None:
            df_forecast = pd.DataFrame({"t": pd.date_range(self.df.iloc[-1][self.time_index], periods=nb_forecast+1, freq=data_freq)[1:]})
            df_forecast["observed"] = np.nan
            predict = self.model.get_forecast(nb_forecast)
        else:
            if deb_predict is None:
                deb_predict = 0
            if fin_predict is None:
                fin_predict = len(self.df)
            if deb_predict > fin_predict:
                temp = deb_predict
                deb_predict = fin_predict
                fin_predict = temp
            df_forecast = pd.DataFrame({"t": pd.date_range(self.df.iloc[0][self.time_index], periods=fin_predict, freq=data_freq)[deb_predict:]})
            df_forecast["observed"] = [np.nan if i>=len(self.df) else self.df.iloc[i][self.serie_values] for i in range(deb_predict, fin_predict)]
            predict = self.model.get_prediction(deb_predict, fin_predict-1)
        df_forecast["forecast"] = predict.predicted_mean.values
        df_forecast["ci_min"] = predict.conf_int().iloc[:,0].values
        df_forecast["ci_max"] = predict.conf_int().iloc[:,1].values
        if np_op is not None:
            if np_op == 'exp':
                df_forecast["observed"] = np.exp(df_forecast.observed)
                df_forecast["forecast"] = np.exp(df_forecast.forecast)
                df_forecast["ci_min"] = np.exp(df_forecast.ci_min)
                df_forecast["ci_max"] = np.exp(df_forecast.ci_max)
        return df_forecast
        
    '''
    data_freq : D, M ou Y pour afficher la date avec le jour, le mois ou l'année 
    '''
    def plot_forecast(self, df_forecast, data_freq=None, title=None, figsize=(12,8), is_mono=False, forced_color=None, only_forecast=False, np_op=None, name_serie=None, ts_graph=None, ax_index=None):
        if name_serie is None:
            name_serie = self.serie_values
        if title is None:
            title = "Estimation de la série temporelle {}".format(name_serie)
        if forced_color is None:
            forced_color = 0;
        if data_freq == "D":
            forecast_index = df_forecast.t.dt.strftime("%d %b %Y")
            data_index = self.df[self.time_index].dt.strftime("%d %b %Y")
        elif data_freq == "M":
            forecast_index = df_forecast.t.dt.strftime("%b %Y")
            data_index = self.df[self.time_index].dt.strftime("%b %Y")
        elif data_freq == "Y":
            forecast_index = df_forecast.t.dt.strftime("%Y")
            data_index = self.df[self.time_index].dt.strftime("%Y")
        else:
            forecast_index = df_forecast.t
            data_index = self.df[self.time_index]
        if ts_graph is None:
            tsgraph = sfg.MyGraph(title, nblin=1, nbcol=1, is_mono=is_mono, figsize=figsize)
            multi_index = 1
        else:
            tsgraph = ts_graph
            if ax_index is None:
                multi_index = 1
            else:
                multi_index = ax_index + 1
        
        tsgraph.add_plot_date(forecast_index, df_forecast.forecast, label="Prévisions", color=tsgraph.liste_couleurs[(forced_color+1)%len(tsgraph.liste_couleurs)], multi_index=multi_index)
        if only_forecast:
            if np.isnan(df_forecast.iloc[0]["observed"]) == False:
                tsgraph.add_plot_date(forecast_index, df_forecast.observed, label="Valeurs observées", color=tsgraph.liste_couleurs[forced_color%len(tsgraph.liste_couleurs)], legend=True, multi_index=multi_index)
        else:
            if np_op is None:
                y_values = self.df[self.serie_values]
            else:
                if np_op == 'exp':
                    y_values = np.exp(self.df[self.serie_values])
            tsgraph.add_plot_date(data_index, y_values, label="Valeurs observées", color=tsgraph.liste_couleurs[forced_color%len(tsgraph.liste_couleurs)], legend=True, multi_index=multi_index)
        tsgraph.add_area(forecast_index, df_forecast.ci_min, df_forecast.ci_max, color=tsgraph.liste_couleurs[(forced_color+1)%len(tsgraph.liste_couleurs)], with_grid='x', multi_index=multi_index)
        tsgraph.set_axe('y', label="Valeurs de {}".format(name_serie), tick_dash=True, multi_index=multi_index)
        tsgraph.set_axe('x', label="Temps", multi_index=multi_index)
        if ts_graph is None:
            return tsgraph.fig, tsgraph.ax
        return tsgraph
    
    
    '''
    models that can't be calculated are ignored, as well as models with non significants parameters (seuil = alpha) and models with convergence warnings
    arrays for p, d and q indicate values to test
    array for season indicate season freq value to consider. 
        if several values, each freq value is tested (0 is permitted - this leads to a double ARIMA model)
        if si, sj and sk == 0 -> ARIMA model
    '''
    def get_sarima_best_models(self, p=[0,1,2], d=[0,1,2], q=[0,1,2], season_freq=[0], alpha=0.05, method='lbfgs', maxiter=50):
        best_aic = [np.inf, np.inf, np.inf]
        best_order = [None, None, None]
        best_model = [None, None, None]
        for i in p:
            for j in d:
                for k in q:
                    for l in season_freq:
                        for si in p:
                            for sj in d:
                                for sk in q:
                                    try:
                                        with warnings.catch_warnings():
                                            warnings.filterwarnings("ignore")
                                            if self.exogenes is None:
                                                temp_model = sm.tsa.statespace.SARIMAX(self.df[self.serie_values], order=(i,j,k), seasonal_order=(si,sj,sk,l)).fit(method=method, maxiter=maxiter)
                                            else:
                                                temp_model = sm.tsa.statespace.SARIMAX(self.df[self.serie_values], order=(i,j,k), exog=self.df[self.exogenes], seasonal_order=(si,sj,sk,l)).fit(method=method, maxiter=maxiter)
                                        verif_params = True
                                        for param in range(len(temp_model.pvalues)):
                                            if (temp_model.pvalues.index[param] != 'sigma2') & (temp_model.pvalues[param] > alpha):
                                                verif_params = False
                                        if verif_params:
                                            temp_aic = temp_model.aic
                                            temp_rank = 3
                                            for verif_rank in np.arange(2,-1,-1):
                                                if temp_aic < best_aic[verif_rank]:
                                                    temp_rank = verif_rank
                                                else:
                                                    verif_rank = -1
                                            if temp_rank < 3:
                                                for verif_rank in np.arange(2, temp_rank, -1):
                                                    best_aic[verif_rank] = best_aic[verif_rank-1]
                                                    best_model[verif_rank] = best_model[verif_rank-1]
                                                    best_order[verif_rank] = best_order[verif_rank-1]
                                                best_aic[temp_rank] = temp_aic
                                                best_model[temp_rank] = temp_model
                                                best_order[temp_rank] = (i,j,k,si,sj,sk,l)
                                    except:
                                        continue
        print("Les trois meilleurs modèles sont :")
        for temp_rank in np.arange(len(best_aic)):
            print("AIC = {:.2f}  |  order ARIMA= {}  |  order Season= {}".format(best_aic[temp_rank], best_order[temp_rank][:3], best_order[temp_rank][3:]))
        return best_order[0]
    
    '''
    models that can't be calculated are ignored, as well as models with non significants parameters (seuil = alpha) and models with convergence warnings
    tuples for p, d and q indicate min value and max value
    '''
    def get_arima_best_models(self, p=[0,1,2,3,4,5,6], d=[0,1,2], q=[0,1,2,3,4,5,6], alpha=0.05, method='lbfgs', maxiter=50):
        best_aic = [np.inf, np.inf, np.inf]
        best_order = [None, None, None]
        for i in p:
            for j in d:
                for k in q:
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            if self.exogenes is None:
                                temp_model = sm.tsa.statespace.SARIMAX(self.df[self.serie_values], order=(i,j,k)).fit(method=method, maxiter=maxiter)
                            else:
                                temp_model = sm.tsa.statespace.SARIMAX(self.df[self.serie_values], order=(i,j,k), exog=self.df[self.exogenes]).fit(method=method, maxiter=maxiter)
                        verif_params = True
                        for param in range(len(temp_model.pvalues)):
                            if (temp_model.pvalues.index[param] != 'sigma2') & (temp_model.pvalues[param] > alpha):
                                verif_params = False
                        if verif_params:
                            temp_aic = temp_model.aic
                            temp_rank = 3
                            for verif_rank in np.arange(2,-1,-1):
                                if temp_aic < best_aic[verif_rank]:
                                    temp_rank = verif_rank
                                else:
                                    verif_rank = -1
                            if temp_rank < 3:
                                for verif_rank in np.arange(2, temp_rank, -1):
                                    best_aic[verif_rank] = best_aic[verif_rank-1]
                                    best_order[verif_rank] = best_order[verif_rank-1]
                                best_aic[temp_rank] = temp_aic
                                best_order[temp_rank] = (i,j,k)
                    except:
                        continue
        print("Les trois meilleurs modèles sont :")
        for temp_rank in np.arange(len(best_aic)):
            print("AIC = {:.2f}  |  order = {}".format(best_aic[temp_rank], best_order[temp_rank]))
        return best_order[0]
    
    

        
        