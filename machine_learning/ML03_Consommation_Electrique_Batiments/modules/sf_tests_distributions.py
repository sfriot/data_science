# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:23:58 2020

@author: Sylvain Friot

Content : tests statistiques sur les distributions
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.stats as sm
import statsmodels.stats.api as sma
import logging as lg
import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    import sf_prints as sfp
else:
    import modules_perso.sf_prints as sfp


'''
Shapiro Wilk : corrélation quantiles empiriques et quantiles théoriques (échantillons < 5000)
Anderson-Darling : sensible aux écarts dans les queues de distribution
Jarque-Bera : basé sur les coefficients d'asymétrie et d'aplatissement
'''

"""
Test de Kolmogorov Smirnov pour vérifier si la distribution de l'échantillon suit une loi continue définie
Paramètres
    tested_law: loi continue testée. Nom des lois continues dispo avec scipy.stats
Valeurs retournées
    dstat = valeur de la statistique D testée
    p_value = p-value du test
"""
def test_loi_continue_ks(x, tested_law='norm', other_parameter=None, forced_mean=None, forced_var=None, alpha=0.05, return_print=False):
    x = np.array(x)
    if len(x) < 5000:
        lg.warning("Il est déconseillé d'utiliser le test de Kolmogorov-Smirnov pour un échantillon de moins de 5000 individus")
    if forced_mean is None:
        forced_mean = x.mean()
    if forced_var is None:
        forced_var = x.var(ddof=1)
    if other_parameter is None:
        dstat, p_value = st.kstest(x, cdf=tested_law, args=(forced_mean, np.sqrt(forced_var)))
    else:
        dstat, p_value = st.kstest(x, cdf=tested_law, args=(other_parameter, forced_mean, np.sqrt(forced_var)))
    if return_print:
        stat_law="D"
        h0 = "La distribution suit une loi {}".format(tested_law)
        h1 = "La distribution ne suit pas une loi {}".format(tested_law)
        sfp.print_one_test("Test de Kolmogorov-Smirnov d'adéquation à la loi {} pour l'échantillon de taille {}".\
                           format(tested_law, len(x)), h0, h1, p_value, stat_law, dstat, alpha)
    return dstat, p_value

"""
Test de Shapiro-Wilk pour vérifier si la distribution de l'échantillon suit une loi normale
Valeurs retournées
    wstat = valeur de la statistique W testée
    p_value = p-value du test
"""
def test_loi_normale_shapiro(x, alpha=0.05, return_print=False):
    x = np.array(x)
    if len(x) > 5000:
        lg.warning("Il est déconseillé d'utiliser le test de Shapiro pour un échantillon de plus de 5000 individus")
    wstat, p_value = st.shapiro(x)
    if return_print:
        stat_law = "W"
        h0 = "La distribution suit une loi normale"
        h1 = "La distribution ne suit pas une loi normale"
        return sfp.print_one_test("Test de Shapiro-Wilk d'adéquation à la loi normale pour l'échantillon de taille {}".\
                                  format(len(x)), h0, h1, p_value, stat_law, wstat, alpha)
    return wstat, p_value

def test_loi_normale_jarquebera(x, alpha=0.05, return_print=False):
    x = np.array(x)
    jb_stat, jb_pvalue, skw, kurt = sm.stattools.jarque_bera(x)
    if return_print:
        stat_law = "JB (chi2 à 2 degrés de liberté)"
        h0 = "La distribution suit une loi normale"
        h1 = "La distribution ne suit pas une loi normale"
        return sfp.print_one_test("Test de Jarque-Bera d'adéquation au Kurtosis (=3) et au Skewness (=0) de la loi normale", \
                                  h0, h1, jb_pvalue, stat_law, jb_stat, alpha)
    return jb_stat, jb_pvalue

def test_loi_normale_andersondarling(x, alpha=0.05, return_print=False):
    x = np.array(x)
    ad_stat, ad_pvalue = sm.diagnostic.normal_ad(x)
    if return_print:
        stat_law = "AD"
        h0 = "La distribution suit une loi normale"
        h1 = "La distribution ne suit pas une loi normale"
        return sfp.print_one_test("Test d'Anderson-Darling d'adéquation aux valeurs extrêmes de la loi normale", \
                                  h0, h1, ad_pvalue, stat_law, ad_stat, alpha)
    return ad_stat, ad_pvalue

"""
Test loi normale
    si n<5000, on utilise par défaut le test de Shapiro-Wilk
    si n>=5000, on utilise par défaut le test de Kolmogorov-Smirnov
Valeurs retournées
    test_stat = valeur de la statistique D ou W testée
    p_value = p-value du test
"""    
def test_loi_normale_auto(x, alpha=0.05, return_print=False):
    x = np.array(x)
    if len(x) < 5000:
        return test_loi_normale_shapiro(x, alpha=alpha, return_print=return_print)
    return test_loi_continue_ks(x, tested_law='norm', alpha=alpha, return_print=return_print)

"""
Test de normalité : compile trois tests de normalité
    Shapiro-Wilk ou Kolmogorov-Smirnov selon la taille de l'échantillon : basé sur la comparaison des quantiles réels et des quantiles théoriques
    Jarque-Bera : basé sur les coefficients d'asymétrie et d'aplatissement
    Anderson-Darling : sensible aux écarts dans les queues de distribution
Valeurs retournées
    DataFrame avec les p-values pour chaque test
"""    
def test_normalite(x, title=None, alpha=0.05, return_print=False):
    x = np.array(x)
    if len(x) < 5000:
        my_index = [" Shapiro-Wilk "," Jarque-Bera "," Anderson-Darling "]
    else:
        my_index = [" Kolmogorov-Smirnov "," Jarque-Bera "," Anderson-Darling "]
    df_normalite = pd.DataFrame(index=my_index, columns=["p_value"])
    df_normalite.iloc[0,0] = np.round(test_loi_normale_auto(x, alpha)[1], 4)
    df_normalite.iloc[1,0] = np.round(test_loi_normale_jarquebera(x, alpha)[1], 4)
    df_normalite.iloc[2,0] = np.round(test_loi_normale_andersondarling(x, alpha)[1], 4)
    df_normalite["Accept H0"] = df_normalite.p_value >= alpha
    if return_print:
        if title is None:
            title = "Test de normalité"
        h0 = "La distribution suit une loi normale"
        h1 = "La distribution ne suit pas une loi normale"
        return sfp.print_df_tests(title, h0, h1, df_normalite, alpha=alpha)
    return df_normalite

def test_correlations_linear(df_data, x, list_y, title=None, alpha=0.05, return_print=False):
    list_y = list(list_y)
    df_correl = pd.DataFrame(index=list_y, columns=["Pearson Correlation","p_value"])
    for lin in range(len(list_y)):
        df_correl.iloc[lin] = st.pearsonr(df_data[x], df_data[list_y[lin]])
    df_correl["Accept H0"] = df_correl.p_value >= alpha
    df_correl["Conclusion"] = ["Absence de corrélation" if h0 else "Coefficient significatif" for h0 in df_correl["Accept H0"]]
    if return_print:
        if title is None:
            title = "Test de corrélation linéaire"
        h0 = "Le coefficient de corrélation linéaire de la variable {} avec les variables suivantes est nul (r=0)".format(x)
        h1 = "Le coefficient de corrélation linéaire n'est pas nul (r!=0)"
        return sfp.print_df_tests(title, h0, h1, df_correl, alpha=alpha, with_transpose=False)
    return df_correl
    
def test_correlations_rank(df_data, x, list_y, title=None, alpha=0.05, return_print=False):
    list_y = list(list_y)
    df_correl = pd.DataFrame(index=list_y, columns=["Spearman Correlation","p_value"])
    for lin in range(len(list_y)):
        df_correl.iloc[lin] = st.spearmanr(df_data[x], df_data[list_y[lin]])
    df_correl["Accept H0"] = df_correl.p_value >= alpha
    df_correl["Conclusion"] = ["Absence de corrélation" if h0 else "Coefficient significatif" for h0 in df_correl["Accept H0"]]
    if return_print:
        if title is None:
            title = "Test de corrélation de rang"
        h0 = "Le coefficient de corrélation de rang de la variable {} avec les variables suivantes est nul (r=0)".format(x)
        h1 = "Le coefficient de corrélation de rang n'est pas nul (r!=0)"
        return sfp.print_df_tests(title, h0, h1, df_correl, alpha=alpha, with_transpose=False)
    return df_correl

def test_anova_fisher(df_data, list_x, category, title=None, alpha=0.05, return_print=False):
    list_cat = np.unique(df_data[category].values)
    df_fisher = pd.DataFrame(index=list_x, columns=["F Stat","p_value"])
    for x in list_x:
        list_values = []
        df_cat = df_data[[x, category]].groupby(category)
        for cat, df_values in df_cat:
            list_values.append(df_values[x].dropna().values)
        df_fisher.loc[x] = st.f_oneway(*list_values)
    df_fisher["Accept H0"] = df_fisher.p_value >= alpha
    df_fisher["Conclusion"] = ["Les moyennes sont égales" if h0 else "Au moins une moyenne diffère des autres" for h0 in df_fisher["Accept H0"]]
    if return_print:
        if title is None:
            title = "Test ANOVA de Fisher"
        h0 = "Les moyennes des {} groupes sont égales".format(len(list_cat))
        h1 = "Au moins une moyenne n'est pas égale aux autres"
        return sfp.print_df_tests(title, h0, h1, df_fisher, alpha=alpha, with_transpose=False)
    return df_fisher

def test_anova_kruskal(df_data, list_x, category, title=None, alpha=0.05, return_print=False):
    list_cat = np.unique(df_data[category].values)
    df_kruskal = pd.DataFrame(index=list_x, columns=["H Stat","p_value"])
    for x in list_x:
        list_values = []
        df_cat = df_data[[x, category]].groupby(category)
        for cat, df_values in df_cat:
            list_values.append(df_values[x].dropna().values)
        df_kruskal.loc[x] = st.kruskal(*list_values)
    df_kruskal["Accept H0"] = df_kruskal.p_value >= alpha
    df_kruskal["Conclusion"] = ["Les médianes sont égales" if h0 else "Au moins une médiane diffère des autres" for h0 in df_kruskal["Accept H0"]]
    if return_print:
        if title is None:
            title = "Test ANOVA de Kruskal-Wallis (non-paramétrique)"
        h0 = "Les médianes des {} groupes sont égales".format(len(list_cat))
        h1 = "Au moins une médiane n'est pas égale aux autres"
        return sfp.print_df_tests(title, h0, h1, df_kruskal, alpha=alpha, with_transpose=False)
    return df_kruskal

def test_egalite_moyennes(x, y, equal_var=True, title=None, alpha=0.05, return_print=True):
    t_stat, p_value = st.ttest_ind(x, y, equal_var=equal_var)
    if return_print:
        h0 = "Les moyennes des 2 échantillons sont égales"
        h1 = "Les moyennes des 2 échantillons sont différentes"
        if equal_var:
            if title is None:
                title = "T-test de Student d'égalité des moyennes de deux échantillons"
            stat_law = "t de Student à {} degrés de liberté".format(len(x)+len(y)-2)
        else:
            if title is None:
                title = "T-test de Welch d'égalité des moyennes de deux échantillons"
            dof = ( (np.var(x,ddof=0)/len(x)) + (np.var(y,ddof=0)/len(y)) )**2 / \
                ( (np.var(x,ddof=0)**2 / ((len(x)-1)*len(x)**2)) + (np.var(y,ddof=0)**2 / ((len(y)-1)*len(y)**2)) )
            stat_law = "t de Student à {:.1f} degrés de liberté".format(dof)
        return sfp.print_one_test(title, h0, h1, p_value, stat_law=stat_law, stat_value=t_stat, alpha=alpha)
    return t_stat, p_value

def test_egalite_medianes(x, y, alternative='two-sided', title=None, alpha=0.05, return_print=True):
    if alternative is None:
        alternative='two-sided' # pour ne pas avoir None qui est déprécié
    u_stat, p_value = st.mannwhitneyu(x, y, alternative=alternative)
    if return_print:
        if title is None:
            title = "Test U de Mann-Whitney d'égalité des médianes de deux échantillons (non paramétrique)"
        h0 = "Les médianes des 2 échantillons sont égales"
        if alternative=='two-sided':
            h1 = "Les médianes des 2 échantillons sont différentes"
        elif alternative=='less':
            h1 = "La médiane du premier groupe est inférieure à la médiane du deuxième groupe"
        else:
            h1 = "La médiane du premier groupe est supérieure à la médiane du deuxième groupe"
        stat_law = "loi normale (asymptotiquement)"
        return sfp.print_one_test(title, h0, h1, p_value, stat_law=stat_law, stat_value=u_stat, alpha=alpha)
    return u_stat, p_value
