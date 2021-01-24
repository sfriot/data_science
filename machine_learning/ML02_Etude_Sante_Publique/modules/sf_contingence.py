# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:06:42 2019

@author: Sylvain Friot

Content: two qualitative variables - analysis with contingency and chi tables
"""
import numpy as np
import pandas as pd
import scipy.stats as st
import logging as lg
import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    import sf_prints as sfp
    import sf_graphiques as sfg
else:
    import modules_perso.sf_prints as sfp
    import modules_perso.sf_graphiques as sfg


"""
Class : Contingency
Runs contingency analysis and test
Attributes
    rawdata = raw data from which contingency data are issued
    data = contingency table without totals (calculated with a pivot table based on rawdata) - observed frequencies
    index = name of the variable used as index in the pivot table
    column = name of the variable used as column in the pivot table
    contingency_table = contingency table with totals
    expected_table = theorical frequencies if variables are independent
    chi_ij_table = table of calculated chi_ij
    chi_n = value of chi-n (same as chi2 stat in the chi2 test)
    contributions_table = table of contributions to the chi-n
    residuals_table = table of residuals  - for analysis of dependance
    adjusted_residuals_table = table of adjusted residuals  - for analysis of dependance
Methods
    calcul_contingency_table, calcul_expected_table, calcul_chi_tables, calcul_residuals_tables : fonctions de calcul des différentes tables, appelées automatiquement au bon endroit
    test_chi2 = runs the chi2 test on H0 : the variables are independant
    group_levels = function to group 2 columns or 2 lines if observed frequencies or theo frequencies is < 5 in a single cell
    graph_contingency = heatmap of contributions table with the observed frequencies in annotation
"""
class Contingency:
    
    def __init__(self, data, index_name, column_name):
        self.rawdata = data.copy()
        self.data = data[[index_name,column_name]].pivot_table(index=index_name, columns=column_name, aggfunc=len, fill_value=0)
        self.index = index_name
        self.column = column_name
        self.contingency_table = None
        self.expected_table = None
        self.chi_ij_table = None
        self.chi_n = None
        self.contributions_table = None
        self.residuals_table = None
        self.adjusted_residuals_table = None
        self.calcul_contingency_table()
        self.calcul_expected_table()
        self.calcul_chi_tables()
        self.calcul_residuals_tables()
    
    def print_contingency_table(self):
        return self.contingency_table.style.format("{:,.0f}").\
                    set_caption("Tableau de contingence des deux variables")
    
    def print_expected_table(self):
        return self.expected_table.style.format("{:,.0f}").\
                    set_caption("Tableau des valeurs attendues si les deux variables sont indépendantes")
    
    def print_contributions_table(self):
        return self.contributions_table.style.format("{:.1%}").\
                    set_caption("Tableau des contributions au Chi-n ({:,.2f})".format(self.chi_n))
                    
    def print_etude_dependance(self):
        return self.adjusted_residuals_table.style.format("{:.3f}").\
                    set_caption("Tableau des résidus ajustés : étude de l'écart à l'indépendance".format(self.chi_n))
    
    def test_chi2(self, alpha=0.05, return_print=False):
        erreur = False
        for lin in range(self.data.shape[0]):
            for col in range(self.data.shape[1]):
                if (self.data.iloc[lin, col] < 5) | (self.expected_table.iloc[lin, col] < 5):
                    erreur = True
        if erreur:
            lg.warning("Les fréquences observées et théoriques doivent être supérieures à 5 pour toutes les cellules. Il faut regrouper des lignes ou des colonnes.")
            chi2_stat = np.nan
            p_value = np.nan
            dof = np.nan
            if return_print:
                return print("Erreur")
        else:
            chi2_stat, p_value, dof, _ = st.chi2_contingency(self.data)
            if return_print:
                test_object = "Test de l'hypothèse d'indépendance des fréquences observées dans le tableau de contingence"
                h0 = "Les variables {} et {} sont indépendantes".format(self.index, self.column)
                h1 = "Les variables {} et {} ne sont pas indépendantes".format(self.index, self.column)
                return sfp.print_one_test(test_object, h0, h1, p_value, stat_value=chi2_stat, alpha=alpha,
                                          stat_law="chi2 à {} degrés de liberté".format(dof))
        return chi2_stat, p_value, dof
        
    """
    Permet de regrouper les lignes ou colonnes position1 et position2 en une seule ligne/colonne
    Attention, position1 et position2 sont en base 0
    """
    def group_levels(self, position1, position2, column=True):
        newdata = self.data.copy()
        if column:
            newdata.iloc[:,position1] = newdata.iloc[:,position1] + newdata.iloc[:,position2]
            newdata.drop(columns=newdata.columns[position2], inplace=True)
        else:
            newdata.iloc[position1,:] = newdata.iloc[position1,:] + newdata.iloc[position2,:]
            newdata.drop(index=newdata.index[position2], inplace=True)
        self.data = newdata
        self.calcul_contingency_table()
        self.calcul_expected_table()
        self.calcul_chi_tables()
        self.calcul_residuals_tables()
    
    def graph_contingency(self, x_label, y_label, title, x_tick_labels=None, y_tick_labels=None, figsize=(12,8)):
        mygraph = sfg.MyGraph(title, figsize=figsize)
        mygraph.graph_contingency(self.contributions_table, annotations=self.data, x_label=x_label, y_label=y_label, chi_n=self.chi_n, x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels)
        return mygraph.fig, mygraph.ax        
        
    def calcul_contingency_table(self):
        self.contingency_table = self.data.copy()
        tx = self.rawdata[self.index].value_counts()
        ty = self.rawdata[self.column].value_counts()
        self.contingency_table.loc[:,"Total"] = tx
        self.contingency_table.loc["Total"] = ty
        self.contingency_table.loc["Total","Total"] = len(self.rawdata)
        
    def calcul_expected_table(self):
        self.expected_table = pd.DataFrame(st.contingency.expected_freq(self.data), index=self.data.index, columns=self.data.columns)
    
    def calcul_chi_tables(self):
        inter_table = self.data.copy()
        self.chi_ij_table = (inter_table - self.expected_table)**2 / self.expected_table
        self.chi_n = self.chi_ij_table.sum().sum()
        self.contributions_table = self.chi_ij_table / self.chi_n
        
    def calcul_residuals_tables(self):
        inter_table = self.data.copy()
        self.residuals_table = inter_table - self.expected_table
        self.adjusted_residuals_table = self.residuals_table / np.ravel(self.residuals_table).std()
        

