# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:48:33 2019

@author: Sylvain Friot

Content: one quantitative variable explained by quantitative variables - analysis and prediction with linear regression models
"""
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

import sf_graphiques as sfg


"""
"""
class TrainTestData:
        
    def __init__(self, save_name, data, train_percentage=0.75):
        self.train = None
        self.test = None
        self.get_split(save_name, data, train_percentage)
        
    def get_split(self, save_name, data, train_percentage):
        try:
            saved_indexes = pd.read_csv(save_name, index_col="index")
            self.train = data[~saved_indexes.test].copy()
            self.test = data[saved_indexes.test].copy()
            print("Train-test split existant chargé")
        except:
            self.train, self.test = model_selection.train_test_split(data, test_size=1.0-train_percentage, shuffle=True)
            inter_test = np.isin(data.index, list(self.test.index))
            saved_indexes = pd.DataFrame({'test':inter_test}, index=data.index)
            saved_indexes.to_csv(save_name, index_label="index")
            print("Nouveau train-test split sauvegardé")
            

"""
Class : LinearRegressionAnalysis
Runs linear regression analysis to select explanatory models and prediction models
Attributes
    
Methods
    
"""
class LinearRegressionAnalysis:
    
    def __init__(self, data, y_name, array_X_names, with_constant=True, with_standardization=True):
        list_variables = [y_name]
        list_variables.extend(array_X_names)
        self.rawdata = data[list_variables].copy()
        self.variable_estimee = y_name
        self.variables_explicatives = list(array_X_names)
        self.y = self.rawdata[self.variable_estimee].copy()
        if with_standardization:
            interX = self.rawdata[self.variables_explicatives].copy()
            self.scaler = preprocessing.StandardScaler().fit(interX)
            self.X = pd.DataFrame(data=self.scaler.transform(interX), index=self.rawdata.index, columns=self.variables_explicatives)
        else:
            self.scaler = None
            self.X = self.rawdata[self.variables_explicatives].copy()
        self.variables_retenues = list(array_X_names)
        self.df_linear_correl = None
        self.df_rankorder_correl = None
        self.with_constant = with_constant
        self.model = None
        self.regression = None
        self.coef_determination = None
        self.df_beta_estimates = None
        self.equation = None
        self.y_estimates = None
        self.residus = None
        self.anova_table = None
        self.sc_expliquee = None
        self.sc_residuelle = None
        self.variance_erreur_estim = None
        self.observations_analyses = None
        self.observations_seuils = None
        self.observations_to_remove = None
        self.run_regression()
        
    def re_init_variables_retenues(self):
        self.variables_retenues = self.variables_explicatives
        self.run_regression()
        
    def remove_from_variables_retenues(self, var_name):
        self.variables_retenues.remove(var_name)
        self.run_regression()

    def remove_from_data(self, index_list):
        self.rawdata = self.rawdata[~self.rawdata.index.isin(index_list)]
        self.y = self.rawdata[self.variable_estimee].copy()
        if self.scaler is None:
            self.X = self.rawdata[self.variables_explicatives].copy()
        else:
            interX = self.rawdata[self.variables_explicatives].copy()
            self.scaler = preprocessing.StandardScaler().fit(interX)
            self.X = pd.DataFrame(data=self.scaler.transform(interX), index=self.rawdata.index, columns=self.variables_explicatives)
        self.run_regression()
        
    def run_regression(self, alpha=0.05, add_constant=None, do_print=True):
        self.calcul_correlations()
        my_y = self.y.copy()
        my_X = self.X[self.variables_retenues].copy()
        n, p = my_X.shape
        if add_constant is not None:
            self.with_constant = add_constant
        if self.with_constant:
            my_X = sm.add_constant(my_X)
        self.model = sm.OLS(my_y, my_X)
        self.regression = self.model.fit()
        self.y_estimates = self.regression.fittedvalues
        self.residus = self.regression.resid
        self.coef_determination = self.regression.rsquared
        self.df_beta_estimates = pd.DataFrame(np.nan, index=["beta_estimates","adjusted_beta"], columns=self.variables_retenues)
        y_std = my_y.std(ddof=1)
        self.equation = "{} =".format(self.variable_estimee)
        for variable in self.variables_retenues:
            coef = self.regression.params[variable]
            x_std = my_X[variable].std(ddof=1)
            self.df_beta_estimates.loc["beta_estimates",variable] = coef
            self.df_beta_estimates.loc["adjusted_beta",variable] = coef * x_std / y_std
            self.equation = self.equation + " {:.3f} * {} +".format(coef, variable)
        if self.with_constant:
            coef = self.regression.params["const"]
            self.df_beta_estimates["constante"] = [coef, np.nan]
            self.equation = self.equation + " {:.3f}".format(coef)
        else:
            self.equation = self.equation[:-2]
        #self.anova_table = sm.stats.anova_lm(self.regression)
        self.sc_residuelle = np.sum((my_y - self.y_estimates) ** 2)
        self.sc_expliquee = np.sum((self.y_estimates - my_y.mean()) ** 2)
        self.variance_erreur_estim = self.sc_residuelle / (n - p - 1)
        self.anova_table = pd.DataFrame({"somme_carres":[self.sc_expliquee,self.sc_residuelle,self.sc_expliquee+self.sc_residuelle]}, index=["expliquee","residuelle","totale"])
        self.anova_table["dl"] = [p, n-p-1, n-1]
        self.anova_table.loc["expliquee":"residuelle","carres_moyens"] = self.anova_table.loc["expliquee":"residuelle","somme_carres"] / self.anova_table.loc["expliquee":"residuelle","dl"]
        self.anova_table.loc["expliquee","F_stats"] = self.anova_table.loc["expliquee","carres_moyens"] / self.anova_table.loc["residuelle","carres_moyens"]
        self.anova_table.loc["expliquee","p_valeur"] = st.f.sf(self.anova_table.loc["expliquee","F_stats"], p, n-p-1)
        self.calcul_dfanalyse_observations(alpha=alpha)
        if do_print:
            print("L'analyse a été calculée")

    def calcul_correlations(self, alpha=0.05):
        self.df_linear_correl = pd.DataFrame(np.nan, index=["Personn Correlation","p-value - H0:r=0","Test result (alpha = {:.0%})".format(alpha)], columns=self.variables_explicatives)
        self.df_rankorder_correl = pd.DataFrame(np.nan, index=["Spearman Correlation","p-value - H0:r=0","Test result (alpha = {:.0%})".format(alpha)], columns=self.variables_explicatives)
        self.df_linear_correl.index.name = self.variable_estimee
        self.df_rankorder_correl.index.name = self.variable_estimee
        for col in np.arange(len(self.variables_explicatives)):
            inter_result = st.pearsonr(self.y, self.X[self.variables_explicatives[col]])
            self.df_linear_correl.iloc[0, col] = inter_result[0]
            self.df_linear_correl.iloc[1, col] = inter_result[1]
            if inter_result[1] > alpha:
                self.df_linear_correl.iloc[2, col] = "Absence de corrélation"
            else:
                self.df_linear_correl.iloc[2, col] = "Corrélation significative"
            inter_result = st.spearmanr(self.y, self.X[self.variables_explicatives[col]])
            self.df_rankorder_correl.iloc[0, col] = inter_result[0]
            self.df_rankorder_correl.iloc[1, col] = inter_result[1]
            if inter_result[1] > alpha:
                self.df_rankorder_correl.iloc[2, col] = "Absence de corrélation"
            else:
                self.df_rankorder_correl.iloc[2, col] = "Corrélation significative"
            
    def calcul_dfanalyse_observations(self, alpha=0.05, correction_seuils=True):
        n = len(self.y)
        p = len(self.variables_retenues)
        seuils = pd.Series(index=["levier", "resid_student", "dffits", "cook","dfbetas"])
        if correction_seuils:
            seuils.levier = 2 * (p + 1) / n
            seuils.dffits = 2 * np.sqrt((p + 1) / n)
            seuils.cook = 4 / (n - p - 1)
        else:
            seuils.levier = 2 * p / n
            seuils.dffits = 2 * np.sqrt(p / n)
            seuils.cook = 4 / (n - p)
        seuils.resid_student = st.t.ppf(1-(alpha/2), n-p-2)
        seuils.dfbetas = 2 / np.sqrt(n)
        reg_influence = self.regression.get_influence()
        analyses = pd.DataFrame({"observation_name":self.y.index})
        analyses["levier"] = reg_influence.hat_matrix_diag
        analyses["external_resid_student"] = reg_influence.resid_studentized_external
        analyses["outlier"] = ((analyses.levier > seuils.levier) | (abs(analyses.external_resid_student) > seuils.resid_student))
        analyses["dffits"] = reg_influence.dffits[0]
        analyses["cook_distance"] = reg_influence.cooks_distance[0]
        analyses["influence"] = (abs(analyses.dffits) > seuils.dffits) | (analyses.cook_distance > seuils.cook)
        dfbetas = pd.DataFrame({"observation_name":self.y.index})
        for i_var in np.arange(len(self.regression.model.exog_names)):
            variable = self.regression.model.exog_names[i_var]
            dfbetas[variable] = reg_influence.dfbetas[:,i_var]
        self.observations_analyses = analyses
        self.observations_seuils = seuils
        self.observations_to_remove = analyses[analyses.outlier & analyses.influence].observation_name.values
        
    def print_residuals_validity(self, alpha=0.05, with_line=True):
        print("\033[1m"+"NORMALITÉ DES RÉSIDUS"+"\033[0m\n")
        if len(self.residus) < 5000:
            stat, p_value = st.shapiro(self.residus)
            test = "Shapiro"
        else:
            stat, p_value = st.kstest(self.residus, tested_law='norm', args=(self.residus.mean(), self.residus.var(ddof=1)))
            test = "Kolmogorov-Smirnov"
        print("Test de {} : p-value = {:.3f}.".format(test, p_value))
        if p_value >= alpha:
             print("L'hypothèse de normalité des résidus est acceptée.")
        else:
            print("L'hypothèse de normalité des résidus est rejetée.")
        print("")
        jb_stat, jb_pvalue, skw, kurt = sm.stats.stattools.jarque_bera(self.residus)
        print("Test de Jarque-Bera : p-value = {:.3f}.".format(jb_pvalue))
        if jb_pvalue > alpha:
            print("L'hypothèse de normalité des résidus est acceptée.")
        else:
            print("L'hypothèse de normalité des résidus est rejetée.")
        print("")
        if len(self.residus) >= 30:
            print("Comme il y a plus de 30 observations ({:,.0f}), on peut accepter la normalité si la distribution est à peu près symétrique.".format(self.residus.size))
            print("")
        fig, ax = self.graph_residus()
        plt.show()
        print("")
        fig, ax = self.graph_droite_henry_residus()
        plt.show()
        print("")
        print("\033[1m"+"HOMOSCEDASTICITÉ DES RÉSIDUS"+"\033[0m\n")
        lm_stat, lm_pvalue, f_stat, f_pvalue = sm.stats.diagnostic.het_breuschpagan(self.residus, self.regression.model.exog)
        print("Test de l'homoscédasticité de Breusch Pagan : p-value = {:.3f}.".format(lm_pvalue))
        if lm_pvalue >= alpha:
            print("L'hypothèse de constance de la variance est acceptée.")
        else:
            print("L'hypothèse de constance de la variance est rejetée.")
        print("")
        fig, ax = self.graph_correlation_residus(with_line=with_line)
        plt.show()
        print("")
        fig, ax = self.graph_independance_residus()
        plt.show()
        
    def print_observations_validity(self, alpha=0.05, graphs=True):
        # remarque : pour les graphiques issus de get_influence(), possibilité d'utiliser directement .plot_index(y_var='dcooks'... , [idx=variable_index], threshold=limite_pour_afficher_numero_observation)
        # notamment pour rajouter dfbetas
        analyses = self.observations_analyses
        seuils = self.observations_seuils
        if graphs:
            fig, ax = self.graph_outliers_levier(analyses, seuils.levier)
            plt.show()
            print("")
            fig, ax = self.graph_outliers_resid_student(analyses, seuils.resid_student)
            plt.show()
            print("")
            fig, ax = self.graph_influence_dffits(analyses, seuils.dffits)
            plt.show()
            print("")
            fig, ax = self.graph_influence_cook(analyses, seuils.cook)
            plt.show()
            print("")
            fig, ax = self.graph_influence_plot(seuils)
            plt.show()
            print("")
        variables_speciales = analyses[(analyses.outlier | analyses.influence) & (analyses.outlier != analyses.influence)]
        variables_investigation = analyses[analyses.outlier & analyses.influence]
        print("\033[1m"+"RAPPEL DES SEUILS"+"\033[0m")
        df_affichage = pd.DataFrame(seuils).T.applymap("{:.3f}".format)
        print(df_affichage.iloc[:,:-1].to_string(index=False))
        print("")
        if graphs:
            print("\033[1m"+"RÉSUMÉ DES OBSERVATIONS ATYPIQUES OU INFLUENTES"+"\033[0m")
            df_affichage = variables_speciales.copy()
            df_affichage = self.__mep_dfaffichage_seuils(df_affichage)
            print(df_affichage)
            print("")
        print("\033[1m"+"RÉSUMÉ DES OBSERVATIONS ATYPIQUES ET INFLUENTES"+"\033[0m")
        print("Ces variables sont à investiguer : elles sont atypiques et pèsent sur la régression")
        df_affichage = variables_investigation.copy()
        df_affichage = self.__mep_dfaffichage_seuils(df_affichage)
        print(df_affichage)
        if graphs:
            print("")
            fig, ax = self.graph_linearite_residuspartiels()
            plt.show()
        
    def print_resultats(self):
        print(self.regression.summary())
        print("Rappel. Les tests portent sur H0 : estimateur=0. Si p-value < alpha, la variable est significative.")
        print("")
        if self.scaler is None:
            print("\033[1m"+"Tableau des coefficients beta estimés et de leurs valeurs standardisées"+"\033[0m")
        else:
            print("\033[1m"+"Tableau des coefficients beta estimés et de leurs valeurs standardisées"+"\033[0m"+" (variables centrées réduites)")
        df_affichage = self.df_beta_estimates.applymap("{:,.3f}".format)
        print(df_affichage)
        print("")
        if self.scaler is None:
            print("\033[1m"+"Equation de la relation estimée par le modèle :"+"\033[0m")
        else:
            print("\033[1m"+"Equation de la relation estimée par le modèle :"+"\033[0m"+" (variables centrées réduites)")
        print(self.equation)
        print("")
        print("\033[1m"+"Tableau d'analyse de la variance"+"\033[0m")
        df_affichage = self.anova_table.copy()
        df_affichage["somme_carres"] = df_affichage["somme_carres"].apply("{:,.0f}".format if df_affichage.somme_carres.max() >= 1000 else "{:.2f}".format)
        df_affichage["dl"] = df_affichage["dl"].apply("{:,.0f}".format)
        df_affichage["carres_moyens"] = df_affichage["carres_moyens"].apply("{:,.0f}".format if df_affichage.carres_moyens.max() >= 1000 else "{:.2f}".format)
        df_affichage["F_stats"] = df_affichage["F_stats"].apply("{:,.0f}".format if df_affichage.F_stats.max() >= 1000 else "{:.3f}".format)
        df_affichage["p_valeur"] = df_affichage["p_valeur"].apply("{:.3f}".format)
        print(df_affichage)
        print("")
        if len(self.variables_retenues) == 1:
            fig, ax = self.graph_linear_regression()
            plt.show()
        else:
            fig, ax = self.graph_estimates_accuracy()
            plt.show()
    
    def print_multicolinearity_validity(self):
        if len(self.variables_retenues) == 1:
            print("Il n'y a qu'une variable explicative. Aucun problème de colinéarité ne peut se poser.")
        else:
            print("\033[1m"+"ANALYSE DE LA COLINÉARITÉ SIMPLE"+"\033[0m")
            print("Les valeurs de la matrice de corrélation doivent être inférieures à 0.8")
            fig, ax = self.graph_heatmap_correlationmatrix()
            plt.show()
            print("")
            variables = self.regression.model.exog
            vif_analysis = pd.DataFrame([variance_inflation_factor(variables, i) for i in np.arange(1,variables.shape[1])], \
                                      index=[var for var in self.regression.model.exog_names if var!="const"], columns=["VIF"])
            print("\033[1m"+"ANALYSE DE LA MULTI-COLINÉARITÉ"+"\033[0m")
            print("L'indice VIF doit être inférieur à 4")
            print(vif_analysis.applymap("{:.3f}".format))
        
    def graph_linear_regression(self, figsize=(12,8)):
        if len(self.variables_retenues) > 1:
            return
        label_correl = "r = {:.3f} - p-value = {:.3f}".format(self.df_linear_correl.iloc[0,0], self.df_linear_correl.iloc[1,0])
        mygraph = sfg.MyGraph("Graphique de la régression linéaire", is_mono=False)
        mygraph.add_plot(self.X[self.variables_retenues], self.y, label=label_correl, marker='o', linestyle="")
        mygraph.add_plot(self.X[self.variables_retenues], self.y_estimates, label=self.equation, legend=True)
        mygraph.set_axe('y', label=self.variable_estimee)
        mygraph.set_axe('x', label=self.variables_retenues[0])
        return mygraph.fig, mygraph.ax
            
    def graph_estimates_accuracy(self, figsize=(12,8)):
        mygraph = sfg.MyGraph("Estimations par rapport aux valeurs réelles", is_mono=False)
        mygraph.add_plot(self.y, self.y_estimates, label="Estimations vs valeurs réelles", marker='o', linestyle="")
        ymin, ymax = mygraph.ax[0].get_ylim()
        xmin, xmax = mygraph.ax[0].get_xlim()
        data_min = min(xmin, ymin)
        data_max = max(xmax, ymax)
        mygraph.add_plot([data_min, data_max], [data_min, data_max], label="Première bissectrice", legend=True)
        mygraph.set_axe('y', label="Estimations de {}".format(self.variable_estimee))
        mygraph.set_axe('x', label="Valeurs réelles de {}".format(self.variable_estimee))
        mygraph.ax[0].autoscale(enable=True, axis='both', tight=True)
        mygraph.fig.tight_layout()
        return mygraph.fig, mygraph.ax
        
    def graph_residus(self, title=None, figsize=(12,8)):
        if title is None:
            title = "Histogramme de la distribution des résidus"
        x_theo = np.arange(self.residus.min(), self.residus.max(), 0.01*(self.residus.max()-self.residus.min()))
        mygraph = sfg.MyGraph(title, is_mono=False)
        mygraph.add_histogramme(self.residus, bins=20, labels="Résidus")
        mygraph.add_plot(x_theo, st.norm.pdf(x_theo, scale=self.residus.std(ddof=1)), label="Loi normale", legend=True)
        xmin, xmax = mygraph.ax[0].get_xlim()
        absmax = max(-xmin, xmax)
        mygraph.set_axe('y', label="Fréquence de distribution")
        mygraph.set_axe('x', label="Résidus", tick_min=-absmax, tick_max=absmax)
        return mygraph.fig, mygraph.ax
    
    def graph_droite_henry_residus(self):
        mygraph = sfg.MyGraph("Droite de Henry : vérification de la normalité des résidus", is_mono=False)
        sm.qqplot(self.residus, fit=True, markeredgecolor=mygraph.liste_couleurs[0], markerfacecolor=mygraph.liste_couleurs[0], alpha=0.5, ax=mygraph.ax[0])
        ymin, ymax = mygraph.ax[0].get_ylim()
        xmin, xmax = mygraph.ax[0].get_xlim()
        data_min = min(xmin, ymin)
        data_max = max(xmax, ymax)
        mygraph.add_plot([data_min, data_max], [data_min, data_max], label=" ", color=mygraph.liste_couleurs[1])
        mygraph.set_axe('x', label="Quantiles théoriques de la loi normale")
        mygraph.set_axe('y', label="Quantiles observés des résidus")
        mygraph.ax[0].autoscale(enable=True, axis='both', tight=True)
        mygraph.fig.tight_layout()
        return mygraph.fig, mygraph.ax
    
    def graph_independance_residus(self):
        if len(self.variables_retenues) == 1:
            nblin = 1
            nbcol = 1
        else:
            nbcol = 2
            nblin = ((len(self.variables_retenues) - 1) // nbcol) + 1
        if nblin == 1:
            figsize = (12,8)
        else:
            figsize = (12, nblin*6)
        mygraph = sfg.MyGraph("Vérification de l'indépendance des résidus", nblin=nblin, nbcol=nbcol, is_mono=False, figsize=figsize)
        cpt = 0
        for variable in self.variables_retenues:
            cpt += 1
            mygraph.add_plot(self.X[variable], self.residus, label=" ", marker='o', linestyle="", markeredgecolor=mygraph.liste_couleurs[(cpt-1)%len(mygraph.liste_couleurs)], \
                             markerfacecolor=mygraph.liste_couleurs[(cpt-1)%len(mygraph.liste_couleurs)], subtitle="Variable {}".format(variable), multi_index=cpt)
            ymin, ymax = mygraph.ax[cpt-1].get_ylim()
            absmax = max(-ymin, ymax)
            mygraph.set_axe('y', tick_min=-absmax, tick_max=absmax, multi_index=cpt)
        while cpt < (nblin * nbcol):
            cpt += 1
            mygraph.ax[cpt-1].set_visible(False)
        mygraph.fig.text(0.5, -0.01, "Variable", ha='center', fontweight='bold')
        mygraph.fig.text(-0.01, 0.5, "Résidus", va='center', rotation='vertical', fontweight='bold')
        mygraph.fig.tight_layout()
        return mygraph.fig, mygraph.ax
    
    def graph_linearite_residuspartiels(self, lissage=False):
        if len(self.y) < 1000:
            nb_lissage = len(self.y) // 10
        else:
            nb_lissage = 100
        if len(self.variables_retenues) == 1:
            nbcol = 1
            nblin = 1
        else:
            nbcol = 2
            nblin = ((len(self.variables_retenues) - 1) // nbcol) + 1
        if nblin == 1:
            figsize = (12,8)
        else:
            figsize = (12, nblin*6)
        mygraph = sfg.MyGraph("Vérification de la linéarité de la relation", nblin=nblin, nbcol=nbcol, is_mono=False, figsize=figsize)
        cpt = 0
        for variable in self.variables_retenues:
            residus_partiels = (self.y - self.y_estimates) + (self.regression.params[variable] * self.X[variable]) 
            inter_data = pd.DataFrame({variable : self.X[variable], "residus_partiels" : residus_partiels})
            if lissage:
                variable_classe = "{}_classe".format(variable)
                moyenne_variable = "moyenne_{}".format(variable)
                inter_data[variable_classe] = pd.cut(inter_data[variable], bins=nb_lissage)
                average_by_class = inter_data[[variable_classe,variable,"residus_partiels"]].groupby(variable_classe).mean()
                average_by_class.reset_index(inplace=True)
                average_by_class.columns = [variable_classe,moyenne_variable,"moyenne_residus_partiels"]
                inter_data = inter_data.merge(average_by_class, on=variable_classe, how="left")
            cpt += 1
            if lissage:
                mygraph.add_regplot(x=inter_data[moyenne_variable], y=inter_data.moyenne_residus_partiels, subtitle="Variable {}".format(variable), show_labels=False, multi_index=cpt)
            else:
                mygraph.add_regplot(x=inter_data[variable], y=inter_data.residus_partiels, subtitle="Variable {}".format(variable), show_labels=False, multi_index=cpt)
        while cpt < (nblin * nbcol):
            cpt += 1
            mygraph.ax[cpt-1].set_visible(False)
        mygraph.fig.text(0.5, -0.01, "Variable", ha='center', fontweight='bold')
        if lissage:
            mygraph.fig.text(-0.01, 0.5, "Moyenne des résidus partiels", va='center', rotation='vertical', fontweight='bold')
        else:
            mygraph.fig.text(-0.01, 0.5, "Résidus partiels", va='center', rotation='vertical', fontweight='bold')
        mygraph.fig.tight_layout()
        return mygraph.fig, mygraph.ax
    
    def graph_correlation_residus(self, with_line=True):
        mygraph = sfg.MyGraph("Graphique des résidus : indépendance avec la variable expliquée", is_mono=False)
        interdata = pd.DataFrame({'y_estimates':self.y_estimates, 'residus':self.residus})
        interdata.sort_values(by="y_estimates", inplace=True)
        if with_line:
            mygraph.add_plot(interdata.y_estimates, interdata.residus, label="", marker='o', linestyle=':', with_grid='both', grid_style=":")
        else:
            mygraph.add_plot(interdata.y_estimates, interdata.residus, label="", marker='o', with_grid='both', grid_style=":")
        yseuil = 2 * np.sqrt(self.variance_erreur_estim)
        mygraph.add_line(-yseuil, vertical=False, color=mygraph.liste_couleurs[1])
        mygraph.add_line(yseuil, vertical=False, label="2 écart-types de l'erreur", color=mygraph.liste_couleurs[1], legend=True)
        ymin, ymax = mygraph.ax[0].get_ylim()
        absmax = max(-ymin, ymax, 1.1*yseuil)
        mygraph.set_axe('x', label="Estimations de y")
        mygraph.set_axe('y', label="Résidus", tick_min=-absmax, tick_max=absmax)
        mygraph.ax[0].autoscale(enable=True, axis='x', tight=True)
        return mygraph.fig, mygraph.ax
    
    def graph_outliers_levier(self, df_analyse, seuil_levier):
        mygraph = sfg.MyGraph("Graphique des leviers : valeurs atypiques sur les variables explicatives", is_mono=False)
        mygraph.add_barv(df_analyse.index, df_analyse.levier, label="", color=mygraph.liste_couleurs[0])
        mygraph.add_line(seuil_levier, vertical=False, label="Seuil = 2 * (p+1) / n", color=mygraph.liste_couleurs[1], legend=True)
        ymin, ymax = mygraph.ax[0].get_ylim()
        absmax = max(ymax, 1.1*seuil_levier)
        mygraph.set_axe('x', label="Observations", tick_dash=True)
        mygraph.set_axe('y', label="Leviers", tick_max=absmax)
        mygraph.ax[0].autoscale(enable=True, axis='x', tight=True)
        return mygraph.fig, mygraph.ax
    
    def graph_outliers_resid_student(self, df_analyse, seuil_resid_student):
        mygraph = sfg.MyGraph("Graphique des résidus studentisés externes : valeurs atypiques sur la variable estimée", is_mono=False)
        mygraph.add_barv(df_analyse.index, df_analyse.external_resid_student, label="", color=mygraph.liste_couleurs[0], with_grid='both')
        mygraph.add_line(-seuil_resid_student, vertical=False, label="Seuil = t de Student à n-p-2 dl", color=mygraph.liste_couleurs[1], legend=True)
        mygraph.add_line(seuil_resid_student, vertical=False, color=mygraph.liste_couleurs[1])
        ymin, ymax = mygraph.ax[0].get_ylim()
        absmax = max(-ymin, ymax, 1.1*seuil_resid_student)
        mygraph.set_axe('x', label="Observations")
        mygraph.set_axe('y', label="Résidus studentisés", tick_min=-absmax, tick_max=absmax)
        mygraph.ax[0].autoscale(enable=True, axis='x', tight=True)
        return mygraph.fig, mygraph.ax
        
    def graph_influence_dffits(self, df_analyse, seuil_dffits):
        mygraph = sfg.MyGraph("Graphique des dffits : observations influentes sur la régression", is_mono=False)
        mygraph.add_barv(df_analyse.index, df_analyse.dffits, label="", color=mygraph.liste_couleurs[0], with_grid='both')
        mygraph.add_line(-seuil_dffits, vertical=False, label="Seuil = 2 * sqrt((p+1) / n)", color=mygraph.liste_couleurs[1], legend=True)
        mygraph.add_line(seuil_dffits, vertical=False, label="", color=mygraph.liste_couleurs[1])
        ymin, ymax = mygraph.ax[0].get_ylim()
        absmax = max(-ymin, ymax, 1.1*seuil_dffits)
        mygraph.set_axe('x', label="Observations")
        mygraph.set_axe('y', label="DFFITS", tick_min=-absmax, tick_max=absmax)
        mygraph.ax[0].autoscale(enable=True, axis='x', tight=True)
        return mygraph.fig, mygraph.ax
    
    def graph_influence_cook(self, df_analyse, seuil_cook):
        mygraph = sfg.MyGraph("Graphique des distances de Cook : observations influentes sur la régression", is_mono=False)
        mygraph.add_barv(df_analyse.index, df_analyse.cook_distance, label="", color=mygraph.liste_couleurs[0])
        mygraph.add_line(seuil_cook, vertical=False, label="Seuil = 4 / (n-p-1)", color=mygraph.liste_couleurs[1], legend=True)
        ymin, ymax = mygraph.ax[0].get_ylim()
        absmax = max(ymax, 1.1*seuil_cook)
        mygraph.set_axe('x', label="Observations", tick_dash=True)
        mygraph.set_axe('y', label="Distance de Cook", tick_max=absmax)
        mygraph.ax[0].autoscale(enable=True, axis='x', tight=True)
        return mygraph.fig, mygraph.ax
    
    def graph_influence_plot(self, seuils):
        mygraph = sfg.MyGraph("Résumé des observations atypiques et influentes", is_mono=False)
        sm.graphics.influence_plot(self.regression, ax=mygraph.ax[0])
        mygraph.add_plot(np.NaN, np.NaN, label="Influence : distance de Cook", marker='o', linestyle="", color=mygraph.liste_couleurs[0])
        ymin, ymax = mygraph.ax[0].get_ylim()
        absmax = max(-ymin, ymax, 1.1*seuils.resid_student)
        mygraph.set_axe('y', label="Atypicité : résidus studentisés", tick_min=-absmax, tick_max=absmax)
        xmin, xmax = mygraph.ax[0].get_xlim()
        absmax = max(xmax, 1.1*seuils.levier)
        mygraph.set_axe('x', label="Atypicité : leviers", tick_max=absmax)
        mygraph.add_line(-seuils.resid_student, vertical=False, color=mygraph.liste_couleurs[1], label="Seuils d'atypicité")
        mygraph.add_line(seuils.resid_student, vertical=False, color=mygraph.liste_couleurs[1])
        mygraph.add_line(seuils.levier, vertical=True, color=mygraph.liste_couleurs[1])
        mygraph.ax[0].set_title("")
        mygraph.fig.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.975))
        return mygraph.fig, mygraph.ax
        
    def graph_heatmap_correlationmatrix(self):
        matrice_correl = np.corrcoef(self.X[self.variables_retenues], rowvar=False)
        mygraph = sfg.MyGraph("Heatmap de la matrice de corrélation des variables explicatives", is_mono=True)
        #cmap=plc.LinearSegmentedColormap.from_list("", [mygraph.liste_couleurs[0],mygraph.liste_couleurs[1]]) with is_mono=False
        #cmap=plt.cm.Blues
        sns.heatmap(matrice_correl, vmin=-1, vmax=1, fmt='.2f', cmap=plc.LinearSegmentedColormap.from_list("", [mygraph.liste_couleurs[-1],mygraph.liste_couleurs[0],mygraph.liste_couleurs[-1]]), \
                    cbar_kws={'ticks':[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]}, annot=True, ax=mygraph.ax[0])
        mygraph.set_axe('y', tick_labels=self.variables_retenues, rotation=0)
        mygraph.set_axe('x', tick_labels=self.variables_retenues, rotation=30, ha='right')
        mygraph.fig.tight_layout()
        return mygraph.fig, mygraph.ax
    
    def __mep_dfaffichage_seuils(self, df_affichage):
        df_affichage["levier"] = df_affichage["levier"].apply("{:.3f}".format)
        df_affichage["external_resid_student"] = df_affichage["external_resid_student"].apply("{:.3f}".format)
        df_affichage["dffits"] = df_affichage["dffits"].apply("{:.3f}".format)
        df_affichage["cook_distance"] = df_affichage["cook_distance"].apply("{:.3f}".format)
        return df_affichage
    
    """
    Linear model designed by backward selection for regression analysis : parameter with the largest p-value is removed if its p-value is >= alpha
    """
    def backward_selection_analysis(self, add_constant=True, alpha=0.05):
        remaining = self.variables_explicatives.copy()
        my_y = self.y.copy()
        cond = True
        while cond:
            liste_variables = "Variables explicatives : " + ", ".join(remaining)
            my_X = self.X[remaining].copy()
            if add_constant:
                my_X = sm.add_constant(my_X)
                liste_variables = liste_variables + ", constante"
            model = sm.OLS(my_y, my_X).fit()
            scores = model.pvalues.loc[remaining]
            to_remove = scores[scores == scores.max()]
            if to_remove.values[0] >= alpha:
                remaining.remove(to_remove.index[0])
                action = "Variable retirée : {} (p-value = {:.3f})".format(to_remove.index[0], to_remove.values[0])
                if len(remaining) < 2:
                    action = action + "\n\nIl ne reste qu'une variable - Modèle final obtenu\nVariable explicative : {}".format(remaining[0])
                    cond = False
            else:
                action = "Modèle final obtenu"
                cond = False
            print(liste_variables)
            print(action)
            print("")
        self.variables_retenues = remaining
        self.run_regression_analysis(add_constant=add_constant)
    
    """
    Linear model designed by forward selection for regression prediction : variables are added as long as the criteria decreases
    Final step : backward selection on selected variables to check for cross-effects on the criteria
    AIC is the default selection criteria ; if bic=True then BIC becomes the selection criteria
    """
    def forward_selection_prediction(self, add_constant=True, bic=False):
        remaining = self.variables_explicatives.copy()
        my_y = self.y.copy()
        my_X = np.ones((len(my_y),1))
        if bic:
            criteria = "BIC"
            current_score = sm.OLS(my_y, my_X).fit().bic
        else:
            criteria = "AIC"
            current_score = sm.OLS(my_y, my_X).fit().aic
        selected = []
        cond = True
        print("Sélection forward basée sur le critère {}".format(criteria))
        while cond:
            scores_with_candidates = []
            for candidate in remaining:
                liste_variables = selected.copy()
                liste_variables.append(candidate)
                my_X = self.X[liste_variables].copy()
                if add_constant:
                    my_X = sm.add_constant(my_X)
                if bic:
                    score = sm.OLS(my_y, my_X).fit().bic
                else:
                    score = sm.OLS(my_y, my_X).fit().aic
                scores_with_candidates.append((score, candidate))
            scores_with_candidates.sort()
            best_new_score, best_candidate = scores_with_candidates[0]
            if best_new_score < current_score:
                action = "Variable ajoutée : {} ({} = {:.3f}).".format(best_candidate, criteria, best_new_score)
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
                if len(remaining) == 0:
                    action += "\nPlus de variable disponible."
                    cond = False
            else:
                action = "Modèle intermédiaire atteint : meilleur {} obtenu.".format(criteria)
                cond = False
            print(action)
            print("")
            
        # backward ajouté pour prendre en compte effets croisés
        print("Sélection backward à partir des variables sélectionnées, pour vérifier d'éventuels effets croisés")
        cond = True
        while cond:
            scores_with_candidates = []
            for candidate in selected:
                liste_variables = selected.copy()
                liste_variables.remove(candidate)
                my_X = self.X[liste_variables].copy()
                if add_constant:
                    my_X = sm.add_constant(my_X)
                if bic:
                    score = sm.OLS(my_y, my_X).fit().bic
                else:
                    score = sm.OLS(my_y, my_X).fit().aic
                scores_with_candidates.append((score, candidate))
            scores_with_candidates.sort()
            best_new_score, best_candidate = scores_with_candidates[0]
            if best_new_score <= current_score:
                action = "Variable supprimée : {} ({} = {:.3f}).".format(best_candidate, criteria, best_new_score)
                selected.remove(best_candidate)
                remaining.append(best_candidate)
                current_score = best_new_score
                if len(selected) == 0:
                    action += "\nModèle final atteint : aucune variable sélectionnée."
                    cond = False
            else:
                action = "Modèle final atteint : meilleur {} obtenu.".format(criteria)
                cond = False
            print(action)
            print("")
        
        liste_variables = "Variables explicatives : " + ", ".join(selected)
        if add_constant:
            liste_variables = liste_variables + ", constante"
        print(liste_variables)
        print("{} du modèle = {:3f}".format(criteria, current_score))
        self.variables_retenues = selected.copy()
        self.run_regression_analysis(add_constant=add_constant)

    def calcul_predictions_old(self, data_input, confidence_interval=0.95):
        n = len(self.y)
        p = len(self.variables_retenues)
        n_predictions = len(data_input)
        if self.scaler is None:
            data_exog = data_input[self.variables_retenues].copy()
        else:
            interX = data_input[self.variables_retenues].copy()
            data_exog = pd.DataFrame(data=self.scaler.transform(interX), index=data_input.index, columns=self.variables_retenues)
        if self.with_constant:
            data_exog = sm.add_constant(data_exog, has_constant='add')
        predictions = self.regression.predict(data_exog)
        # calculation of prediction interval
        inv_xtx = self.model.normalized_cov_params  # matrice (X'X)-1
        X_pred = data_exog.as_matrix()
        std_err = np.zeros((n_predictions,))
        for i in range(n_predictions):
            tmp = X_pred[i,:]
            pm = np.dot(np.dot(tmp,inv_xtx),np.transpose(tmp))
            std_err[i] = np.sqrt(self.regression.scale * (1 + pm))
        #quantile of the Student distribution
        alpha = 1 - confidence_interval
        qt = st.t.ppf(1 - (alpha / 2), df=n-p-1)
        low_predictions = predictions - (qt * std_err)
        high_predictions = predictions + (qt * std_err)
        data_output = data_input[self.variables_retenues].copy()
        data_output["prediction"] = predictions
        data_output["prediction_basse"] = low_predictions
        data_output["prediction_hautee"] = high_predictions
        return data_output
        
    def calcul_predictions(self, data_input, confidence_interval=0.95):
        if self.scaler is None:
            data_exog = data_input[self.variables_retenues].copy()
        else:
            interX = data_input[self.variables_retenues].copy()
            data_exog = pd.DataFrame(data=self.scaler.transform(interX), index=data_input.index, columns=self.variables_retenues)
        if self.with_constant:
            data_exog = sm.add_constant(data_exog, has_constant='add')
        predictions = self.regression.get_prediction(data_exog)
        predictions_interval = predictions.conf_int(obs=True, alpha=1-confidence_interval)
        data_output = data_input[self.variables_retenues].copy()
        data_output["prediction"] = predictions.predicted_mean
        data_output["prediction_basse"] = predictions_interval[:,0]
        data_output["prediction_haute"] = predictions_interval[:,1]
        return data_output
    
    
class LinearRegressionPrediction:
    
    def __init__(self, data, y_name, array_X_names, with_constant=True, with_standardization=True):
        self.variable_estimee = y_name
        self.variables_explicatives = array_X_names
        self.y = np.array(data[self.variable_estimee].copy())
        if with_standardization:
            interX = np.array(data[self.variables_explicatives].copy())
            self.scaler = preprocessing.StandardScaler().fit(interX)
            self.X = self.scaler.transform(interX)
        else:
            self.scaler = None
            self.X = np.array(data[self.variables_explicatives].copy())
        self.with_constant = with_constant
        self.model = LinearRegression(fit_intercept=self.with_constant)
        self.regression = self.model.fit(self.X, self.y)
        self.y_estimates = self.regression.predict(self.X)
        self.variance_erreur_estim = None
        self.covariance_matrix = None
        self.calcul_covariance_matrix()
        self.pred_input = None
        self.pred_output = None
        
    def cross_validation(self, n_splits=5):
        kfold = model_selection.KFold(n_splits=n_splits)
        results = model_selection.cross_val_score(self.model, self.X, self.y, cv=kfold)
        dfresults = pd.DataFrame(data=[results], columns=["{}".format(cpt) for cpt in np.arange(n_splits)])
        dfresults["Moyenne"] = results.mean()
        dfresults["Ecart-type"] = results.std(ddof=1)
        return dfresults
        
    def calcul_covariance_matrix(self):
        n, p =  self.X.shape
        my_x = np.hstack((np.ones((n, 1)), self.X))
        self.covariance_matrix = np.linalg.inv(np.dot(my_x.T, my_x))
        self.variance_erreur_estim = np.sum((self.y - self.y_estimates) ** 2) / (n - p - 1)
        
    def calcul_error_prediction(self, data_exog):
        n_predictions = len(data_exog)
        inv_xtx = self.covariance_matrix  # matrice (X'X)-1
        X_predictions = np.hstack((np.ones((n_predictions, 1)), data_exog))
        std_err = np.zeros((n_predictions,))
        for i in range(n_predictions):
            tmp = X_predictions[i,:]
            pm = np.dot(np.dot(tmp, inv_xtx), tmp.T)
            std_err[i] = np.sqrt(self.variance_erreur_estim * (1 + pm))
        return std_err
        
    def calcul_prediction(self, data_input, confidence_interval=0.95, with_interval=True):
        if self.scaler is None:
            self.pred_input = np.array(data_input[self.variables_explicatives].copy())
        else:
            interX = np.array(data_input[self.variables_explicatives].copy())
            self.pred_input = self.scaler.transform(interX)
        y_predictions = self.regression.predict(self.pred_input)
        self.pred_output = data_input[self.variables_explicatives].copy()
        self.pred_output["prediction"] = y_predictions
        if with_interval:
            n, p =  self.X.shape
            std_err = self.calcul_error_prediction(self.pred_input)
            alpha = 1 - confidence_interval
            qt = st.t.ppf(1 - (alpha / 2), df=n-p-1)
            low_predictions = y_predictions - (qt * std_err)
            high_predictions = y_predictions + (qt * std_err)
            self.pred_output["prediction_basse"] = low_predictions
            self.pred_output["prediction_hautee"] = high_predictions
        return self.pred_output
    
    def calcul_score_prediction(self, data_input, true_output):
        #data_input = np.array(pred_input)
        self.calcul_prediction(data_input, with_interval=False)
        target_output = np.array(true_output)
        return self.regression.score(self.pred_input, target_output)
    