# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:38:16 2019

@author: Sylvain Friot

Content: one qualitative variable explained by quantitative variables - analysis and prediction
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
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy import interp

import sf_graphiques as sfg


def readme(classe=None):
    if classe is None:
        print("Liste des classes : LogisticRegressionAnalysis, LogisticRegressionPrediction")
    if classe == "LogisticRegressionAnalysis":
        pass
    if classe == "LogisticRegressionPrediction":
        pass
    
    
"""
"""
class TrainTestData:
    
    def __init__(self, save_name, data, y_name, train_percentage=0.75):
        self.train = None
        self.test = None
        self.get_split(save_name, data, y_name, train_percentage)
        
    def get_split(self, save_name, data, y_name, train_percentage):
        try:
            saved_indexes = pd.read_csv(save_name, index_col="index")
            self.train = data[~saved_indexes.test].copy()
            self.test = data[saved_indexes.test].copy()
            print("Train-test split existant chargé")
        except:
            self.train, self.test = model_selection.train_test_split(data, test_size=1.0-train_percentage, shuffle=True, stratify=data[y_name])
            inter_test = np.isin(data.index, list(self.test.index))
            saved_indexes = pd.DataFrame({'test':inter_test}, index=data.index)
            saved_indexes.to_csv(save_name, index_label="index")
            print("Nouveau train-test split sauvegardé")


class GenericFunctions:
    
    def graph_confusion_matrix(self, y_true, y_pred, pos_neg_labels, normalize=False, title=None, cmap=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        adapted from : https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        """
        if title is None:
            if normalize:
                title = "Heatmap de la matrice de confusion normalisée"
            else:
                title = "Heatmap de la matrice de confusion"
        cm = metrics.confusion_matrix(np.array(y_true), np.array(y_pred), labels=pos_neg_labels)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            vmin = 0
            vmax = 1
        else:
            fmt = 'd'
            vmin = cm.sum().sum()
            vmax = 0
        mygraph = sfg.MyGraph(title=title, is_mono=True)
        if cmap is None:
            cmap = plc.LinearSegmentedColormap.from_list("", [mygraph.liste_couleurs[0],mygraph.liste_couleurs[-1]])
        sns.heatmap(cm, vmin=vmin, vmax=vmax, fmt=fmt, cmap=cmap, annot=True, ax=mygraph.ax[0])
        mygraph.set_axe('y', label="Vraie catégorie", tick_labels=pos_neg_labels, rotation=0)
        mygraph.set_axe('x', label="Catégorie prédite", tick_labels=pos_neg_labels, rotation=30, ha='right')
        mygraph.fig.tight_layout()
        return mygraph.fig, mygraph.ax


"""
Class : LogisticRegressionAnalysis
Runs linear regression analysis to select explanatory models and prediction models
Attributes
    
Methods
    
"""
class LogisticRegressionAnalysis(GenericFunctions):
    
    def __init__(self, data, y_name, array_X_names, pos_neg_labels=[True, False], with_constant=True, method='lbfgs', with_standardization=True):
        GenericFunctions.__init__(self)
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
        self.with_constant = with_constant
        self.pos_neg_labels = pos_neg_labels
        self.method = method
        self.model = None
        self.classification = None
        self.df_beta_estimates = None
        self.equation = None
        self.threshold = None
        self.df_y_estimates = None
        self.residus = None
        self.df_matrice_confusion = None
        self.df_stats = None
        self.covariance_matrix = None
        self.observations_analyses = None
        self.observations_seuils = None
        self.observations_to_remove = None
        self.run_classification()
        
    def re_init_variables_retenues(self):
        self.variables_retenues = self.variables_explicatives
        self.run_classification()
        
    def remove_from_variables_retenues(self, var_name):
        self.variables_retenues.remove(var_name)
        self.run_classification()
        
    def remove_from_data(self, index_list):
        self.rawdata = self.rawdata[~self.rawdata.index.isin(index_list)]
        self.y = self.rawdata[self.variable_estimee].copy()
        if self.scaler is None:
            self.X = self.rawdata[self.variables_explicatives].copy()
        else:
            interX = self.rawdata[self.variables_explicatives].copy()
            self.scaler = preprocessing.StandardScaler().fit(interX)
            self.X = pd.DataFrame(data=self.scaler.transform(interX), index=self.rawdata.index, columns=self.variables_explicatives)
        self.run_classification()
        
    def run_classification(self, threshold=0.5, alpha=0.05, add_constant=None, do_print=True):
        my_y = self.y.copy()
        my_X = self.X[self.variables_retenues].copy()
        n, p = my_X.shape
        if add_constant is not None:
            self.with_constant = add_constant
        if self.with_constant:
            my_X = sm.add_constant(my_X)
        self.model = sm.Logit(my_y, my_X)
        self.classification = self.model.fit(disp=False, method=self.method)
        self.df_y_estimates = pd.DataFrame(np.nan, index=my_y.index, columns=["logit","proba","binaire"])
        self.df_y_estimates["logit"] = self.classification.fittedvalues
        self.df_y_estimates["proba"] = self.classification.predict()
        self.calc_binaire_estimates(threshold)
        self.residus = self.classification.resid_response
        self.df_beta_estimates = pd.DataFrame(np.nan, index=["odd_ratio","beta_estimates","adjusted_beta_absolute","adjusted_beta_relative"], columns=self.variables_retenues)
        logit_std = self.df_y_estimates.logit.std(ddof=1)
        self.equation = "{} =".format(self.variable_estimee)
        for variable in self.variables_retenues:
            coef = self.classification.params[variable]
            x_std = my_X[variable].std(ddof=1)
            self.df_beta_estimates.loc["odd_ratio",variable] = np.exp(coef)
            self.df_beta_estimates.loc["beta_estimates",variable] = coef
            self.df_beta_estimates.loc["adjusted_beta_absolute",variable] = coef * x_std
            self.df_beta_estimates.loc["adjusted_beta_relative",variable] = coef * x_std / logit_std
            self.equation = self.equation + " {:.3f} * {} +".format(coef, variable)
        if self.with_constant:
            coef = self.classification.params["const"]
            self.df_beta_estimates["constante"] = [np.nan, coef, np.nan, np.nan]
            self.equation = self.equation + " {:.3f}".format(coef)
        else:
            self.equation = self.equation[:-2]
        self.calc_confusion_matrix_stats(threshold)
        self.calcul_dfanalyse_observations(alpha)
        if do_print:
            print("L'analyse a été calculée")
        
    def calc_binaire_estimates(self, threshold):
        self.threshold = threshold
        #inter_binaire = np.zeros(len(self.df_y_estimates))
        inter_binaire = np.full(len(self.df_y_estimates), self.pos_neg_labels[1])
        inter_binaire[self.df_y_estimates.proba > threshold] = self.pos_neg_labels[0]
        self.df_y_estimates["binaire"] = inter_binaire
        
    def calc_confusion_matrix_stats(self, new_threshold):
        if new_threshold != self.threshold:
            self.calc_binaire_estimates(new_threshold)
        my_y = self.y.copy()
        self.df_matrice_confusion = pd.DataFrame(metrics.confusion_matrix(self.y, self.df_y_estimates.binaire, labels=self.pos_neg_labels), index=["positifs_reels", "negatifs_réels"], columns=["positifs_estimes", "negatifs_estimes"])
        #self.matrice_confusion = pd.DataFrame(metrics.confusion_matrix(y, self.y_binaire_estimates), index=["positifs_reels", "negatifs_réels"], columns=["positifs_estimes", "negatifs_estimes"])
        #self.df_matrice_confusion = pd.DataFrame(self.classification.pred_table(new_threshold), index=["positifs_reels", "negatifs_reels"], columns=["positifs_estimes", "negatifs_estimes"])
        for lin in np.arange(len(self.df_matrice_confusion)):
            self.df_matrice_confusion.loc[self.df_matrice_confusion.index[lin],"total_reels"] = self.df_matrice_confusion.iloc[lin,:].sum()
            self.df_matrice_confusion.loc["total_estimes",self.df_matrice_confusion.columns[lin]] = self.df_matrice_confusion.iloc[:,lin].sum()
        self.df_matrice_confusion.loc["total_estimes","total_reels"] = self.df_matrice_confusion.iloc[0:1,2].sum()
        self.df_stats = pd.Series(np.nan, index=["taux_succes","precision","rappel_sensibilite","specificite","f_mesure","indice_youden","rapport_vraisemblance","pseudo_r_squared"])
        self.df_stats.taux_succes = (self.df_matrice_confusion.iloc[0,0] + self.df_matrice_confusion.iloc[1,1]) / len(my_y)
        self.df_stats.precision = self.df_matrice_confusion.iloc[0,0] / (self.df_matrice_confusion.iloc[0,0] + self.df_matrice_confusion.iloc[1,0])
        self.df_stats.rappel_sensibilite = self.df_matrice_confusion.iloc[0,0] / (self.df_matrice_confusion.iloc[0,0] + self.df_matrice_confusion.iloc[0,1])
        self.df_stats.specificite = self.df_matrice_confusion.iloc[1,1] / (self.df_matrice_confusion.iloc[1,0] + self.df_matrice_confusion.iloc[1,1])
        self.df_stats.f_mesure = (2 * self.df_stats.rappel_sensibilite * self.df_stats.precision) / (self.df_stats.rappel_sensibilite + self.df_stats.precision)
        self.df_stats.indice_youden = self.df_stats.rappel_sensibilite + self.df_stats.specificite - 1
        self.df_stats.rapport_vraisemblance = self.df_stats.rappel_sensibilite / (1 - self.df_stats.specificite)
        nb_y_0 = len(my_y[my_y == 0])
        nb_y_1 = len(my_y[my_y == 1])
        erreur_ref = min(nb_y_0, nb_y_1) / len(my_y)
        self.df_stats.pseudo_r_squared = 1 - ((1 - self.df_stats.taux_succes) / erreur_ref)
        self.df_stats = pd.DataFrame(self.df_stats)
        
    def calcul_dfanalyse_observations(self, alpha=0.05):
        n = len(self.y)
        p = len(self.variables_retenues)
        seuils = pd.Series(index=["levier", "residus_pearson", "residus_pearson_standard", "cook","dfbetas"])
        seuils.levier = 2 * (p + 1) / n
        seuils.residus_pearson = st.norm.ppf(1-(alpha/2))
        seuils.residus_pearson_standard = st.norm.ppf(1-(alpha/2))
        seuils.cook = 4 / (n - p - 1)
        seuils.dfbetas = 2 / np.sqrt(n)
        V = np.diagflat(np.array(self.df_y_estimates.proba * (1 - self.df_y_estimates.proba)))
        self.covariance_matrix = np.linalg.inv(np.dot(np.dot(self.model.exog.T, V), self.model.exog))  # (X'VX)-1 = H-1
        hat_matrix = np.dot(np.dot(np.dot(np.dot(np.sqrt(V), self.model.exog), self.covariance_matrix), self.model.exog.T), np.sqrt(V))
        analyses = pd.DataFrame({"observation_name":self.y.index})
        analyses["levier"] = np.diagonal(hat_matrix)
        analyses["residus_pearson"] = self.classification.resid_pearson
        analyses["residus_pearson_standard"] = analyses.residus_pearson / np.sqrt(1 - analyses.levier)
        # levier -> sur les observations. levier = proba[i] * (1-proba)[i] * exog.iloc[i] * H-1 * exog.iloc[i].T  seuil = 2 * (p+1) / n
        analyses["outlier"] = (analyses.levier > seuils.levier) | (abs(analyses.residus_pearson) > seuils.residus_pearson) | (abs(analyses.residus_pearson_standard) > seuils.residus_pearson_standard)
        analyses["cook_distance"] = (analyses.residus_pearson_standard * analyses.residus_pearson_standard / (p + 1)) * (analyses.levier / (1 - analyses.levier))
        analyses["influence"] = (analyses.cook_distance > seuils.cook)
        dfbetas = pd.DataFrame({"observation_name":self.y.index})
        inter_dfbetas = np.dot(self.model.exog, self.covariance_matrix)
        for cpt in np.arange(len(self.classification.model.exog_names)):
            variable = self.classification.model.exog_names[cpt]
            dfbetas[variable] = (inter_dfbetas[:,cpt] / self.classification.bse[variable]) * self.residus / (1 - analyses.levier)
        self.observations_analyses = analyses
        self.observations_seuils = seuils
        self.observations_to_remove = analyses[analyses.outlier & analyses.influence].observation_name.values
        
    def print_analyse_residus(self, alpha=0.05, with_line=True):
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
        print("Dans le cadre de la régression logistique, la variance de l'erreur (= proba_estimee * (1-proba_estimee)) dépend des individus. Il y a hétéroscédasticité.")
        print("")
        fig, ax = self.graph_independance_residus()
        plt.show()
        
    def print_analyse_observations(self, alpha=0.05, graphs=True):
        analyses = self.observations_analyses
        seuils = self.observations_seuils
        if graphs:
            fig, ax = self.graph_outliers_levier(analyses, seuils.levier)
            plt.show()
            print("")
            fig, ax = self.graph_outliers_residus_pearson(analyses, seuils.residus_pearson)
            plt.show()
            print("")
            fig, ax = self.graph_influence_residus_pearson_standard(analyses, seuils.residus_pearson_standard)
            plt.show()
            print("")
            fig, ax = self.graph_influence_cook(analyses, seuils.cook)
            plt.show()
            print("")
            #fig, ax = self.graph_influence_plot(seuils)
            #plt.show()
            #print("")
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
        print(self.classification.summary())
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
        print("\033[1m"+"Matrice des confusions et principales statistiques"+"\033[0m")
        print(self.df_matrice_confusion)
        print("")
        df_affichage = self.df_stats.T.copy()
        for col in np.arange(4):
            df_affichage[df_affichage.columns[col]] = df_affichage[df_affichage.columns[col]].apply("{:.2%}".format)
        for col in np.arange(4, 8):
            df_affichage[df_affichage.columns[col]] = df_affichage[df_affichage.columns[col]].apply("{:.3f}".format)
        print(df_affichage.to_string(index=False))
        #←print(self.df_stats.T.to_string(index=False))
        print("")
        fig, ax = self.graph_confusion_matrix(y_true=self.y, y_pred=self.df_y_estimates.binaire, pos_neg_labels=self.pos_neg_labels, normalize=True)
        plt.show()
        print("")
        fig, ax = self.graph_diagramme_fiabilite()
        plt.show()

    def print_analyse_colinearite(self):
        if len(self.variables_retenues) == 1:
            print("Il n'y a qu'une variable explicative. Aucun problème de colinéarité ne peut se poser.")
        else:
            print("\033[1m"+"ANALYSE DE LA COLINÉARITÉ SIMPLE"+"\033[0m")
            print("Les valeurs de la matrice de corrélation doivent être inférieures à 0.8")
            fig, ax = self.graph_heatmap_correlationmatrix()
            plt.show()
            print("")
            variables = self.classification.model.exog
            vif_analysis = pd.DataFrame([variance_inflation_factor(variables, i) for i in np.arange(1,variables.shape[1])], \
                                      index=[var for var in self.classification.model.exog_names if var!="const"], columns=["VIF"])
            print("\033[1m"+"ANALYSE DE LA MULTI-COLINÉARITÉ"+"\033[0m")
            print("L'indice VIF doit être inférieur à 4")
            print(vif_analysis.applymap("{:.3f}".format))
        
    def graph_diagramme_fiabilite(self, n_groups=10, on_intervals=False, figsize=(12,8)):
        data_fiabilite = pd.DataFrame({self.variable_estimee:self.y})
        data_fiabilite["proba_score"] = self.df_y_estimates.proba
        data_fiabilite.sort_values(by="proba_score", inplace=True)
        moyenne_score = np.zeros(n_groups)
        pourcentage_positive = np.zeros(n_groups)
        for i in np.arange(n_groups):
            if on_intervals:
                min_interval = np.round((i/n_groups) * len(data_fiabilite), 0)
                max_interval = np.round(((i+1)/n_groups) * len(data_fiabilite), 0) - 1
                if i==n_groups:
                    max_interval = max_interval + 1.0
                moyenne_score[i] = data_fiabilite.iloc[min_interval:max_interval,"proba_score"].mean()
                pourcentage_positive[i] = data_fiabilite.iloc[min_interval:max_interval,self.variable_estimee].mean()
            else:
                min_interval = np.quantile(data_fiabilite.proba_score, i/n_groups)
                max_interval = np.quantile(data_fiabilite.proba_score, (i+1)/n_groups)
                if i==n_groups:
                    max_interval = max_interval + 1.0
                moyenne_score[i] = data_fiabilite[(data_fiabilite.proba_score >= min_interval) & (data_fiabilite.proba_score < max_interval)].proba_score.mean()
                pourcentage_positive[i] = data_fiabilite[(data_fiabilite.proba_score >= min_interval) & (data_fiabilite.proba_score < max_interval)][self.variable_estimee].mean()
        mygraph = sfg.MyGraph("Diagramme de fiabilité de la régression logistique", is_mono=False)
        mygraph.add_plot(moyenne_score, pourcentage_positive, label="Alignement des estimations", marker='o', linestyle=":")
        mygraph.add_plot([0,1],[0,1], label="Alignement idéal", legend=True)
        mygraph.set_axe('x', label="Moyenne des probabilités estimées")
        mygraph.set_axe('y', label="Proportion réelle de positifs")
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
        mygraph.add_plot([data_min, data_max], [data_min, data_max], label="", color=mygraph.liste_couleurs[1])
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
    
    def graph_linearite_residuspartiels(self, lissage=True):
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
        mygraph = sfg.MyGraph("Vérification de la linéarité des résidus partiels", nblin=nblin, nbcol=nbcol, is_mono=False, figsize=figsize)
        cpt = 0
        for variable in self.variables_retenues:
            residus_partiels = ((self.y.values - self.df_y_estimates.proba) / (self.df_y_estimates.proba * (1-self.df_y_estimates.proba))) + (self.classification.params[variable] * self.X[variable])
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
    
    def graph_outliers_residus_pearson(self, df_analyse, seuil_residus_pearson):
        mygraph = sfg.MyGraph("Graphique des résidus de Pearson : valeurs atypiques sur la variable estimée", is_mono=False)
        mygraph.add_barv(df_analyse.index, df_analyse.residus_pearson, label="", color=mygraph.liste_couleurs[0], with_grid='both')
        mygraph.add_line(-seuil_residus_pearson, vertical=False, label="Seuil = loi normale", color=mygraph.liste_couleurs[1], legend=True)
        mygraph.add_line(seuil_residus_pearson, vertical=False, label="", color=mygraph.liste_couleurs[1])
        ymin, ymax = mygraph.ax[0].get_ylim()
        absmax = max(-ymin, ymax, 1.1*seuil_residus_pearson)
        mygraph.set_axe('x', label="Observations")
        mygraph.set_axe('y', label="Résidus de Pearson", tick_min=-absmax, tick_max=absmax)
        mygraph.ax[0].autoscale(enable=True, axis='x', tight=True)
        return mygraph.fig, mygraph.ax
        
    def graph_influence_residus_pearson_standard(self, df_analyse, seuil_residus_pearson_standard):
        mygraph = sfg.MyGraph("Graphique des résidus de Pearson standardisés : valeurs atypiques sur la variable estimée", is_mono=False)
        mygraph.add_barv(df_analyse.index, df_analyse.residus_pearson_standard, label="", color=mygraph.liste_couleurs[0], with_grid='both')
        mygraph.add_line(-seuil_residus_pearson_standard, vertical=False, label="Seuil = loi normale", color=mygraph.liste_couleurs[1], legend=True)
        mygraph.add_line(seuil_residus_pearson_standard, vertical=False, label="", color=mygraph.liste_couleurs[1])
        ymin, ymax = mygraph.ax[0].get_ylim()
        absmax = max(-ymin, ymax, 1.1*seuil_residus_pearson_standard)
        mygraph.set_axe('x', label="Observations")
        mygraph.set_axe('y', label="Résidus de Pearson standardisés", tick_min=-absmax, tick_max=absmax)
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
        #mygraph = sfg.MyGraph("Résumé des observations atypiques et influentes", is_mono=False)
        #sm.graphics.influence_plot(self.classification, ax=mygraph.ax[0])
        #mygraph.add_plot(np.NaN, np.NaN, label="Influence : distance de Cook", marker='o', linestyle="", color=mygraph.liste_couleurs[0])
        #ymin, ymax = mygraph.ax[0].get_ylim()
        #absmax = max(-ymin, ymax, 1.1*seuils.residus_pearson)
        #mygraph.set_axe('y', label="Atypicité : résidus studentisés", tick_min=-absmax, tick_max=absmax)
        #xmin, xmax = mygraph.ax[0].get_xlim()
        #absmax = max(xmax, 1.1*seuils.levier)
        #mygraph.set_axe('x', label="Atypicité : leviers", tick_max=absmax)
        #mygraph.add_line(-seuils.residus_pearson, vertical=False, color=mygraph.liste_couleurs[1], label="Seuils d'atypicité")
        #mygraph.add_line(seuils.residus_pearson, vertical=False, color=mygraph.liste_couleurs[1])
        #mygraph.add_line(seuils.levier, vertical=True, color=mygraph.liste_couleurs[1])
        #mygraph.ax[0].set_title("")
        #mygraph.fig.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.975))
        #return mygraph.fig, mygraph.ax
        pass
    
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
        df_affichage["residus_pearson"] = df_affichage["residus_pearson"].apply("{:.3f}".format)
        df_affichage["levier"] = df_affichage["levier"].apply("{:.3f}".format)
        df_affichage["residus_pearson_standard"] = df_affichage["residus_pearson_standard"].apply("{:.3f}".format)
        df_affichage["cook_distance"] = df_affichage["cook_distance"].apply("{:.3f}".format)
        return df_affichage
    
    """
    Logistic model designed by backward selection for regression analysis : parameter with the largest p-value is removed if its p-value is >= alpha
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
            model = sm.Logit(my_y, my_X).fit(disp=False, method=self.method)
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
        self.run_classification(add_constant=add_constant)
        
    """
    Logistic model designed by forward selection for regression analysis, based on the score test : parameter with the largest score stats is added if its p-value is < alpha
    """
    def forward_selection_analysis(self, add_constant=True, alpha=0.05):
        remaining = self.variables_explicatives.copy()
        my_y = np.array(self.y.copy())
        my_X = np.ones((len(my_y), 1))
        selected = []
        cond = True
        while cond:
            regresult = sm.Logit(my_y, my_X).fit(disp=False, method=self.method)
            scores_with_candidate = []
            for candidate in remaining:
                inter_calc = (my_y - np.array(regresult.predict())) * np.array(self.X[candidate])
                gradient_vector_u = np.hstack((np.zeros(my_X.shape[1]), inter_calc.sum()))
                hessian_matrix = np.zeros((my_X.shape[1]+1, my_X.shape[1]+1))
                for lin in np.arange(hessian_matrix.shape[0]):
                    for col in np.arange(hessian_matrix.shape[1]):
                        if col < lin:
                            hessian_matrix[lin, col] = hessian_matrix[col, lin]  # déjà calculé dans une ligne précédente
                        else:
                            if lin == my_X.shape[1]:
                                inter_calc = np.array(self.X[candidate].copy())
                            else:
                                inter_calc = my_X[:,lin].copy()
                            if col == my_X.shape[1]:
                                inter_calc = inter_calc * np.array(self.X[candidate])
                            else:
                                inter_calc = inter_calc * my_X[:,col]
                            inter_calc = inter_calc * np.array(regresult.predict()) * (1 - np.array(regresult.predict()))
                            hessian_matrix[lin, col] = inter_calc.sum()
                score_stat = np.dot(np.dot(gradient_vector_u.T, np.linalg.inv(hessian_matrix)), gradient_vector_u)
                scores_with_candidate.append((score_stat, candidate))
            scores_with_candidate.sort(reverse=True)
            best_score, best_candidate = scores_with_candidate[0]
            if st.chi2.sf(best_score, 1) < alpha:
                action = "Variable ajoutée : {} (p-value du test du score = {:.3f}).".format(best_candidate, st.chi2.sf(best_score, 1))
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                if len(remaining) == 0:
                    action += "\nPlus de variable disponible."
                    cond = False
                else:
                    my_X = np.array(self.X[selected].copy())
                    if add_constant:
                        my_X = sm.add_constant(my_X)
            else:
                action = "Plus de variable significative à ajouter."
                cond = False
            print(action)
            print("")
        liste_variables = "Variables explicatives retenues : " + ", ".join(selected)
        if add_constant:
            liste_variables = liste_variables + ", constante"
        print(liste_variables)
        self.variables_retenues = selected
        self.run_classification(add_constant=add_constant)
    
    
    """
    Logistic model designed by forward selection for regression prediction : variables are added as long as the criteria decreases
    Final step : backward selection on selected variables to check for cross-effects on the criteria
    AIC is the default selection criteria ; if bic=True then BIC becomes the selection criteria
    """
    def forward_selection_prediction(self, add_constant=True, bic=False):
        remaining = self.variables_explicatives.copy()
        my_y = self.y.copy()
        my_X = np.ones((len(my_y),1))
        if bic:
            criteria = "BIC"
            current_score = sm.Logit(my_y, my_X).fit(disp=False, method=self.method).bic
        else:
            criteria = "AIC"
            current_score = sm.Logit(my_y, my_X).fit(disp=False, method=self.method).aic
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
                    score = sm.Logit(my_y, my_X).fit(disp=False, method=self.method).bic
                else:
                    score = sm.Logit(my_y, my_X).fit(disp=False, method=self.method).aic
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
                    score = sm.Logit(my_y, my_X).fit(disp=False, method=self.method).bic
                else:
                    score = sm.Logit(my_y, my_X).fit(disp=False, method=self.method).aic
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
        self.variables_retenues = selected
        self.run_classification(add_constant=add_constant)

    def calcul_predictions_old(self, data_input, confidence_interval=0.95):
        n_predictions = len(data_input)
        if self.scaler is None:
            data_exog = data_input[self.variables_retenues].copy()
        else:
            interX = data_input[self.variables_retenues].copy()
            data_exog = pd.DataFrame(data=self.scaler.transform(interX), index=data_input.index, columns=self.variables_retenues)
        if self.with_constant:
            data_exog = sm.add_constant(data_exog, has_constant='add')
        predictions_logit = self.classification.predict(exog=data_exog, linear=True)  
        # calcul de la std erreur de chaque prédiction
        hessian_inv = self.covariance_matrix
        X_pred = np.array(data_exog)
        std_err = np.zeros((n_predictions,))
        for i in range(n_predictions):
            tmp = X_pred[i,:]
            pm = np.dot(np.dot(tmp, hessian_inv), tmp.T)
            std_err[i] = np.sqrt(pm)
        alpha = 1 - confidence_interval
        qt = st.norm.ppf(1 - (alpha / 2))
        low_predictions_logit = predictions_logit - (qt * std_err)
        high_predictions_logit = predictions_logit + (qt * std_err)
        predictions_proba = 1 / (1 + np.exp(-predictions_logit))
        low_predictions_proba = 1 / (1 + np.exp(-low_predictions_logit))
        high_predictions_proba = 1 / (1 + np.exp(-high_predictions_logit))        
        data_output = data_input[self.variables_retenues].copy()
        data_output["proba_prediction"] = predictions_proba
        data_output["proba_prediction_basse"] = low_predictions_proba
        data_output["proba_prediction_haute"] = high_predictions_proba
        return data_output
        
    def calcul_predictions(self, data_input, confidence_interval=0.95):
        if self.scaler is None:
            data_exog = data_input[self.variables_retenues].copy()
        else:
            interX = data_input[self.variables_retenues].copy()
            data_exog = pd.DataFrame(data=self.scaler.transform(interX), index=data_input.index, columns=self.variables_retenues)
        if self.with_constant:
            data_exog = sm.add_constant(data_exog, has_constant='add')
        predictions = self.classification.get_prediction(data_exog)
        predictions_interval = predictions.conf_int(obs=True, alpha=1-confidence_interval)
        data_output = data_input[self.variables_retenues].copy()
        data_output["prediction"] = predictions.predicted_mean
        data_output["prediction_basse"] = predictions_interval[:,0]
        data_output["prediction_hautee"] = predictions_interval[:,1]
        return data_output
    
    
class LogisticRegressionPrediction(GenericFunctions):
    
    def __init__(self, data, y_name, array_X_names, pos_neg_labels=[True, False], with_constant=True, method='lbfgs', with_standardization=True):
        GenericFunctions.__init__(self)
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
        self.method = method
        self.pos_neg_labels = pos_neg_labels
        self.model = LogisticRegression(fit_intercept=self.with_constant, solver=self.method)
        self.classification = self.model.fit(self.X, self.y)
        self.df_y_estimates = pd.DataFrame(np.nan, index=data.index, columns=["logit","proba","binaire"])
        self.df_y_estimates["logit"] = self.classification.predict_log_proba(self.X)[:,1]
        self.df_y_estimates["proba"] = self.classification.predict_proba(self.X)[:,1]
        self.df_y_estimates["binaire"] = self.classification.predict(self.X)
        self.covariance_matrix = None
        self.calcul_covariance_matrix()
        self.pred_input = None
        self.pred_out = None
        
    def cross_validation(self, n_splits=5):
        kfold = model_selection.StratifiedKFold(n_splits=n_splits)
        results_accuracy = model_selection.cross_val_score(self.model, self.X, self.y, cv=kfold)
        results_f_measure = model_selection.cross_val_score(self.model, self.X, self.y, cv=kfold, scoring='f1')
        results_roc_auc = model_selection.cross_val_score(self.model, self.X, self.y, cv=kfold, scoring='roc_auc')
        results_average_precision = model_selection.cross_val_score(self.model, self.X, self.y, cv=kfold, scoring='average_precision')
        cv_results = pd.DataFrame([results_accuracy, results_f_measure, results_roc_auc, results_average_precision], \
              index=["taux_succes","f_mesure","auc","precision_rappel_score"], columns=["{}".format(cpt) for cpt in np.arange(n_splits)])
        cv_mean = cv_results.T.mean()
        cv_std = cv_results.T.std(ddof=1)
        cv_results["moyenne"] = cv_mean
        cv_results["ecart_type"] = cv_std
        fpr = dict()
        tpr = dict()
        roc_auc = []
        i = -1
        #for i in np.arange(n_splits):
        for train_index, test_index in kfold.split(self.X, self.y):
            i += 1
            fpr[i], tpr[i], temp = metrics.roc_curve(y_true=self.y[train_index], y_score=self.df_y_estimates.iloc[train_index].proba, pos_label=self.pos_neg_labels[0])
            roc_auc.append(metrics.auc(fpr[i], tpr[i]))
        fig_roc, ax_roc = self.graph_roc(fpr, tpr, roc_auc)
        return cv_results, fig_roc, ax_roc
    
    def graph_roc(self, fpr, tpr, roc_auc):
        graph = sfg.MyGraph("Graphique de la courbe ROC", is_mono=False)
        if isinstance(fpr, dict):
            mean_fpr = np.linspace(0, 1, 101)
            mean_tpr = []
            for i in np.arange(len(fpr)):
                mean_tpr.append(interp(mean_fpr, fpr[i], tpr[i]))
                if roc_auc[i] == max(roc_auc):
                    tpr_max = interp(mean_fpr, fpr[i], tpr[i])
                if roc_auc[i] == min(roc_auc):
                    tpr_min = interp(mean_fpr, fpr[i], tpr[i])
            mean_tpr = np.mean(mean_tpr, axis=0)
            mean_tpr[0] = 0
            tpr_max[0] = 0
            tpr_min[0] = 0
            graph.add_plot(mean_fpr, tpr_max, label="AUC de la courbe ROC max = {:.3f}".format(max(roc_auc)), color=graph.liste_couleurs[0])
            graph.add_plot(mean_fpr, tpr_min, label="AUC de la courbe ROC min = {:.3f}".format(min(roc_auc)), color=graph.liste_couleurs[0])
            graph.add_plot(mean_fpr, mean_tpr, label="AUC de la courbe ROC moyenne = {:.3f}".format(np.mean(roc_auc)), color=graph.liste_couleurs[1])
            graph.add_plot([0,1], [0,1], label="Classification aléatoire (ROC = 0,5)", color=graph.liste_couleurs[2], legend=True)
            graph.ax[0].fill_between(mean_fpr, tpr_min, tpr_max, color=graph.liste_couleurs[0], alpha=0.25)
        else:
            graph.add_plot(fpr, tpr, label="AUC de la courbe ROC = {:.3f}".format(roc_auc), color=graph.liste_couleurs[1])
            graph.add_plot([0,1], [0,1], label="Classification aléatoire (ROC = 0,5)", color=graph.liste_couleurs[2], legend=True)
        return graph.fig, graph.ax
        
    def calcul_covariance_matrix(self):
        n, p =  self.X.shape
        my_x = np.hstack((np.ones((n, 1)), self.X))
        interV = self.df_y_estimates.proba * (1 - self.df_y_estimates.proba)
        V = np.diagflat(np.array(interV))
        self.covariance_matrix = np.linalg.inv(np.dot(np.dot(my_x.T, V), my_x))
        
    def calcul_prediction(self, data_input):
        if self.scaler is None:
            self.pred_input = np.array(data_input[self.variables_explicatives].copy())
        else:
            interX = np.array(data_input[self.variables_explicatives].copy())
            self.pred_input = self.scaler.transform(interX)
        self.pred_output = data_input[self.variables_explicatives].copy()
        self.pred_output["log_proba"] = self.classification.predict_log_proba(self.pred_input)[:,1]
        self.pred_output["proba"] = self.classification.predict_proba(self.pred_input)[:,1]
        self.pred_output["prediction"] = self.classification.predict(self.pred_input)
        return self.pred_output
    
    def calcul_score_prediction(self, data_input, true_y):
        model_output = self.calcul_prediction(data_input)
        target_output = np.array(true_y)
        accuracy = metrics.accuracy_score(target_output, model_output.prediction)
        f_measure = metrics.f1_score(target_output, model_output.prediction)
        roc_auc = metrics.roc_auc_score(target_output, model_output.proba)
        average_precision = metrics.average_precision_score(target_output, model_output.proba)
        scores = pd.Series(data=[accuracy,f_measure,roc_auc,average_precision], index=["taux_succes","f_mesure","auc","precision_rappel_score"])
        fig_cm, ax_cm = self.graph_confusion_matrix(true_y, model_output.prediction, pos_neg_labels=self.pos_neg_labels, normalize=False)
        fpr, tpr, temp = metrics.roc_curve(y_true=target_output, y_score=model_output.proba, pos_label=self.pos_neg_labels[0])
        roc_auc = metrics.auc(fpr, tpr)
        fig_roc, ax_roc = self.graph_roc(fpr, tpr, roc_auc)
        return scores, fig_cm, ax_cm, fig_roc, ax_roc


class ProductionLogisticRegression():
    
    def __init__(self, var_estimee, var_explicatives, scaler, with_constant, method, model, classif):
        self.variable_estimee = var_estimee
        self.variables_explicatives = var_explicatives
        self.scaler = scaler
        self.with_constant = with_constant
        self.method = method
        self.model = model
        self.classification = classif
        