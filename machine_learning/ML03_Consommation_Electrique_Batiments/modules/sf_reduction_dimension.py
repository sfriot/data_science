# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 22:05:59 2020

@author: Sylvain Friot

Content : classification and acp - analyses and graphs
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import decomposition
import logging as lg
from IPython.core import display as ICD
import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    import sf_graphiques as sfg
else:
    import modules_perso.sf_graphiques as sfg


"""
Class : ACP
Generates principal component analysis and its attached graphs
Attributes
    data : dataframe with data
    data_size : number of observations in the sample
    data_values : data to classify
    data_names : names of observations in the data dataframe (index of data)
    data_variables : names of variables in the data dataframe (columns of data)
    with_reduction : indicates if reduction is necessary (False by default because scaling should be done previously in the Pipeline)
    data_scaled : centered and reduced data values
    n_components : number of components of the PCA
    pca : principal components
    variables_projected : projection of variables on principal components (correlation circles)
    data_projected : projection of members on principal components (factorial planes)
    df_variables_projected : dataframe with features coefficients used to project data on the principal inertia axes
    df_data_projected : dataframe with the coordinates of members on the principal inertia axes
    df_valeurs_propres : dataframe with explained_variance and explained_variance_ratio for each principal component axe
    df_centroids_projected : dataframe with the coordinates of centroids on the principal inertia axes
Methods
    calculate_kmeans : calculates the clusters for different numbers of clusters, and stores the results in OneKmean objects
    graphique_choix_nclusters : graph to help to choose the number of clusters, based on the inertia
    define_n_components : sets or changes the number of clusters
    calculate_clusters_centroids : fills in the 3 dataframes df_clusters, df_centroids and df_data_clusters (automatically called by define_n_clusters)
"""
class ACP:

    def __init__(self, df_data, n_components=None, with_reduction=False):
        self.data = df_data.select_dtypes([np.number, np.bool_])
        if n_components is None:
            self.n_components = min(self.data.shape[0]-1, self.data.shape[1])
        else:
            self.n_components = n_components
            if n_components < 2:
                self.n_components = 2
            elif (n_components > self.data.shape[0] - 1) | (n_components > self.data.shape[1]):
                self.n_components = min(self.data.shape[0] - 1, self.data.shape[1])
        self.with_reduction = with_reduction
        if with_reduction:
            self.scaler = preprocessing.StandardScaler().fit(self.data.values)
            self.data_scaled = self.scaler.transform(self.data.values)
        else:
            self.scaler = None
            self.data_scaled = self.data.values
        self.pca = None
        self.variables_projected = None
        self.data_projected = None
        self.valeurs_propres = None
        self.variables_correlation = None
        self.variables_qualite_representation = None
        self.variables_contribution = None
        self.individus_inertie = None
        self.individus_qualite_representation = None
        self.individus_contribution = None
        self.batons_brises = None
        self.calculate_acp()
        
    def print_etude_variables(self, n_variables=None, n_components=None):
        if n_components is None:
            n_components = self.n_components
        if n_variables is None:
            n_variables = self.data.shape[1]
        ICD.display(self.variables_correlation.iloc[:n_variables,:n_components].style.format("{:.3f}").set_caption("Corrélation des variables avec les axes de l'ACP"))
        ICD.display(self.variables_qualite_representation.iloc[:n_variables,:n_components].style.format("{:.3f}").set_caption("Qualité de représentation des variables sur les axes de l'ACP"))
        ICD.display(self.variables_contribution.iloc[:n_variables,:n_components].style.format("{:.3f}").set_caption("Contribution relative des variables aux axes de l'ACP"))
    
    def print_etude_individus(self, n_individus=100, n_components=None):
        if n_components is None:
            n_components = self.n_components
        if n_individus is None:
            n_individus = self.data.shape[0]
        ICD.display(self.individus_inertie.iloc[:n_individus].style.format("{:.3f}").set_caption("Contribution des individus à l'inertie totale"))
        ICD.display(self.individus_qualite_representation.iloc[:n_individus,:n_components].style.format("{:.3f}").set_caption("Qualité de représentation des individus sur les axes de l'ACP"))
        ICD.display(self.individus_contribution.iloc[:n_individus,:n_components].style.format("{:.3f}").set_caption("Contribution des individus aux axes de l'ACP"))

    def get_n_seuil_inertie(self, seuil_inertie=0.90):
        #return np.searchsorted(self.pca.explained_variance_ratio_.cumsum(), seuil_inertie) + 1
        return np.argmax(np.cumsum(self.pca.explained_variance_ratio_) >= seuil_inertie) + 1
    
    def get_n_batons_brises(self):
        return np.argmax(np.array(self.batons_brises.explained_variance < self.batons_brises.seuils))
        
    """
    Sets the choosen number of components
    Call the calculation of the PCA and of the 3 output dataframes
    """
    def define_n_components(self, n_components):
        if n_components < 2:
            n_components = 2
        elif (n_components > self.data.shape[0] - 1) | (n_components > self.data.shape[1]):
            n_components = min(self.data.shape[0] - 1, self.data.shape[1])
        self.n_components = n_components
        self.calculate_acp()
            
    """
    Calculates the 3 output dataframes : variables_projected, data_projected, valeurs_propres
    """
    def calculate_acp(self):
        correct_valeurs_propres = (self.data.shape[0] - 1.0) / self.data.shape[0]
        self.pca = decomposition.PCA(n_components=self.n_components) # calcul des composantes principales
        self.pca.fit(self.data_scaled)
        self.variables_projected = pd.DataFrame(self.pca.components_, \
            index=["F{} ({:.1%})".format(i+1, self.pca.explained_variance_ratio_[i]) for i in np.arange(self.n_components)], \
            columns=self.data.columns)  # coefficients de combinaison linéaire des variables pour déterminer la projection des individus sur les axes d'inertie
        self.data_projected = pd.DataFrame(self.pca.transform(self.data_scaled), \
            index=self.data.index, \
            columns=["F{} ({:.1%})".format(i+1, self.pca.explained_variance_ratio_[i]) for i in np.arange(self.n_components)])  # projection des individus sur les axes d'inertie
        self.valeurs_propres = pd.DataFrame(data={"explained_variance" : correct_valeurs_propres * self.pca.explained_variance_, \
            "explained_variance_ratio" : self.pca.explained_variance_ratio_}, \
            index=["F{}".format(i+1) for i in np.arange(self.n_components)])
        self.calcul_variables_interpretation()
        self.calcul_individus_interpretation()
        seuils_batons_brises = 1 / np.arange(self.n_components, 0 , -1)
        seuils_batons_brises = np.cumsum(seuils_batons_brises)
        seuils_batons_brises = seuils_batons_brises[::-1]
        self.batons_brises = self.valeurs_propres[["explained_variance"]].copy()
        self.batons_brises["seuils"] = seuils_batons_brises
        
    def calcul_variables_interpretation(self):
        p = self.data.shape[1]
        corvar = np.zeros((p, self.n_components))
        sqrt_eigval = np.sqrt(self.valeurs_propres.explained_variance)
        for k in np.arange(self.n_components):
            corvar[:,k] = self.pca.components_[k,:] * sqrt_eigval[k]  # correlation des variables avec les axes COR
        cos2var = corvar**2  # qualité de représentation des variables COS²
        ctrvar = corvar**2
        for k in np.arange(self.n_components):
            ctrvar[:,k] = ctrvar[:,k] / self.valeurs_propres.explained_variance[k]  # contribution relative des variables aux axes CTR
        self.variables_correlation = pd.DataFrame(data=corvar, index=self.data.columns, columns=["COR F{}".format(i+1) for i in np.arange(self.n_components)])
        self.variables_qualite_representation = pd.DataFrame(data=cos2var, index=self.data.columns, columns=["COS² F{}".format(i+1) for i in np.arange(self.n_components)])
        self.variables_contribution = pd.DataFrame(data=ctrvar, index=self.data.columns, columns=["CTR F{}".format(i+1) for i in np.arange(self.n_components)])
        
    def calcul_individus_interpretation(self):
        carre_distances = np.sum(self.data_scaled**2, axis=1)  # contrib des individus à l'inertie totale (carrés des distances à l'origine)
        cos2 = self.data_projected.values**2
        for j in np.arange(self.n_components):
            cos2[:,j] = cos2[:,j] / carre_distances  # qualité de représentation des individus (cos2) sur chaque axe
        ctr = self.data_projected.values**2
        for j in np.arange(self.n_components):
            ctr[:,j] = ctr[:,j] / (len(self.data) * self.valeurs_propres.explained_variance[j])  # contribution des individus aux axes CTR
        self.individus_inertie = pd.DataFrame(data=carre_distances, index=self.data.index, columns=["contrib_inertie"])
        self.individus_qualite_representation = pd.DataFrame(data=cos2, index=self.data.index, columns=["COS² F{}".format(i+1) for i in np.arange(self.n_components)])
        self.individus_contribution = pd.DataFrame(data=ctr, index=self.data.index, columns=["CTR F{}".format(i+1) for i in np.arange(self.n_components)])
        
        
    """
    Generate a dataframe with coordinates of data projected on principal components axes
    """
    def projection_data(self, df_input):
        num_data = df_input.select_dtypes([np.number, np.bool_])
        if self.with_reduction:
            input_data_scaled = self.scaler.transform(num_data.values)
        else:
            input_data_scaled = num_data.values
        input_projected = pd.DataFrame(self.pca.transform(input_data_scaled), \
                                       index=num_data.index, \
                                       columns=["F{} ({:.1%})".format(i+1, self.pca.explained_variance_ratio_[i]) for i in np.arange(self.n_components)])
        return input_projected

    """
    Generate a dataframe with coordinates of centroids projected on principal components axes
    Parameter
        df_clusters : dataframe df_clusters generated by hierarchical classification or k-means classification
    Returned data
        df_centroids_projected : dataframe df_centroids_projected with the coordinates of centroids principal components axes for each cluster
    """
    def projection_centroids(self, df_clusters, df_data_projected=None):
        if df_data_projected is None:
            df_data_projected = self.data_projected.copy()
        df_data_projected["cluster"] = df_clusters.cluster
        centroids_projected = df_data_projected.groupby("cluster").mean()
        centroids_projected.index = ["cluster {}".format(i) for i in centroids_projected.index]
        return centroids_projected
        
    """
    Graph to choose the number of components : explained variance ratio (éboulis des valeurs propres) and broken sticks (batons brisés)
    Parameters
        figsize : size of the figure
        show_gridlines : to show or hide gridlines
    Returned data
        Matplotlib fig and ax
    """
    def graphique_choix_n(self, seuil_inertie=0.90, title=None, with_grid='y', grid_style=':', figsize=(12,8)):
        if title is None:
            title = "Choix du nombre n de composantes principales"
        graph = sfg.MyGraph(title, nbcol=2, nblin=1, figsize=(12,8))
        graph.graph_eboulis_valeurspropres(self.pca.explained_variance_ratio_, seuil_inertie=seuil_inertie, with_grid=with_grid, grid_style=grid_style, subtitle="Eboulis des valeurs propres", multi_index=1)
        graph.graph_batons_brises(self.batons_brises, with_grid=with_grid, grid_style=grid_style, subtitle="Seuils des bâtons brisés", multi_index=2)
        graph.fig.tight_layout()
        return graph.fig, graph.ax
        
    """
    Graph of correlation circles : projection of variables on the principal components axes
    Parameters
        axis_ranks : list of the two axes to graph (0-based)
        show_labels : to choose to print or not the name of the variables
        label_rotation : label rotation angle in degree 
        figsize : size of the figure
        show_gridlines : to show or hide gridlines
        lims : None (default graphs with all variables) or list of 4 coordinates to zoom in or out (xmin, xmax, ymin, ymax)
    Returned data
        Matplotlib fig and ax
    """
    def graphique_correlation_circles(self, axis_ranks=[1,2], nbcol=None,
                                      show_labels=True, labels_fontsize=None, labels_rotation=0, labels_color='blue', labels_alpha=0.5, \
                                      color='grey', alpha=1, limits=None, axis_linestyle='--', axis_color='grey', 
                                      with_grid='both', grid_style=':', y_title=1.0, figsize=None):
        if np.max(axis_ranks) > self.n_components:
            lg.warning("Le rang de l'axe ne peut pas être supérieur au nombre de composantes principales")
            return np.nan, np.nan
        axis_ranks = np.array(axis_ranks)
        if axis_ranks.ndim > 1:
            nb_graphes = axis_ranks.shape[0]
            title = "Cercle des corrélations"
            if nbcol is None:
                nbcol = 2
            nblin = (nb_graphes // nbcol) + (nb_graphes % nbcol)
            if figsize is None:
                figsize = (12, (6 * nblin))
        else:
            nb_graphes = 1
            title = "Cercle des corrélations sur les axes F{} et F{}".format(axis_ranks[0], axis_ranks[1])
            nbcol = 1
            nblin = 1
            if figsize is None:
                figsize = (12,12)
        graph = sfg.MyGraph(title, y_title=y_title, nblin=nblin, nbcol=nbcol, figsize=figsize)        
        if nb_graphes == 1:
            graph.graph_correlation_circles(self.variables_correlation, self.variables_projected.index, idx1=axis_ranks[0]-1, idx2=axis_ranks[1]-1, show_labels=show_labels, \
                                            labels_fontsize=labels_fontsize, labels_rotation=labels_rotation, labels_color=labels_color, labels_alpha=labels_alpha, \
                                            color=color, alpha=alpha, limits=limits, axis_linestyle=axis_linestyle, axis_color=axis_color, with_grid=with_grid, grid_style=grid_style)
        else:
            for g in range(nb_graphes):
                graph.graph_correlation_circles(self.variables_correlation, self.variables_projected.index, idx1=axis_ranks[g,0]-1, idx2=axis_ranks[g,1]-1, show_labels=show_labels, \
                                                labels_fontsize=labels_fontsize, labels_rotation=labels_rotation, labels_color=labels_color, labels_alpha=labels_alpha, \
                                                color=color, alpha=alpha, limits=limits, axis_linestyle=axis_linestyle, axis_color=axis_color, with_grid=with_grid, grid_style=grid_style, \
                                                subtitle = "Axes F{} et F{}".format(axis_ranks[g,0], axis_ranks[g,1]), multi_index=g+1)
            if nb_graphes % 2 == 1:
                graph.ax[nb_graphes].set_axis_off()
        graph.fig.tight_layout()
        return graph.fig, graph.ax
    
    def zoom_correlation_circle(self, zooms=[-0.5,0.5,-0.5,0.5], axis1=1, axis2=2, nbcol=None, show_labels=True, labels_fontsize=None, labels_rotation=0, labels_color='blue', labels_alpha=0.5, \
                                      color='grey', alpha=1, axis_linestyle='--', axis_color='grey', with_grid='both', grid_style=':', y_title=1.025, figsize=None):
        zooms = np.array(zooms)
        if zooms.ndim > 1:
            nb_graphes = zooms.shape[0]
            if nbcol is None:
                nbcol = 2
            nblin = (nb_graphes // nbcol) + (nb_graphes % nbcol)
            if figsize is None:
                figsize = (12, (6 * nblin))
        else:
            nb_graphes = 1
            nbcol = 1
            nblin = 1
            if figsize is None:
                figsize = (12,12)
        graph = sfg.MyGraph("Zoom sur le cercle des corrélations sur les axes F{} et F{}".format(axis1, axis2), y_title=y_title, nblin=nblin, nbcol=nbcol, figsize=figsize)
        if nb_graphes == 1:
            graph.graph_correlation_circles(self.variables_correlation, self.variables_projected.index, idx1=axis1-1, idx2=axis2-1, show_labels=show_labels, \
                                            labels_fontsize=labels_fontsize, labels_rotation=labels_rotation, labels_color=labels_color, labels_alpha=labels_alpha, \
                                            color=color, alpha=alpha, limits=zooms, axis_linestyle=axis_linestyle, axis_color=axis_color, with_grid=with_grid, grid_style=grid_style)
        else:
            for g in range(nb_graphes):
                graph.graph_correlation_circles(self.variables_correlation, self.variables_projected.index, idx1=axis1-1, idx2=axis2-1, show_labels=show_labels, \
                                                labels_fontsize=labels_fontsize, labels_rotation=labels_rotation, labels_color=labels_color, labels_alpha=labels_alpha, \
                                                color=color, alpha=alpha, limits=zooms[g,:], axis_linestyle=axis_linestyle, axis_color=axis_color, with_grid=with_grid, grid_style=grid_style, multi_index=g+1)
            if nb_graphes % 2 == 1:
                graph.ax[nb_graphes].set_axis_off()
        graph.fig.tight_layout()
        return graph.fig, graph.ax

    """
    Graph of factorial planes : projection of members on the principal components axes
    Parameters
        axis_ranks : list of the two axes to graph (0-based)
        show_labels : to choose to print or not the name of the members
        alpha : alpha of scatter points color
        illustrative_var : list of the same length as member to color scatter points according to a category (example: clusters)
        illustrative_legend : title for legend (used only if illustrative_var is not None)
        figsize : size of the figure
        lims : None (default graphs with all variables) or list of 4 coordinates to zoom in or out (xmin, xmax, ymin, ymax)
    Returned data
        Matplotlib fig and ax
    """
    def graphique_data_projected(self, df_data_projected, axis_ranks=[1,2], nbcol=None, marker='x', marker_alpha=0.75, marker_color=None, show_labels=False, labels_fontsize=None, labels_rotation=0, labels_color="blue", \
                                 labels_alpha=0.5, hue=None, hue_legend_title=None, hue_color_base_index=None, limits=None, axis_linestyle='--', axis_color='grey', with_grid='both', grid_style=':', y_title=1.025, figsize=None):
        if np.max(axis_ranks) > self.n_components:
            lg.warning("Le rang de l'axe ne peut pas être supérieur au nombre de composantes principales")
            return np.nan, np.nan
        axis_ranks = np.array(axis_ranks)
        if axis_ranks.ndim > 1:
            nb_graphes = axis_ranks.shape[0]
            title = "Projection des données"
            if nbcol is None:
                nbcol = 2
            nblin = (nb_graphes // nbcol) + (nb_graphes % nbcol)
            if figsize is None:
                figsize = (12, (6 * nblin))
        else:
            nb_graphes = 1
            title = "Projection des données sur les axes F{} et F{}".format(axis_ranks[0], axis_ranks[1])
            nbcol = 1
            nblin = 1
            if figsize is None:
                figsize = (12,12)
        graph = sfg.MyGraph(title, y_title=y_title, nblin=nblin, nbcol=nbcol, figsize=figsize)        
        if nb_graphes == 1:
            graph.graph_factorial_planes(df_data_projected, marker=marker, marker_alpha=marker_alpha, marker_color=marker_color, idx1=axis_ranks[0]-1, idx2=axis_ranks[1]-1, \
                                         show_labels=show_labels, labels_fontsize=labels_fontsize, labels_rotation=labels_rotation, labels_color=labels_color, labels_alpha=labels_alpha, \
                                         hue=hue, hue_legend_title=hue_legend_title, hue_color_base_index=hue_color_base_index, limits=limits, \
                                         axis_linestyle=axis_linestyle, axis_color=axis_color, with_grid=with_grid, grid_style=grid_style)
        else:
            for g in range(nb_graphes):
                graph.graph_factorial_planes(df_data_projected, marker=marker, marker_alpha=marker_alpha, marker_color=marker_color, idx1=axis_ranks[g,0]-1, idx2=axis_ranks[g,1]-1, \
                                             show_labels=show_labels, labels_fontsize=labels_fontsize, labels_rotation=labels_rotation, labels_color=labels_color, labels_alpha=labels_alpha, \
                                             hue=hue, hue_legend_title=hue_legend_title, hue_color_base_index=hue_color_base_index, limits=limits, \
                                             axis_linestyle=axis_linestyle, axis_color=axis_color, with_grid=with_grid, grid_style=grid_style, \
                                             subtitle = "Axes F{} et F{}".format(axis_ranks[g,0], axis_ranks[g,1]), multi_index=g+1)
            if nb_graphes % 2 == 1:
                graph.ax[nb_graphes].set_axis_off()
        graph.fig.tight_layout()
        return graph.fig, graph.ax
    
    def zoom_data_projected(self, df_data_projected, zooms=[-0.5,0.5,-0.5,0.5], axis1=1, axis2=2, nbcol=None, marker='x', marker_alpha=0.75, marker_color=None, show_labels=False, labels_fontsize=None, labels_rotation=0, labels_color="blue", \
                                 labels_alpha=0.5, hue=None, hue_legend_title=None, hue_color_base_index=None, axis_linestyle='--', axis_color='grey', with_grid='both', grid_style=':', y_title=1.025, figsize=None):
        zooms = np.array(zooms)
        if zooms.ndim > 1:
            nb_graphes = zooms.shape[0]
            if nbcol is None:
                nbcol = 2
            nblin = (nb_graphes // nbcol) + (nb_graphes % nbcol)
            if figsize is None:
                figsize = (12, (6 * nblin))
        else:
            nb_graphes = 1
            nbcol = 1
            nblin = 1
            if figsize is None:
                figsize = (12,12)
        graph = sfg.MyGraph("Zoom sur la projection des données sur les axes F{} et F{}".format(axis1, axis2), y_title=y_title, nblin=nblin, nbcol=nbcol, figsize=figsize)        
        if nb_graphes == 1:
            graph.graph_factorial_planes(df_data_projected, marker=marker, marker_alpha=marker_alpha, marker_color=marker_color, idx1=axis1-1, idx2=axis2-1, \
                                         show_labels=show_labels, labels_fontsize=labels_fontsize, labels_rotation=labels_rotation, labels_color=labels_color, labels_alpha=labels_alpha, \
                                         hue=hue, hue_legend_title=hue_legend_title, hue_color_base_index=hue_color_base_index, limits=zooms, \
                                         axis_linestyle=axis_linestyle, axis_color=axis_color, with_grid=with_grid, grid_style=grid_style)
        else:
            for g in range(nb_graphes):
                graph.graph_factorial_planes(df_data_projected, marker=marker, marker_alpha=marker_alpha, marker_color=marker_color, idx1=axis1-1, idx2=axis2-1, \
                                             show_labels=show_labels, labels_fontsize=labels_fontsize, labels_rotation=labels_rotation, labels_color=labels_color, labels_alpha=labels_alpha, \
                                             hue=hue, hue_legend_title=hue_legend_title, hue_color_base_index=hue_color_base_index, limits=zooms[g,:], \
                                             axis_linestyle=axis_linestyle, axis_color=axis_color, with_grid=with_grid, grid_style=grid_style, multi_index=g+1)
            if nb_graphes % 2 == 1:
                graph.ax[nb_graphes].set_axis_off()
        graph.fig.tight_layout()
        return graph.fig, graph.ax
