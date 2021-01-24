# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:07:10 2020

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import dendrogram
import logging as lg
import sys

IN_COLAB = "google.colab" in sys.modules
PATH_DRIVE = "/content/drive/My Drive/MachineLearning/"

if IN_COLAB:
    import sf_graphiques as sfg
else:
    import modules_perso.sf_graphiques as sfg


def silhouette_analysis(nclusters_list, clusterlabels_list,
                        data_calculation, data_visualisation,
                        min_silhouette=-1, max_silhouette=1,
                        clust_figsize=(12,6), save_figs=False, save_name=None, dpi=300):
    dataviz = data_visualisation.copy()
    for idxgraph in np.arange(len(nclusters_list)):
        nclust = nclusters_list[idxgraph]
        ncolors = nclust
        if ncolors < 8:
            ncolors = 8
        cluster_labels = clusterlabels_list[idxgraph]
        silhouette_avg_score = silhouette_score(data_calculation, cluster_labels)
        sample_silhouette_values = silhouette_samples(data_calculation, cluster_labels)
        graph = sfg.MyGraph("Nombre de clusters = {} - Coefficient de silhouette moyen = {}"\
                            .format(nclust, silhouette_avg_score), nblin=1, nbcol=2,
                            color_palette="hls", ncolors=ncolors, figsize=clust_figsize)
        #graph de gauche : analyse des silhouettes
        y_lower = 10
        for i in range(nclust):
            cluster_silhouette_values = sample_silhouette_values[cluster_labels==i]
            cluster_silhouette_values.sort()
            cluster_size = cluster_silhouette_values.shape[0]
            y_upper = y_lower + cluster_size
            graph.add_area(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                           orientation='horizontal', color=graph.liste_couleurs[i],
                           alpha=0.7, subtitle="Silhouette plot pour chaque cluster")
            graph.add_text(-0.05, y_lower + 0.5 * cluster_size, str(i), backgroundalpha=0)
            y_lower = y_upper + 10  # 10 for the 0 samples
        graph.set_axe_x(label="Valeurs du coefficient de silhouette",
                        tick_min=min_silhouette, tick_max=max_silhouette)
        graph.set_axe_y(label="Clusters")
        graph.add_line(silhouette_avg_score, color="red", linestyle="--")
        #graph de droite : visualisation des clusters
        dataviz["cluster"] = cluster_labels
        clusters_center = dataviz.groupby("cluster").mean()
        graph.add_scatter(dataviz.iloc[:,0], dataviz.iloc[:,1],
                          marker='.', color=cluster_labels, size=30, alpha=0.7, 
                          subtitle="Visualisation des clusters", multi_index=2)
        graph.add_scatter(clusters_center.iloc[:, 0], clusters_center.iloc[:, 1],
                          marker='o', color="white", size=200, multi_index=2)
        for i in range(nclust):
            graph.add_text(clusters_center.iloc[i, 0], clusters_center.iloc[i, 1], str(i), ha="center", va="center", backgroundalpha=0, multi_index=2)
#            graph.add_scatter(clusters_center.iloc[:, 0], clusters_center.iloc[:, 1],
#                              marker="{}".format(i), color=graph.liste_couleurs[i],
#                              size=50, multi_index=2)
        graph.set_axe_x(label="Premier axe de projection des données", multi_index=2)
        graph.set_axe_y(label="Deuxième axe de projection des données", multi_index=2)        
        if save_figs:
            if save_name is None:
                save_name = "silhouette"
            path = save_name + "_{}".format(nclust)
            if IN_COLAB:
                path = PATH_DRIVE + "/" + path
            plt.savefig(path, dpi=dpi)
        plt.show()
    return "done"

"""
Class : MyKmeans
Calculates k-means classification, and a graph to choose the number of clusters
Attributes
    data : dataframe with data
    data_size : number of observations in the sample
    data_values : data to classify
    data_names : names of observations in the data dataframe (index of data)
    data_variables : names of variables in the data dataframe (columns of data)
    with_reduction : indicates if reduction is necessary (True by default)
    data_scaled : centered and reduced data values
    kmeans_list : array of OneKmean objects, to hold results of the kmean calculation for different numbers of clusters
    selected_n_clusters : selected number of clusters
    df_clusters : dataframe with the cluster to which each original observation belongs to (based on selected_n_clusters)
    df_centroids : dataframe with the mean values for the clusters centroids
    df_data_clusters : dataframe with original data and the cluster for each observation
Methods
    calculate_kmeans : calculates the clusters for different numbers of clusters, and stores the results in OneKmean objects
    graphique_choix_nclusters : graph to help to choose the number of clusters, based on the inertia
    define_n_clusters : sets or changes the number of clusters
    calculate_clusters_centroids : fills in the 3 dataframes df_clusters, df_centroids and df_data_clusters (automatically called by define_n_clusters)
"""

"""
TO DO : rajouter analyse avec score de silhouette (cf livre)
        dans le calcul des centroïdes : plus simple et rapide d'utiliser les centroïdes stockés dans le OneKmean de kmeans_list
"""
class MyKmeans:
    
    def __init__(self, data_df, random_state=None):
        self.data = data_df.copy()
#        self.data_size = len(x)
#        self.data_values = x.values
#        self.data_names = x.index
#        self.data_variables = x.columns
        self.random_state = random_state
        self.nclusters_list = None
        self.kmeans_list = None
        self.silhouette_avg_score_list = None
        self.sample_silhouette_values_list = None
        self.selected_n_clusters = None
        self.km = None
        self.df_clusters = None
        self.df_centroids = None

    """
    Calculates the clusters for different numbers of clusters, and stores the results in OneKmean objects
    Parameters
        n_clust_min : minimum number of clusters (>=2 and <=data_size-1)
        n_clust_max : maximum number of clusters (>n_clust_min and <=data_size)
        nb_essais_par_cluster : number of times the kmeans calculation algorythm is run with different starting points
    """
    def choix_n_clusters(self, n_clust_min, n_clust_max, nb_try_per_cluster=10):
        if n_clust_min <= 1:
            n_clust_min = 2
            lg.warning("Warning : n_clust_min must be >1. It was changed to 2.")
        if n_clust_min > len(self.data) - 1:
            n_clust_min = len(self.data) - 1
            lg.warning("Warning : n_clust_min muste be <(size of data). It was changed to (size of data - 1)")
        if n_clust_max <= n_clust_min:
            n_clust_max = n_clust_min + 1
            lg.warning("Warning : n_clust_max must be >(n_clust_min). It was changed to (n_clust_min + 1).")
        if n_clust_max > len(self.data):
            n_clust_max = len(self.data)
            lg.warning("Warning : n_clust_min muste be <=(size of data). It was changed to (size of data)")
        self.nclusters_list = []
        self.kmeans_list = []
        self.silhouette_avg_score_list = []
        self.sample_silhouette_values_list = []
        for nb in np.arange(n_clust_min, n_clust_max + 1):
            km = KMeans(n_clusters=nb, n_init=nb_try_per_cluster, random_state=self.random_state)
            km.fit(self.data)
            self.nclusters_list.append(nb)
            self.kmeans_list.append(km)
            self.silhouette_avg_score_list.append(silhouette_score(self.data, km.labels_))
            self.sample_silhouette_values_list.append(silhouette_samples(self.data, km.labels_))

    """
    Graph to help to choose the number of clusters, based on inertia. The function calculate_kmeans must be run before.
    Parameters
        figsize : size of the figure
        show_gridlines : to show or hide gridlines
    Returned data
        Matplotlib fig and ax
    """
    def graphique_choix_nclusters(self, figsize=(12,8)):
        graph = sfg.MyGraph("KMeans - Choix du nombre de clusters basé sur l'inertie", figsize=figsize)
        if self.nclusters_list is None:
            lg.warning("Warning : you must first run choix_n_clusters")
            return graph.fig, graph.ax
        inertia = []
        for i in np.arange(len(self.nclusters_list)):
            inertia.append(self.kmeans_list[i].inertia_)
        graph.graph_choix_nclusters_inertia(self.nclusters_list, inertia)
        return graph.fig, graph.ax
    
    def graphique_silhouette_scores(self, figsize=(12,8)):
        graph = sfg.MyGraph("KMeans - Choix du nombre de clusters basé sur le coefficient de silhouette", figsize=figsize)
        if self.nclusters_list is None:
            lg.warning("Warning : you must first run choix_n_clusters")
            return graph.fig, graph.ax
        graph.graph_choix_nclusters_silhouette(self.nclusters_list, self.silhouette_avg_score_list)
        return graph.fig, graph.ax
    
    def graphique_silhouette_details(self, min_nclust=None, max_nclust=None,
                                      min_silhouette=-1, max_silhouette=1,
                                      nbcol=2, plt_mult_param=1.0,
                                      figsize=(12,8)):
        if self.nclusters_list is None:
            graph = sfg.MyGraph("KMeans - Coefficient de silhouette en fonction du nombre de clusters", figsize=(12,4))
            lg.warning("Warning : you must first run choix_n_clusters")
            return graph.fig, graph.ax
        if min_nclust is None:
            min_nclust = self.nclusters_list[0]
        if max_nclust is None:
            max_nclust = self.nclusters_list[-1]
        nblin = (max_nclust - min_nclust + 1) // nbcol
        last_off = (max_nclust - min_nclust + 1) % nbcol
        if last_off > 0:
            nblin += 1
        graph = sfg.MyGraph("KMeans - Coefficient de silhouette en fonction du nombre de clusters",
                            nblin=nblin,
                            nbcol=nbcol,
                            color_palette="hls",
                            ncolors=8,
                            plt_mult_param=plt_mult_param,
                            figsize=figsize)
        graphidx = 1
        for nclust in np.arange(min_nclust, max_nclust+1):
            idxclust = self.nclusters_list.index(nclust)
            cluster_labels = self.kmeans_list[idxclust].labels_
            graph.graph_silhouette_detail(nclust,
                                          cluster_labels,
                                          self.silhouette_avg_score_list[idxclust],
                                          self.sample_silhouette_values_list[idxclust],
                                          min_silhouette=min_silhouette, 
                                          max_silhouette=max_silhouette,
                                          multi_index=graphidx)
            graphidx += 1
        if last_off > 0:
            for i in range(nbcol - last_off):
                graph.ax[graphidx-1].set_axis_off()
                graphidx += 1
        return graph.fig, graph.ax
        
    def graphique_silhouette_analysis(self, data_visualisation=None, min_nclust=None, max_nclust=None,
                                      min_silhouette=-1, max_silhouette=1,
                                      clust_figsize=(12,6), plt_mult_param=1.0,
                                      save_figs=False, save_path=None, save_name=None, dpi=300):
        if self.nclusters_list is None:
            graph = sfg.MyGraph("KMeans - Coefficient de silhouette en fonction du nombre de clusters", figsize=(12,4))
            lg.warning("Warning : you must first run choix_n_clusters")
            return graph.fig, graph.ax
        if data_visualisation is None:
            data_visualisation = self.data.iloc[:,:2]
        if min_nclust is None:
            min_nclust = self.nclusters_list[0]
        if max_nclust is None:
            max_nclust = self.nclusters_list[-1]
        for nclust in np.arange(min_nclust, max_nclust+1):
            idxclust = self.nclusters_list.index(nclust)
            ncolors = nclust
            if ncolors < 8:
                ncolors = 8
            cluster_labels = self.kmeans_list[idxclust].labels_
            graph = sfg.MyGraph("Nombre de clusters = {} - Coefficient de silhouette moyen = {:.3f}"\
                            .format(nclust, self.silhouette_avg_score_list[idxclust]), nblin=1, nbcol=2,
                            color_palette="hls", ncolors=ncolors, figsize=clust_figsize, plt_mult_param=plt_mult_param)
            graph.graph_silhouette_analysis(nclust, cluster_labels, self.silhouette_avg_score_list[idxclust],
                                            self.sample_silhouette_values_list[idxclust], data_visualisation,
                                            min_silhouette=min_silhouette, max_silhouette=max_silhouette)
            if save_figs:
                if save_name is None:
                    save_name = "silhouette"
                if save_path is None:
                    if IN_COLAB:
                        save_path = PATH_DRIVE
                    else:
                        save_path = ""
                path = save_path + save_name + "_{}".format(nclust)
                plt.savefig(path, dpi=dpi)
            plt.show()
    
    """
    Sets the choosen number of clusters
    Call the calculation of the 3 output dataframes
    """
    def define_n_clusters(self, n_clusters=None, nb_try_per_cluster=10):
        self.selected_n_clusters = n_clusters
        if n_clusters is None:
            self.km = None
            self.df_clusters = None
            self.df_centroids = None
        else:
            self.__calculate_clusters_centroids(nb_try_per_cluster)
        
    """
    Calculates the 3 output dataframes : df_clusters, df_centroids and df_data_clusters
    """
    def __calculate_clusters_centroids(self, nb_try_per_cluster):
        self.km = KMeans(n_clusters=self.selected_n_clusters,
                    n_init=nb_try_per_cluster,
                    random_state=self.random_state).fit(self.data)
        self.df_clusters = self.data.copy()
        self.df_clusters["cluster"] = self.km.labels_
        self.df_centroids = self.df_clusters.groupby("cluster").mean()
        self.df_centroids["nombre_individus"] = self.df_clusters.groupby("cluster").cluster.count()
        self.df_centroids.index = ["cluster {}".format(i) for i in self.df_centroids.index]
        print("Done")
        