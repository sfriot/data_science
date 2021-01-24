# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:42:22 2019

@author: Sylvain Friot

Content : classification and acp - analyses and graphs
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import dendrogram
import logging as lg
import sys


# changes in matplotlib default parameters
def mydefaut_plt_parameters(figsize=(12,8)):
    plt.rcParams['figure.figsize'] = figsize
    if figsize[0] == 12:
        mult_param = 1.0
        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.titlepad'] = 20
        plt.rcParams['axes.labelpad'] = 15
    else:
        mult_param = np.sqrt(1.0 * figsize[0] / 12)
        plt.rcParams['font.size'] = np.around(18 * mult_param)
        plt.rcParams['axes.titlepad'] = np.around(20 * mult_param)
        plt.rcParams['axes.labelpad'] = np.around(15 * mult_param)
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['legend.facecolor'] = (0.95,0.95,0.95)
    plt.rcParams['legend.edgecolor'] = (0.95,0.95,0.95)
    plt.rcParams['savefig.orientation'] = 'landscape'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    return mult_param


"""
Class : ClassificationHierarchique
Calculates hierarchical classification, and graphs to choose the number of clusters and to visualize dendrogram
Attributes
    data : dataframe with data
    data_size : number of observations in the sample
    data_values : data to classify
    data_names : names of observations in the data dataframe (index of data)
    data_variables : names of variables in the data dataframe (columns of data)
    method : method used for the classification calculations ('ward', 'single', 'complete', 'average', 'centroid', 'median', 'weighted')
    with_reduction : indicates if reduction is necessary (True by default)
    data_scaled : centered and reduced data values
    Z : calculated hierarchical clustering
    selected_n_clusters : selected number of clusters
    df_clusters : dataframe with the cluster to which each original observation belongs to (based on selected_n_clusters)
    df_centroids : dataframe with the mean values for the clusters centroids
    df_data_clusters : dataframe with original data and the cluster for each observation
Methods
    graphique_choix_nclusters : graph to help to choose the number of clusters, based on the distance between the two last separated clusters
    define_n_clusters : sets or changes the number of clusters
    calculate_clusters_centroids : fills in the 3 dataframes df_clusters, df_centroids and df_data_clusters (automatically called by define_n_clusters)
    graphique_dendrogramme : dendrogram graph (full graph or limited to n_clusters)
"""
class ClassificationHierarchique:
    
    def __init__(self, x, method='ward', with_reduction=True):
        try:
            if method not in ('ward', 'single', 'complete', 'average', 'centroid', 'median', 'weighted'):
                raise ValueError("Value Error : method must be one of 'ward', 'single', 'complete', 'average', 'centroid', 'median' or 'weighted'.")
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()[0]))
        else:
            self.data = x
            self.data_size = len(x)
            self.data_values = x.values
            self.data_names = x.index
            self.data_variables = x.columns
            self.method = method
            self.with_reduction = with_reduction
            self.data_scaled = preprocessing.StandardScaler(with_std=with_reduction).fit_transform(self.data_values)
            self.Z = linkage(self.data_scaled, self.method)
            self.selected_n_clusters = None
            self.df_clusters = None
            self.df_centroids = None
            self.df_data_clusters = None

    """
    Graph to help to choose the number of clusters, based on the distance between the two last separated clusters
    Parameters
        max_clusters : maximum number of clusters shown by the graph (if none, all possible numbers of clusters are shown)
        figsize : size of the figure
        show_gridlines : to show or hide gridlines
    Returned data
        Matplotlib fig and ax
    """
    def graphique_choix_nclusters(self, max_clusters=None, figsize=(12,8), show_gridlines=True):
        try:
            if (max_clusters <= 1) | (max_clusters > self.data_size):
                raise ValueError("Value Error : max_clusters must be between 2 and data_size ({})".format(self.data_size))
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()[0]))
        else:
            if max_clusters is None:
                graph_values = self.Z[:,2]
                graph_index = np.arange(len(self.Z)+1, 1, -1)
            else:
                graph_values = self.Z[-max_clusters+1:,2]
                graph_index = np.arange(max_clusters, 1, -1)
            mult_plt = mydefaut_plt_parameters(figsize=figsize)
            fig = plt.figure()
            ax = plt.axes()
            ax.bar(graph_index[::-1], graph_values[::-1])
            ax.set_xlabel("Nombre de clusters")
            ax.set_xlim(graph_index[-1]-0.5, graph_index[0]+0.5)
            nticks = graph_index[0] - graph_index[-1] + 1
            if nticks > 10:
                ax.xaxis.set_ticks(np.arange(graph_index[-1], graph_index[0]+1, int(round(nticks/10))))
                ax.set_xticklabels(np.arange(graph_index[-1], graph_index[0]+1, int(round(nticks/10))))
            else:
                ax.xaxis.set_ticks(np.arange(graph_index[-1], graph_index[0]+1))
                ax.set_xticklabels(np.arange(graph_index[-1], graph_index[0]+1))
            ax.set_ylabel("Distance entre les clusters séparés")
            if show_gridlines:
                plt.grid(axis="y", linestyle=":")
            plt.title("Dendrogramme - Choix du nombre de clusters")
            return fig, ax
    
    """
    Sets the choosen number of clusters
    Call the calculation of the 3 output dataframes
    """
    def define_n_clusters(self, n_clusters):
        try:
            if (n_clusters <= 1) | (n_clusters > self.data_size):
                raise ValueError("Value Error : n_clusters must be between 2 and data_size ({})".format(self.data_size))
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()[0]))
        else:
            self.selected_n_clusters = n_clusters
            self.calculate_clusters_centroids()
    
    """
    Calculates the 3 output dataframes : df_clusters, df_centroids and df_data_clusters
    """
    def calculate_clusters_centroids(self):
        try:
            if self.selected_n_clusters is None:
                raise ValueError("Value Error : selected_n_clusters must be defined before calculating the clusters")
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()[0]))
        else:
            clusters = fcluster(self.Z, self.selected_n_clusters, criterion='maxclust')
            df_clusters = pd.DataFrame(clusters, index=self.data_names, columns=["cluster"])
            temp = self.data.copy()
            temp["cluster"] = df_clusters.cluster
            df_centroids = temp.groupby("cluster").mean()
            df_centroids.index = ["cluster {}".format(i) for i in df_centroids.index]
            temp = df_clusters.groupby("cluster")[["cluster"]].count()
            temp.index = ["cluster {}".format(i) for i in temp.index]
            temp.columns = ["nombre_individus"]
            df_centroids["nombre_individus"] = temp.nombre_individus
            self.df_centroids = df_centroids
            self.df_clusters = df_clusters
            temp = self.data.copy()
            temp["cluster"] = self.df_clusters.cluster
            self.df_data_clusters = temp

    """
    Dendrogram graph
    Parameters
        title : graph title
        clusters_label_title : title for data.index
        figsize : size of the figure
        orientation : orientation of the graph. Indicates where the single cluster is.
        n_clusters : to limit the dendrogram to n_clusters
        mult_clusters_label_size : multiplier to increase or decrease the label size
        color_threshold : distance threshold to change clusters color
        show_gridlines : to show or hide gridlines
        label_rotation : degrees of rotation of labels
    Returned data
        Matplotlib fig and ax
    """
    def graphique_dendrogramme(self, title, clusters_label_title, figsize=(12,8), orientation='top', n_clusters=None, mult_clusters_label_size=1.0, color_threshold=None, show_gridlines=True, label_rotation=0):
        try:
            if orientation not in ('top','bottom','right','left'):
                raise ValueError("Value Error : orientation must be one of 'top','bottom','right','left'.".format(orientation))
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()[0]))
        else:
            mult_plt = mydefaut_plt_parameters(figsize=figsize)
            fig = plt.figure()
            ax = plt.axes()
            if n_clusters is None:
                dendrogram(self.Z, labels=self.data_names, orientation=orientation, leaf_font_size=(plt.rcParams['font.size']*mult_clusters_label_size), color_threshold=color_threshold)
                temptxt = clusters_label_title
            else:
                dendrogram(self.Z, labels=self.data_names, orientation=orientation, leaf_font_size=(plt.rcParams['font.size']*mult_clusters_label_size), truncate_mode='lastp', p=n_clusters, show_contracted=True)
                temptxt = "{} ou (nombre de lignes regroupées)".format(clusters_label_title)
            if (orientation == 'top') | (orientation == 'bottom'):
                #plt.ylabel("Distance")
                #plt.xlabel(temptxt)
                ax.set_ylabel("Distance")
                ax.set_xlabel(temptxt)
                if show_gridlines:
                    plt.grid(axis="y", linestyle=":")
                if label_rotation != 0:
                    xticks_locs, xticks_labels = plt.xticks()
                    plt.xticks(xticks_locs, xticks_labels, rotation=label_rotation, ha='right')
            else:
                #plt.xlabel("Distance")
                #plt.ylabel(temptxt)
                ax.set_xlabel("Distance")
                ax.set_ylabel(temptxt)
                if show_gridlines:
                    plt.grid(axis="x", linestyle=":")
                if label_rotation != 0:
                    yticks_locs, yticks_labels = plt.yticks()
                    plt.yticks(yticks_locs, yticks_labels, rotation=label_rotation)
            plt.title(title)
            return fig, ax


"""
Class : One Kmean
Object to hold the results of the Kmeans calculation for a given number of clusters
Attributes
    n_clusters : number of clusters
    centroids : coordinates of clusters centers
    points_label : cluster for each observation
    inertia : sum of squared distances of samples to their closest cluster center
    n_iterations : number of iterations run to get a stable partition
"""
class OneKmean:
    
    def __init__(self, n_clust, centroids, points_label, inertia, n_iter):
        self.n_clusters = n_clust
        self.centroids = centroids
        self.points_label = points_label
        self.inertia = inertia
        self.n_iterations = n_iter


"""
Class : ClassificationKmeans
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
class ClassificationKmeans:
    
    def __init__(self, x, with_reduction=True):
        self.data = x
        self.data_size = len(x)
        self.data_values = x.values
        self.data_names = x.index
        self.data_variables = x.columns
        self.with_reduction = with_reduction
        self.data_scaled = preprocessing.StandardScaler(with_std=with_reduction).fit_transform(self.data_values)
        self.kmeans_list = None
        self.selected_n_clusters = None
        self.df_clusters = None
        self.df_centroids = None
        self.df_data_clusters = None

    """
    Calculates the clusters for different numbers of clusters, and stores the results in OneKmean objects
    Parameters
        n_clust_min : minimum number of clusters (>=2 and <=data_size-1)
        n_clust_max : maximum number of clusters (>n_clust_min and <=data_size)
        nb_essais_par_cluster : number of times the kmeans calculation algorythm is run with different starting points
    """
    def calculate_kmeans(self, n_clust_min, n_clust_max, nb_essais_par_cluster=10):
        try:
            if (n_clust_min <= 1) | (n_clust_min > self.data_size - 1):
                raise ValueError("Value Error : n_clust_min must be between 2 and ({})".format(self.data_size - 1))
            if (n_clust_max <= n_clust_min) | (n_clust_max > self.data_size):
                raise ValueError("Value Error : n_clust_max must be between n_clust_min + 1 and ({})".format(self.data_size))
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()[0]))
        else:
            self.kmeans_list = []
            for nb in np.arange(n_clust_min, n_clust_max + 1):
                km = KMeans(n_clusters=nb, n_init=nb_essais_par_cluster)
                km.fit(self.data_scaled)
                #km.fit_transform(self.data_scaled)
                self.kmeans_list.append(OneKmean(n_clust=nb, centroids=km.cluster_centers_, points_label=km.labels_, inertia=km.inertia_, n_iter=km.n_iter_))
            self.selected_n_clusters = None

    """
    Graph to help to choose the number of clusters, based on inertia. The function calculate_kmeans must be run before.
    Parameters
        figsize : size of the figure
        show_gridlines : to show or hide gridlines
    Returned data
        Matplotlib fig and ax
    """
    def graphique_choix_nclusters(self, figsize=(12,8), show_gridlines=True):
        try:
            if self.kmeans_list is None:
                raise ValueError("Value Error : you must first calculate kmeans")
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()[0]))
        else:
            inertia = []
            n_clust = []
            for kmean in self.kmeans_list:
                inertia.append(kmean.inertia)
                n_clust.append(kmean.n_clusters)
            graph_data = pd.DataFrame(inertia, index=n_clust)
            mult_plt = mydefaut_plt_parameters(figsize=figsize)
            fig = plt.figure()
            ax = plt.axes()
            ax.bar(n_clust, graph_data[0])
            ax.set_xlabel("Nombre de clusters")
            ax.set_xlim(n_clust[0]-0.5, n_clust[-1]+0.5)
            nticks = n_clust[-1] - n_clust[0] + 1
            if nticks > 10:
                ax.xaxis.set_ticks(np.arange(n_clust[0], n_clust[-1]+1, int(round(nticks/10))))
                ax.set_xticklabels(np.arange(n_clust[0], n_clust[-1]+1, int(round(nticks/10))))
            else:
                ax.xaxis.set_ticks(np.arange(n_clust[0], n_clust[-1]+1))
                ax.set_xticklabels(np.arange(n_clust[0], n_clust[-1]+1))
            ax.set_ylabel("Inertie intra-classe")
            if show_gridlines:
                plt.grid(axis="y", linestyle=":")
            plt.title("KMeans - Choix du nombre de clusters")
            return fig, ax
    
    """
    Sets the choosen number of clusters
    Call the calculation of the 3 output dataframes
    """
    def define_n_clusters(self, n_clusters=None):
        try:
            if self.kmeans_list is None:
                raise ValueError("Value Error : you must first calculate kmeans")
            if n_clusters is not None:
                if (n_clusters < self.kmeans_list[0].n_clusters) | (n_clusters > self.kmeans_list[-1].n_clusters):
                    raise ValueError("Value Error : n_clusters must be in the range of calculated kmeans")
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()[0]))
        else:
            self.selected_n_clusters = n_clusters
            if n_clusters is None:
                self.df_clusters = None
                self.df_centroids = None
                self.df_data_clusters = None
            else:
                self.calculate_clusters_centroids()
        
    """
    Calculates the 3 output dataframes : df_clusters, df_centroids and df_data_clusters
    """
    def calculate_clusters_centroids(self):
        try:
            if self.selected_n_clusters is None:
                raise ValueError("Value Error : selected_n_clusters must be defined before calculating the clusters")
            if self.kmeans_list is None:
                raise ValueError("Value Error : you must first calculate kmeans")
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()[0]))
        else:
            clust_index = 0
            while self.kmeans_list[clust_index].n_clusters != self.selected_n_clusters:
                clust_index += 1
                
            df_clusters = pd.DataFrame(self.kmeans_list[clust_index].points_label, index=self.data_names, columns=["cluster"])
            temp = self.data.copy()
            temp["cluster"] = df_clusters.cluster
            df_centroids = temp.groupby("cluster").mean()
            df_centroids.index = ["cluster {}".format(i) for i in df_centroids.index]
            temp = df_clusters.groupby("cluster")[["cluster"]].count()
            temp.index = ["cluster {}".format(i) for i in temp.index]
            temp.columns = ["nombre_individus"]
            df_centroids["nombre_individus"] = temp.nombre_individus
            self.df_centroids = df_centroids
            self.df_clusters = df_clusters
            temp = self.data.copy()
            temp["cluster"] = self.df_clusters.cluster
            self.df_data_clusters = temp
            

"""
Class : ACP
Generates principal component analysis and its attached graphs
Attributes
    data : dataframe with data
    data_size : number of observations in the sample
    data_values : data to classify
    data_names : names of observations in the data dataframe (index of data)
    data_variables : names of variables in the data dataframe (columns of data)
    with_reduction : indicates if reduction is necessary (True by default)
    data_scaled : centered and reduced data values
    n_components : number of components of the PCA
    pca : principal components
    variables_projected : projection of variables on principal components (correlation circles)
    data_projected : projection of members on principal components (factorial planes)
    df_variables_projected : dataframe with the linear combination of variables (coordinates of variables on the principal components axes)
    df_data_projected : dataframe with the coordinates of members on the principal inertia axes
    df_explained_variance : dataframe with explained_variance and explained_variance_ratio for each principal component axe
    df_centroids_projected : dataframe with the coordinates of centroids on the principal inertia axes
Methods
    calculate_kmeans : calculates the clusters for different numbers of clusters, and stores the results in OneKmean objects
    graphique_choix_nclusters : graph to help to choose the number of clusters, based on the inertia
    define_n_components : sets or changes the number of clusters
    calculate_clusters_centroids : fills in the 3 dataframes df_clusters, df_centroids and df_data_clusters (automatically called by define_n_clusters)
"""
class ACP:

    def __init__(self, x, n_components, with_reduction=True):
        try:
            if (n_components <= 1) | (n_components > (len(x) - 1)) | (n_components > len(x.columns)):
                raise ValueError("Value Error : n_components must be at least 2 and no more than the maximum between the number of variables {} and the number of data - 1 {}.".format(x.columns, (len(x) - 1)))
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()[0]))
        else:
            self.data = x
            self.data_size = len(x)
            self.data_values = x.values
            self.data_names = x.index
            self.data_variables = x.columns
            self.with_reduction = with_reduction
            self.scaler = preprocessing.StandardScaler(with_std=with_reduction).fit(self.data_values)
            self.data_scaled = self.scaler.transform(self.data_values)
            self.n_components = n_components
            self.pca = None
            self.variables_projected = None
            self.data_projected = None
            self.df_variables_projected = None
            self.df_data_projected = None
            self.df_explained_variance = None
            self.df_centroids_projected = None
            self.calculate_acp()
        
    """
    Sets the choosen number of components
    Call the calculation of the PCA and of the 3 output dataframes
    """
    def define_n_components(self, n_components):
        try:
            if (n_components <= 1) | (n_components > (self.data_size - 1)) | (n_components > len(self.data_variables)):
                raise ValueError("Value Error : n_components must be at least 2 and no more than the maximum between the number of variables {} and the number of data - 1 {}.".format(self.data_variables, (self.data_size - 1)))
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()[0]))
        else:
            self.n_components = n_components
            self.calculate_acp()
            
    """
    Calculates the 3 output dataframes : df_variables_projected, df_data_projected, df_explained_variance
    """
    def calculate_acp(self):
        self.pca = decomposition.PCA(n_components=self.n_components) # calcul des composantes principales
        self.pca.fit(self.data_scaled)  
        self.variables_projected = self.pca.components_ # projection des variables sur les axes d'inertie (cercles des corrélations)
        self.data_projected = self.pca.transform(self.data_scaled) # projection des individus sur les axes d'inertie
        self.df_variables_projected = pd.DataFrame(self.variables_projected, \
            index=["F{} ({}%)".format(i+1, round(100*self.pca.explained_variance_ratio_[i],1)) for i in np.arange(self.n_components)], \
            columns=self.data_variables)
        self.df_data_projected = pd.DataFrame(self.data_projected, index=self.data_names, \
            columns=["F{} ({}%)".format(i+1, round(100*self.pca.explained_variance_ratio_[i],1)) for i in np.arange(self.n_components)])
        self.df_explained_variance = pd.DataFrame(data={"explained_variance" : self.pca.explained_variance_, \
            "explained_variance_ratio" : self.pca.explained_variance_ratio_}, index=["F{}".format(i+1) for i in np.arange(self.n_components)])
        self.df_centroids_projected = None
        
    """
    Generate a dataframe with coordinates of data projected on principal components axes
    """
    def calculate_newdata_projection(self, input_data):
        input_data_scaled = self.scaler.transform(input_data)
        input_data_projected = self.pca.transform(input_data_scaled)
        df_input_projected = pd.DataFrame(input_data_projected, index=input_data.index, \
            columns=["F{} ({}%)".format(i+1, round(100*self.pca.explained_variance_ratio_[i],1)) for i in np.arange(self.n_components)])
        return df_input_projected

    """
    Generate a dataframe with coordinates of centroids projected on principal components axes
    Parameter
        df_clusters : dataframe df_clusters generated by hierarchical classification or k-means classification
    Returned data
        df_centroids_projected : dataframe df_centroids_projected with the coordinates of centroids principal components axes for each cluster
    """
    def calculate_centroids_projection(self, df_clusters):
        temp = self.df_data_projected.copy()
        temp["cluster"] = df_clusters.cluster
        df_centroids_projected = temp.groupby("cluster").mean()
        df_centroids_projected.index = ["cluster {}".format(i) for i in df_centroids_projected.index]
        self.df_centroids_projected = df_centroids_projected
        return df_centroids_projected
        
    """
    Graph of scree (éboulis des valeurs propres) to help to choose the number of components, based on the explained variance ratio
    Parameters
        figsize : size of the figure
        show_gridlines : to show or hide gridlines
    Returned data
        Matplotlib fig and ax
    """
    def graphique_scree_plot(self, figsize=(12,8), show_gridlines=True):
        mult_plt = mydefaut_plt_parameters(figsize=figsize)
        scree = self.pca.explained_variance_ratio_*100
        fig = plt.figure()
        ax = plt.axes()
        ax.bar(np.arange(len(scree))+1, scree)
        ax.plot(np.arange(len(scree))+1, scree.cumsum(), c="red", marker='o')
        ax.set_xlabel("Rang de l'axe d'inertie")
        ax.set_xlim(0.5, len(scree)+0.5)
        if len(scree) > 10:
            ax.xaxis.set_ticks(np.arange(len(scree)+1, step=int(round(len(scree)/10))))
            ax.set_xticklabels(np.arange(len(scree)+1, step=int(round(len(scree)/10))))
        else:
            ax.xaxis.set_ticks(np.arange(len(scree))+1)
            ax.set_xticklabels(np.arange(len(scree))+1)
        ax.set_ylabel("Pourcentage d'inertie (%)")
        ax.set_ylim(0,100)
        if show_gridlines==True:
            plt.grid(axis="y", linestyle=":")
        plt.title("Eboulis des valeurs propres")
        return fig, ax
    
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
    def graphique_correlation_circles(self, axis_ranks=(0,1), show_labels=True, label_rotation=0, figsize=(12,13), show_gridlines=True, lims=None):
        mult_plt = mydefaut_plt_parameters(figsize=figsize)
        d1 = axis_ranks[0]
        d2 = axis_ranks[1]
        if (d1 < self.n_components) & (d2 < self.n_components):
            fig = plt.figure(figsize=figsize)
            ax = plt.axes()
            if lims is not None :  # détermination des limites du graphique
                xmin, xmax, ymin, ymax = lims
            elif self.variables_projected.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(self.variables_projected[d1,:]), max(self.variables_projected[d1,:]), min(self.variables_projected[d2,:]), max(self.variables_projected[d2,:])
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            if self.variables_projected.shape[1] < 30 :  # affichage des flèches ; s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
                plt.quiver(np.zeros(self.variables_projected.shape[1]), np.zeros(self.variables_projected.shape[1]), self.variables_projected[d1,:], self.variables_projected[d2,:], angles='xy', scale_units='xy', scale=1, color="grey")
            else:
                lines = [[[0,0],[x,y]] for x,y in self.variables_projected[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            if show_labels:  # affichage des noms des variables
                for i,(x, y) in enumerate(self.variables_projected[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y + (ymax-ymin)*0.015*np.sign(y-((ymin+ymax)/2)), self.data_variables[i], fontsize=plt.rcParams['font.size'], ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')  # affichage du cercle
            plt.gca().add_artist(circle)
            #plt.axis('equal')
            if show_gridlines==True:  # affichage des gridlines
                plt.grid(axis="both", linestyle=":", linewidth=0.5)
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')  # affichage des lignes horizontales et verticales
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*self.pca.explained_variance_ratio_[d1],1)))  # nom des axes, avec le pourcentage d'inertie expliqué
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*self.pca.explained_variance_ratio_[d2],1)))
            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
        else:
            fig, ax = self.error_graph(d1, d2, figsize)
        return fig, ax

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
    def graphique_factorial_planes(self, axis_ranks=(0,1), show_labels=False, alpha=1, illustrative_var=None, illustrative_legend=None, figsize=(12,13), lims=None):
        mult_plt = mydefaut_plt_parameters(figsize=figsize)
        d1 = axis_ranks[0]
        d2 = axis_ranks[1]
        if (d1 < self.n_components) & (d2 < self.n_components):      
            data_graph = np.array(self.data_projected)
            data_labels = np.array(self.data_names)
            title = "Projection des individus"
            fig, ax = self.graphique_generic_projection(data_graph, data_labels, d1, d2, title, axis_ranks=axis_ranks, show_labels=show_labels, alpha=alpha, illustrative_var=illustrative_var, illustrative_legend=illustrative_legend, figsize=figsize, lims=lims)
        else:
            fig, ax = self.error_graph(d1, d2, figsize)
        return fig, ax
    
    """
    Graph of centroids projection : projection of centroids on the principal components axes
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
    def graphique_centroids_projection(self, axis_ranks=(0,1), show_labels=False, alpha=1, illustrative_var=None, illustrative_legend=None, figsize=(12,13), lims=None):
        mult_plt = mydefaut_plt_parameters(figsize=figsize)
        d1 = axis_ranks[0]
        d2 = axis_ranks[1]
        if (d1 < self.n_components) & (d2 < self.n_components) & (self.df_centroids_projected is not None):
            data_graph = np.array(self.df_centroids_projected)
            data_labels = np.array(self.df_centroids_projected.index)
            title = "Projection des centroïdes"
            fig, ax = self.graphique_generic_projection(data_graph, data_labels, d1, d2, title, axis_ranks=axis_ranks, show_labels=show_labels, alpha=alpha, illustrative_var=illustrative_var, illustrative_legend=illustrative_legend, figsize=figsize, lims=lims)
        else:
            if self.df_centroids_projected is None:
                fig, ax = self.error_graph(d1, d2, figsize, centroids_calculation=True)
            else:
                fig, ax = self.error_graph(d1, d2, figsize)
        return fig, ax
    
    """
    graph projections of points or centroids - generic graph
    """
    def graphique_generic_projection(self, data_graph, data_labels, d1, d2, title, axis_ranks, show_labels, alpha, illustrative_var, illustrative_legend, figsize, lims):
        sns.set_palette("Set2")
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        if illustrative_var is None:  # affichage des points
            plt.scatter(data_graph[:, d1], data_graph[:, d2], alpha=alpha)
        else:
            illustrative_var = np.array(illustrative_var)
            for value in np.unique(illustrative_var):
                selected = np.where(illustrative_var == value)
                plt.scatter(data_graph[selected, d1], data_graph[selected, d2], alpha=alpha, label=value)
            if illustrative_legend is None:
                plt.legend()
            else:
                plt.legend(title=illustrative_legend)
        if lims is None:
            boundary = np.max(np.abs(data_graph[:, [d1,d2]])) * 1.1  # détermination des limites du graphique
            xmax = boundary
            xmin = -boundary
            ymax = boundary
            ymin = -boundary
        else:
            xmin, xmax, ymin, ymax = lims
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        if show_labels:  # affichage des labels des points
            for i,(x,y) in enumerate(data_graph[:,[d1,d2]]):
                if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                    plt.text(x, y + (2*boundary)*0.015*np.sign(y), self.data_labels[i], fontsize=plt.rcParams['font.size'], ha='center',va='center') 
        plt.plot([-100, 100], [0, 0], color='grey', ls='--')  # affichage des lignes horizontales et verticales
        plt.plot([0, 0], [-100, 100], color='grey', ls='--')
        plt.xlabel('F{} ({}%)'.format(d1+1, round(100*self.pca.explained_variance_ratio_[d1],1)))  # nom des axes, avec le pourcentage d'inertie expliqué
        plt.ylabel('F{} ({}%)'.format(d2+1, round(100*self.pca.explained_variance_ratio_[d2],1)))
        plt.title("{} (sur F{} et F{})".format(title, d1+1, d2+1))
        return fig, ax
    
    """
    Error graph if an axe has not been calculated before being graphed
    """
    def error_graph(self, d1, d2, figsize, centroids_calculation=False):
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        boundary = 1  # détermination des limites du graphique et affichage du message
        plt.xlim([-boundary,boundary])
        plt.ylim([-boundary,boundary])
        if centroids_calculation:
            plt.text(0, 0.2, "Impossible de créer le graphique\ncar la fonction calculate_centroids_projection n'a pas été appelée auparavant".format(self.n_components),
                          fontsize=plt.rcParams['font.size'], color='red', ha='center',va='center')
        else:
            plt.text(0, 0.2, "Impossible de créer le graphique\ncar l'ACP n'a été calculée que sur {} axes d'inertie".format(self.n_components),
                          fontsize=plt.rcParams['font.size'], color='red', ha='center',va='center')
        plt.plot([-100, 100], [0, 0], color='grey', ls='--')  # affichage des lignes horizontales et verticales
        plt.plot([0, 0], [-100, 100], color='grey', ls='--')
        plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
        return fig, ax
    
    
class ProductionACP:

    def __init__(self, variables, scaler, n_components, pca):
        self.variables = variables
        self.scaler = scaler
        self.n_components = n_components
        self.pca = pca
        