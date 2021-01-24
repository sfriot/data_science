# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:37:00 2019

@author: Sylvain Friot

Content: generic graphs
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.colors import Normalize as pltnormalize
from matplotlib.collections import LineCollection
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import missingno as msno
from sklearn.metrics import confusion_matrix

# changes in matplotlib default parameters
def mydefault_plt_parameters(figsize=(12,8), mult_param=1.0):
    plt.rcParams['figure.figsize'] = figsize
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

def set_colors(is_mono, color_palette=None, ncolors=None, desat=None):
    if color_palette is None:
        if is_mono:
            sns.set_palette(sns.light_palette("navy"))
        else:
            sns.set_palette("Set2")
    else:
        sns.set_palette(color_palette, ncolors, desat)
    return sns.color_palette()

"""
def set_rainbow_colors(nb_colors=8, gist=True):
    if gist:
        name = "gist_rainbow"
    else:
        name = "rainbow"
    sns.set_palette(sns.color_palette(name, n_colors=nb_colors))
    return sns.color_palette()

def set_rainbow_colors(nb_colors=8, gist=True):
    if gist:
        name = "gist_rainbow"
    else:
        name = "rainbow"
    sns.set_palette(sns.color_palette(name, n_colors=nb_colors))
    return sns.color_palette()

def get_colors(is_mono):
    if is_mono:
        liste_couleurs = sns.color_palette(sns.light_palette("navy"))
    else:
        liste_couleurs = sns.color_palette("Set2")
    return liste_couleurs

def get_rainbow_colors(nb_colors=8, gist=True):
    if gist:
        name = "gist_rainbow"
    else:
        name = "rainbow"
    return sns.color_palette(name, n_colors=nb_colors)
"""

class MyGraph:
    """
    with_grid : None, 'both', 'x', 'y'
    """
    def __init__(self, title, y_title=1.025, nblin=1, nbcol=1, is_mono=False, color_palette=None, ncolors=None, desat=None,
                 figsize=(12,8), plt_mult_param=None, wspace=0.25, hspace=0.25, frameon=True, polar=False, gridspec=False):
        sns.set_style("white")
        if (nblin > 1) & (figsize == (12,8)):
            figsize = (12, 6 * nblin)
        if plt_mult_param is None:
            plt_mult_param = np.sqrt(1.0 * figsize[0] / 12)
        mydefault_plt_parameters(figsize=figsize, mult_param=plt_mult_param)
        self.liste_couleurs = set_colors(is_mono, color_palette, ncolors, desat)
        self.polar = polar
        self.frameon = frameon
        self.fig = plt.figure(frameon=frameon)
        self.ax = []
        self.ax_desc = []
        self.objets = []
        self.objets_desc = []
        self.gs = None
        if (nblin == 1) & (nbcol == 1):
            self.__init_graph_simple(title=title, y_title=y_title, polar=polar)
        else:
            if gridspec:
                self.__init_graph_gridspec(title=title, y_title=y_title, nblin=nblin, nbcol=nbcol)
            else:
                self.__init_graph_multiple(title=title, y_title=y_title, nblin=nblin, nbcol=nbcol, wspace=wspace, hspace=hspace, frameon=frameon, polar=polar)
    
    def __init_graph_simple(self, title, y_title, polar):
        self.ax.append(plt.axes(polar=polar))
        self.ax_desc.append("Axe principal - index 1")
        self.fig.suptitle(title, y=y_title)
        self.fig.tight_layout()

    def __init_graph_multiple(self, title, y_title, nblin, nbcol, wspace, hspace, frameon, polar):
        cpt = 0
        self.fig.subplots_adjust(wspace=wspace, hspace=hspace)
        for lin in np.arange(nblin):
            for col in np.arange(nbcol):
                cpt += 1
                self.ax.append(plt.subplot(nblin, nbcol, cpt, frameon=frameon, polar=polar))
                self.ax_desc.append("Axe principal du subplot {} - index {}".format(cpt, cpt))
        self.fig.suptitle(title, y=y_title)
        self.fig.tight_layout()
        
    def __init_graph_gridspec(self, title, y_title, nblin, nbcol):
        self.fig.subplots_adjust(wspace=0.25, hspace=0.25)
        self.gs = plt.GridSpec(nblin, nbcol, figure=self.fig)
        self.fig.suptitle(title, y=y_title)
        self.fig.tight_layout()
        
    def __add_grid_twinx(self, multi_index, with_grid, grid_style="-", twinx=False):
        if twinx:
            self.ax[multi_index-1].spines['right'].set_visible(True)
            self.ax.append(self.ax[multi_index-1].twinx())
            self.ax_desc.append("Axe secondaire du subplot {} - index {}".format(multi_index, len(self.ax)))
            multi_index = len(self.ax)
        else:
            if with_grid is not None:
                self.ax[multi_index-1].grid(axis=with_grid, linestyle=grid_style)
            if self.ax[multi_index-1].name == 'rectilinear':
                self.ax[multi_index-1].spines['right'].set_visible(False)
                self.ax[multi_index-1].spines['top'].set_visible(False)
        return multi_index
    
    def __add_legend_subtitle(self, multi_index, legend, subtitle, twinx=False):
        if legend:
            self.ax[multi_index-1].legend()
        if (subtitle is not None) & (not twinx):
            self.ax[multi_index-1].set_title(subtitle, fontweight='regular')
        self.fig.tight_layout()
    
    def add_histogramme(self, values, labels=None, bins=20, range=None, density=True, stacked=False, bar_width_percent=None, color=None, alpha=1, subtitle=None, legend=False, with_grid='both', grid_style="-", multi_index=1, twinx=False):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style, twinx)
        self.objets.append(self.ax[multi_index-1].hist(values, bins=bins, range=range, label=labels, density=density, stacked=stacked, rwidth=bar_width_percent, color=color, alpha=alpha))
        self.objets_desc.append("Histogramme de l'axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, subtitle, twinx)
    
    def __add_bar(self, orientation, x_values, y_height, with_grid, grid_style, label=None, y_bottom=None, y_error=None, bar_width=0.8, color=None, alpha=1, error_kw=None, subtitle=None, legend=False, multi_index=1, twinx=False):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style, twinx)
        if orientation == 'horizontal':
            self.objets.append(self.ax[multi_index-1].barh(x_values, y_height, left=y_bottom, xerr=y_error, height=bar_width, color=color, label=label, alpha=alpha, error_kw=error_kw))
        else:
            self.objets.append(self.ax[multi_index-1].bar(x_values, y_height, bottom=y_bottom, yerr=y_error, width=bar_width, color=color, orientation=orientation, label=label, alpha=alpha, error_kw=error_kw))
        self.objets_desc.append("Barre de l'axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, subtitle, twinx)
            
    def add_barv(self, x_values, y_height, label=None, y_bottom=None, y_error=None, bar_width=0.8, color=None, alpha=1, error_kw=dict(lw=3, capsize=5, capthick=3, ecolor='dimgrey'), subtitle=None, legend=False, with_grid='y', grid_style=":", multi_index=1, twinx=False):
        orientation = 'vertical'
        self.__add_bar(orientation, x_values, y_height, with_grid, grid_style, label, y_bottom, y_error, bar_width, color, alpha, error_kw, subtitle, legend, multi_index, twinx)
    
    def add_barh(self, y_values, x_width, label=None, x_left=None, x_error=None, bar_height=0.8, color=None, alpha=1, error_kw=dict(lw=3, capsize=5, capthick=3, ecolor='dimgrey'), subtitle=None, legend=False, with_grid='x', grid_style=":", multi_index=1, twinx=False):
        orientation = 'horizontal'
        self.__add_bar(orientation, y_values, x_width, with_grid, grid_style, label, x_left, x_error, bar_height, color, alpha, error_kw, subtitle, legend, multi_index, twinx)

    def add_plot(self, x_values, y_values, label=None, marker=None, linestyle="-", color=None, markeredgecolor=None, markerfacecolor=None, alpha=1, linewidth=1.5, subtitle=None, legend=False, with_grid=None, grid_style="-", multi_index=1, twinx=False):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style, twinx)
        self.objets.append(self.ax[multi_index-1].plot(x_values, y_values, label=label, marker=marker, linestyle=linestyle, color=color, markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor, alpha=alpha, linewidth=linewidth))
        self.objets_desc.append("Plot de l'axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, subtitle, twinx)
        
    def add_fill(self, x_values, y_values, color=None, alpha=1, subtitle=None, legend=False, with_grid=None, grid_style="-", multi_index=1, twinx=False):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style, twinx)
        self.objets.append(self.ax[multi_index-1].fill(x_values, y_values, color=color, alpha=alpha))
        self.objets_desc.append("Fill de l'axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, subtitle, twinx)
        
    def add_plot_date(self, x_values, y_values, label=None, marker=None, linestyle="-", color=None, markeredgecolor=None, markerfacecolor=None, alpha=1, linewidth=1.5, subtitle=None, legend=False, with_grid=None, grid_style="-", multi_index=1, twinx=False):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style, twinx)
        self.objets.append(self.ax[multi_index-1].plot_date(x_values, y_values, label=label, marker=marker, linestyle=linestyle, color=color, markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor, alpha=alpha, linewidth=linewidth))
        self.objets_desc.append("Plot de l'axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, subtitle, twinx)
        
    def add_scatter(self, x_values, y_values, label=None, marker=None, size=None, color=None, uniquecolor=None, cmap=None, alpha=1, subtitle=None, legend=False, with_grid='both', grid_style="-", multi_index=1, twinx=False):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style, twinx)
        self.objets.append(self.ax[multi_index-1].scatter(x_values, y_values, label=label, marker=marker, s=size, c=color, color=uniquecolor, cmap=cmap, alpha=alpha))
        self.objets_desc.append("Scatter de l'axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, subtitle, twinx)
        
    def add_pie(self, values, labels, explodes=None, colors=None, autopct='%1.1f%%', startangle=90, shadow=False, counterclock=True, center_white=False, subtitle=None, legend=False, multi_index=1):
        # pie ne peut pas prendre twinx et n'affiche pas de grille -> pas d'appel à __add_grid_twinx
        self.objets.append(self.ax[multi_index-1].pie(values, labels=labels, autopct=autopct, startangle=startangle, explode=explodes, colors=colors, shadow=shadow, counterclock=counterclock))
        self.objets_desc.append("Camembert de l'axe {}".format(multi_index-1))
        if center_white:
            self.add_cercle(0, 0, 0.75, facecolor='w')
        self.ax[multi_index-1].axis('equal')
        self.__add_legend_subtitle(multi_index, legend, subtitle, False)
    
    def add_area(self, basis_values, bornes_min, bornes_max, orientation='vertical', label=None, color=None, alpha=0.2, linestyle='-', linewidth=0, subtitle=None, legend=False, with_grid=None, grid_style="-", multi_index=1, twinx=False):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style, twinx)
        if orientation == 'vertical':
            self.objets.append(self.ax[multi_index-1].fill_between(basis_values, bornes_min, bornes_max, label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth))
        elif orientation == 'horizontal':
            self.objets.append(self.ax[multi_index-1].fill_betweenx(basis_values, bornes_min, bornes_max, label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth))
        self.objets_desc.append("Aire de l'axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, subtitle, twinx)
    '''
    def add_area(self, x_values, y_min, y_max, label=None, color=None, alpha=0.2, linestyle='-', linewidth=0, subtitle=None, legend=False, with_grid=None, grid_style="-", multi_index=1, twinx=False):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style, twinx)
        self.objets.append(self.ax[multi_index-1].fill_between(x_values, y_min, y_max, label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth))
        self.objets_desc.append("Aire de l'axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, subtitle, twinx)
    '''
    # df_values : dataframe with features in columns and individuals in lines.
    def add_radar_plot(self, df_values, features_names=None, indiv_labels=None, min_value=0, max_value=1, nb_graduations=4, format_graduations=":.2f", colors=None, fill=True, subtitle=None, legend=True, legend_fontsize=None, multi_index=1):
        # radar ne peut pas prendre twinx et n'affiche pas de grille -> pas d'appel à __add_grid_twinx
        if features_names is None:
            features_names = list(df_values.columns)
        if indiv_labels is None:
            indiv_labels = list(df_values.index)
        if colors is None:
            colors = [self.liste_couleurs[i] for i in range(df_values.shape[0])]
        nb_features = df_values.shape[1]
        angles = [n / float(nb_features) * 2 * np.pi for n in range(nb_features)]
        angles += angles[:1]  # on rajoute toujours la première valeur à la fin pour fermer le cercle
        self.ax[multi_index-1].set_theta_offset(np.pi / 2)  # first feature on top
        self.ax[multi_index-1].set_theta_direction(-1)  # features in clockwise direction
        self.ax[multi_index-1].set_xticks(angles[:-1])
        self.ax[multi_index-1].set_xticklabels(features_names)
        rotated_labels = []  # rotation of xticklabels is reset to at drawing time for polar graph -> creation of new labels
        for i in range(nb_features):
            interlabel = self.ax[multi_index-1].text(angles[i], max_value * 1.03, features_names[i],
                                fontsize=12, ha="left", va="center", 
                                rotation=angles[i]*-180./np.pi + 90, rotation_mode="anchor")
            rotated_labels.append(interlabel)
        self.ax[multi_index-1].set_xticklabels([])
        self.ax[multi_index-1].set_rlabel_position(0)
        ylabels = [min_value + i * (max_value-min_value) / nb_graduations for i in range(1, nb_graduations)]
        ylabels_format = "{" + format_graduations + "}"
        self.ax[multi_index-1].set_yticks(ylabels)
        self.ax[multi_index-1].set_yticklabels([ylabels_format.format(i) for i in ylabels], fontsize=12, color="grey")
        self.ax[multi_index-1].set_ylim([min_value, max_value])
        for indiv in range(df_values.shape[0]):
            plot_values = df_values.iloc[indiv].values.tolist()
            plot_values += plot_values[:1]
            self.add_plot(angles, plot_values, label=indiv_labels[indiv],
                          color=colors[indiv], multi_index=multi_index)
            if fill:
                self.add_fill(angles, plot_values, color=colors[indiv],
                              alpha=0.1, multi_index=multi_index)                
        if (df_values.shape[0] > 1) & (legend):
            self.set_legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=legend_fontsize, multi_index=multi_index)
    
    def add_colorbar(self, objets_index=0):
        self.objets.append(self.fig.colorbar(self.objets[objets_index]))
        self.objets_desc.append("Colorbar de l'objet {}".format(objets_index))
        
    def add_boxplot(self, values, cat_labels=None, color=None, color_base_index=None, outliers=True, mult_iq=1.5, means=False, width=0.7, vertical=True, alpha=1, subtitle=None, legend=False, grid_style=":", multi_index=1):
        if vertical:
            with_grid = 'y'
        else:
            with_grid = 'x'
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style)
        bplot = self.ax[multi_index-1].boxplot(values, labels=cat_labels, showfliers=outliers, whis=mult_iq, showmeans=means, widths=width, vert=vertical, patch_artist=True)
        if cat_labels is None:
            nb_cat_labels = 1
        else:
            nb_cat_labels = len(cat_labels)
        for cpt in np.arange(nb_cat_labels):
            if color is None:
                if color_base_index is None:
                    color = self.liste_couleurs[(cpt + multi_index - 1) % len(self.liste_couleurs)]
                else:
                    color = self.liste_couleurs[(cpt + color_base_index) % len(self.liste_couleurs)]
            bplot['boxes'][cpt].set_facecolor(color)
            bplot['boxes'][cpt].set_alpha(alpha)
            bplot['medians'][cpt].set_color('black')
            if outliers:
                bplot['fliers'][cpt].set_markeredgecolor('grey')
                bplot['fliers'][cpt].set_markerfacecolor(color)
                bplot['fliers'][cpt].set_alpha(alpha)
            if means:
                bplot['means'][cpt].set_marker('o')
                bplot['means'][cpt].set_markeredgecolor('black')
                bplot['means'][cpt].set_markerfacecolor('black')
        self.objets.append(bplot)
        self.objets_desc.append("Boxplot de l'axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, subtitle, False)
    
    def add_texts_boxplot(self, text, x, y, color=None, multi_index=1):
        artists = self.ax[multi_index-1].artists
        for cpt in np.arange(len(artists)):
            if color is None:
                mycolor = artists[cpt].get_facecolor()
            else:
                mycolor = color
            self.objets.append(self.ax[multi_index-1].text(x[cpt], y[cpt], text[cpt], horizontalalignment='center',verticalalignment='center', color=mycolor))
            self.objets_desc.append("Texte du boxplot {} - axe {}".format(cpt, multi_index-1))
        
    def add_sns_regplot(self, x, y, color_index=None, subtitle=None, show_labels=True, multi_index=1):
        if color_index is None:
            color = self.liste_couleurs[0]
        else:
            color = self.liste_couleurs[color_index]
        self.objets.append(sns.regplot(x=x, y=y, color=color, ax=self.ax[multi_index-1]))
        self.objets_desc.append("Plot de la régression linéaire - axe {}".format(multi_index-1))
        if subtitle is not None:
            self.ax[multi_index-1].set_title(subtitle, fontweight='regular')
        if show_labels == False:
            self.set_axe('x', label="", multi_index=multi_index)
            self.set_axe('y', label="", multi_index=multi_index)
        self.ax[multi_index-1].spines['right'].set_visible(False)
        self.ax[multi_index-1].spines['top'].set_visible(False)
        
    def add_sns_heatmap(self, table_values, annotations=False, cmap='Blues', fmt='g', vmin=None, vmax=None, cbar=True, subtitle=None, multi_index=1):
        self.objets.append(sns.heatmap(table_values, annot=annotations, cmap=cmap, fmt=fmt, vmin=vmin, vmax=vmax, cbar=cbar, ax=self.ax[multi_index-1]))
        self.objets_desc.append("Heatmap - axe {}".format(multi_index-1))
        if subtitle is not None:
            self.ax[multi_index-1].set_title(subtitle, fontweight='regular')
    
    def add_sns_qqplot(self, data, fit=True, color_index=None, alpha=0.5, subtitle=None, multi_index=1):
        if color_index is None:
            color = self.liste_couleurs[0]
        else:
            color = self.liste_couleurs[color_index]
        self.objets.append(sm.qqplot(data, fit=fit, markeredgecolor=color, markerfacecolor=color, alpha=alpha, ax=self.ax[multi_index-1]))
        self.objets_desc.append("QQ Plot - axe {}".format(multi_index-1))
        if subtitle is not None:
            self.ax[multi_index-1].set_title(subtitle, fontweight='regular')
    
    def add_sns_boxenplot(self, x, y, data, hue=None, order=None, hue_order=None, orient=None, color_index=None, subtitle=None, multi_index=1):
        if color_index is None:
            color = None
            palette= self.liste_couleurs
        else:
            color = self.liste_couleurs[color_index]
            palette = None
        self.objets.append(sns.boxenplot(x=x, y=y, data=data, hue=hue, order=order, hue_order=hue_order, orient=orient, color=color, palette=palette, ax=self.ax[multi_index-1]))
        self.objets_desc.append("Boxen Plot - axe {}".format(multi_index-1))
        if subtitle is not None:
            self.ax[multi_index-1].set_title(subtitle, fontweight='regular')
            
    def add_sns_boxplot(self, x, y, data, hue=None, order=None, hue_order=None, orient=None, color_index=None, subtitle=None, multi_index=1):
        if color_index is None:
            color = None
            palette= self.liste_couleurs
        else:
            color = self.liste_couleurs[color_index]
            palette = None
        self.objets.append(sns.boxplot(x=x, y=y, data=data, hue=hue, order=order, hue_order=hue_order, orient=orient, color=color, palette=palette, ax=self.ax[multi_index-1]))
        self.objets_desc.append("Box Plot - axe {}".format(multi_index-1))
        if subtitle is not None:
            self.ax[multi_index-1].set_title(subtitle, fontweight='regular')
        
    def graph_custom_regplot(self, data, x, y, hue=None, marker='o', linestyle="-", color_base_index=0, marker_alpha=0.5, reg_color=None, subtitle=None, with_grid=None, grid_style="-", legend=True, multi_index=1):
        cpt = 0
        self.liste_couleurs[(cpt + color_base_index) % len(self.liste_couleurs)]
        if hue is None:
            self.add_plot(data[x], data[y], marker=marker, label="", linestyle='', color=self.liste_couleurs[(cpt + color_base_index) % len(self.liste_couleurs)], alpha=marker_alpha, with_grid=with_grid, grid_style=grid_style, multi_index=multi_index)
            cpt += 1
        else:
            for h in np.sort(data[hue].unique()):
                self.add_plot(data[data[hue]==h][x], data[data[hue]==h][y], marker=marker, label=h, linestyle='', color=self.liste_couleurs[(cpt + color_base_index) % len(self.liste_couleurs)], alpha=marker_alpha, with_grid=with_grid, grid_style=grid_style, multi_index=multi_index)
                cpt += 1
        slope, intercept = st.linregress(data[x], data[y])[0:2]
        if reg_color is None:
            reg_color = self.liste_couleurs[(cpt + color_base_index) % len(self.liste_couleurs)]
        self.add_plot(data[x], intercept + slope*data[x], label="Régression linéaire", linestyle=linestyle, color=reg_color, with_grid=with_grid, grid_style=grid_style, subtitle=subtitle, legend=legend, multi_index=multi_index)
        self.set_axe_x(label=x, multi_index=multi_index)
        self.set_axe_y(label=y, multi_index=multi_index)
        self.fig.tight_layout()
        
    def graph_concentration(self, lorenz, gini, x_label, y_label, subtitle=None, with_grid='both', grid_style=":", print_gini=True, multi_index=1):
        self.add_plot(np.linspace(0,1,len(lorenz)), label="Première bissectrice", subtitle=subtitle, with_grid=with_grid, grid_style=grid_style, multi_index=multi_index)
        self.add_plot(lorenz, label="Courbe de Lorenz", with_grid=with_grid, grid_style=grid_style, multi_index=multi_index)
        self.set_axe_x(label=x_label, tick_min=-0.025*len(lorenz), tick_max=1.025*len(lorenz), multi_index=multi_index)
        self.set_axe_y(label=y_label, tick_min=-0.025, tick_max=1.025, tick_labels_format=':.0%', multi_index=multi_index)
        self.ax[multi_index-1].legend(loc="upper left")
        if print_gini:
            self.add_text(0.16*len(lorenz), 0.8, " Indice de Gini = {:.2f} ".format(gini), multi_index=multi_index)
        self.fig.tight_layout()
        
    def graph_droite_henry(self, data, subtitle=None, with_grid=None, grid_style=":", multi_index=1):
        self.add_sns_qqplot(data, subtitle=subtitle, multi_index=multi_index)    
        ymin, ymax = self.ax[multi_index-1].get_ylim()
        xmin, xmax = self.ax[multi_index-1].get_xlim()
        data_min = min(xmin, ymin)
        data_max = max(xmax, ymax)
        self.add_plot([data_min, data_max], [data_min, data_max], color=self.liste_couleurs[1], multi_index=multi_index)
        if with_grid is not None:
            self.ax[multi_index-1].grid(axis=with_grid, linestyle=grid_style)
        self.set_axe_x(label="Quantiles théoriques de la loi normale", multi_index=multi_index)
        self.set_axe_y(label="Quantiles observés des résidus", multi_index=multi_index)
        self.ax[multi_index-1].autoscale(enable=True, axis='both', tight=True)
        self.fig.tight_layout()

    def graph_compar_distrib_normal(self, data, label_data, bins=20, color_distrib=None, alpha_distrib=1, linewidth=1.5, color_loi_normale=None, show_mean=True, color_mean=None, show_median=False, color_median=None, subtitle=None, legend=True, with_grid=None, grid_style=":", multi_index=1):
        self.add_histogramme(data, labels=label_data, bins=bins, density=True, color=color_distrib, alpha=alpha_distrib, with_grid=with_grid, grid_style=grid_style, subtitle=subtitle, multi_index=multi_index)
        x_theo = np.linspace(data.min(), data.max(), 100)
        data_mean = data.mean()
        data_std = data.std(ddof=1)
        self.add_plot(x_theo, st.norm.pdf(x_theo, loc=data_mean, scale=data_std), label="Loi normale", color=color_loi_normale, linewidth=linewidth, with_grid=with_grid, grid_style=grid_style, multi_index=multi_index)
        if show_mean:
            self.add_line(coord=data_mean, color=color_mean, linewidth=linewidth, label="Moyenne des données", multi_index=multi_index)
        if show_median:
            data_median = data.median()
            self.add_line(coord=data_median, color=color_median, linewidth=linewidth, label="Médiane des données", multi_index=multi_index)
        self.set_axe_x(label=label_data, multi_index=multi_index)
        self.set_axe_y(label="Distribution (%)", multi_index=multi_index)
        if legend:
            self.ax[multi_index-1].legend(loc='upper right')
        self.fig.tight_layout()
        
    def graph_compar_distrib_binomial(self, freq_empirique, n, p, label_data, color_distrib=None, alpha_distrib=1, linewidth=1.5, color_loi_binomiale=None, color_mean=None, subtitle=None, with_grid=None, grid_style=":", multi_index=1):
        self.add_barv(range(n+1), freq_empirique, label=label_data, color=color_distrib, alpha=alpha_distrib, subtitle=subtitle, with_grid=with_grid, grid_style=grid_style, multi_index=multi_index)
        x_theo = range(n+1)
        freq_theo = st.binom.pmf(x_theo, n, p) * freq_empirique.sum()
        data_mean = 0.0
        for i in x_theo:
            data_mean += (freq_empirique[i] * i)
        data_mean = data_mean / freq_empirique.sum()
        self.add_plot(x_theo, freq_theo, color=color_loi_binomiale, linewidth=linewidth, label="Distribution de la loi binomiale", with_grid=with_grid, grid_style=grid_style, multi_index=multi_index)
        self.add_line(coord=data_mean, color=color_mean, linewidth=linewidth, label="Moyenne des données", multi_index=multi_index)
        self.ax[multi_index-1].legend(loc='upper right')
        self.fig.tight_layout()
    
    def graph_contingency(self, table_values, annotations, x_label, y_label, chi_n, cmap='Blues', x_tick_labels=None, y_tick_labels=None, vmin=None, vmax=None, multi_index=1):
        self.add_sns_heatmap(table_values, annotations=annotations, cmap=cmap, vmin=vmin, vmax=vmax, subtitle="La couleur de fond indique la contribution au chi-n ({:,.2f})".format(chi_n), multi_index=multi_index)
        self.set_axe_x(label=x_label, tick_labels=x_tick_labels, multi_index=multi_index)
        self.set_axe_y(label=y_label, tick_labels=y_tick_labels, rotation=0, multi_index=multi_index)
        self.fig.tight_layout()
    
    def graph_confusion_matrix(self, y_true, y_pred, labels=None, xticks_rotation=0, yticks_rotation=0, labels_capitalize=False, normalize=None, include_values=True, values_format='.0f', cmap=plt.cm.Blues, cbar=True, subtitle=None, multi_index=1):
        if labels is None:
            labels = np.unique([y_true,y_pred])
        conf_matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
        self.add_sns_heatmap(conf_matrix, annotations=include_values, cmap=cmap, fmt=values_format, cbar=cbar, subtitle=subtitle, multi_index=multi_index)
        if labels_capitalize:
            labels = np.char.capitalize(labels.astype(np.unicode_))
        self.set_axe_x(label="Prédictions", tick_labels=labels, rotation=xticks_rotation, multi_index=multi_index)
        self.set_axe_y(label="Valeurs réelles", tick_labels=labels, rotation=yticks_rotation, multi_index=multi_index)
        self.ax[multi_index-1].spines['right'].set_visible(True)
        self.ax[multi_index-1].spines['left'].set_visible(True)
        self.ax[multi_index-1].spines['top'].set_visible(True)
        self.ax[multi_index-1].spines['bottom'].set_visible(True)
        
    """
    ACP : scree plot
    """
    def graph_eboulis_valeurspropres(self, scree, seuil_inertie=0.90, x_rotation=0, with_grid='y', grid_style=':', subtitle=None, multi_index=1):
        x = np.arange(len(scree)) + 1
        if len(x) > 10:
            tick_step = int(round(len(x)/10))
        else:
            tick_step = 1
        self.add_barv(x, scree, color=self.liste_couleurs[0], with_grid=with_grid, grid_style=grid_style, multi_index=multi_index)
        self.add_plot(x, np.cumsum(scree), marker='o', color=self.liste_couleurs[1], with_grid=with_grid, grid_style=grid_style, subtitle=subtitle, multi_index=multi_index)
        if seuil_inertie is not None:
            #x_inertie = np.searchsorted(np.cumsum(scree), seuil_inertie) + 1
            x_inertie = np.argmax(np.cumsum(scree) >= seuil_inertie) + 1
            self.add_line(coord=seuil_inertie, vertical=False, line_max=(x_inertie-0.5)/len(x), color='k', linestyle='--', multi_index=multi_index)
            self.add_line(coord=x_inertie, vertical=True, line_max=seuil_inertie, color='k', linestyle='--', multi_index=multi_index)
        self.set_axe_x(label="Rang de l'axe d'inertie", tick_min=0.5, tick_max=len(x)+0.5, multi_index=multi_index)
        self.ax[multi_index-1].xaxis.set_ticks(np.arange(1, len(x)+1, step=tick_step))
        self.ax[multi_index-1].set_xticklabels(np.arange(1, len(x)+1, step=tick_step), rotation=x_rotation)
        self.set_axe_y(label="Pourcentage d'inertie expliquée", tick_min=0, tick_max=1, multi_index=multi_index)
    
    """
    ACP : broken sticks
    """    
    def graph_batons_brises(self, df_batons_brises, x_rotation=0, plot_seuil=True, with_grid='y', grid_style=':', subtitle=None, multi_index=1):
        x = np.arange(len(df_batons_brises)) + 1
        if len(x) > 10:
            tick_step = int(round(len(x)/10))
        else:
            tick_step = 1
        self.add_barv(x, df_batons_brises.explained_variance.values, label="Valeurs propres", color=self.liste_couleurs[0], with_grid=with_grid, grid_style=grid_style, multi_index=multi_index)
        self.add_plot(x, df_batons_brises.seuils, marker='o', label="Bâtons brisés", color=self.liste_couleurs[1], with_grid=with_grid, grid_style=grid_style, subtitle=subtitle, legend=True, multi_index=multi_index)
        if plot_seuil:
            seuil = np.argmax(np.array(df_batons_brises.explained_variance < df_batons_brises.seuils))
            self.add_line(coord=df_batons_brises.seuils[seuil-1], vertical=False, line_max=(seuil-0.5)/len(x), color='k', linestyle='--', multi_index=multi_index)
            self.add_line(coord=seuil, vertical=True, line_max=df_batons_brises.seuils[seuil-1]/self.ax[multi_index-1].get_ylim()[1], color='k', linestyle='--', multi_index=multi_index)
        self.set_axe_x(label="Rang de l'axe d'inertie", tick_min=0.5, tick_max=len(x)+0.5, multi_index=multi_index)
        self.ax[multi_index-1].xaxis.set_ticks(np.arange(1, len(x)+1, step=tick_step))
        self.ax[multi_index-1].set_xticklabels(np.arange(1, len(x)+1, step=tick_step), rotation=x_rotation)
        self.set_axe_y(label="Variance expliquée", multi_index=multi_index)
    
    """
    ACP : correlation circles (features projection)   ADD COLOR, ALPHA
    """        
    def graph_correlation_circles(self, df_variables_correlation, variables_projected_index, idx1=0, idx2=1, show_labels=True, labels_fontsize=None, labels_rotation=0, labels_color="blue", labels_alpha=0.5, 
                                  color='grey', alpha=1, limits=None, axis_linestyle='--', axis_color='grey', with_grid='both', grid_style=':', subtitle=None, multi_index=1):
        if limits is not None:
            xmin, xmax, ymin, ymax = limits
        #elif df_variables_correlation.shape[0] >= 30 :
        #    xmin, xmax, ymin, ymax = min(df_variables_correlation.iloc[:,idx1]), max(df_variables_correlation.iloc[:,idx1]), min(df_variables_correlation.iloc[:,idx2]), max(df_variables_correlation.iloc[:,idx2])
        else:
            xmin, xmax, ymin, ymax = -1, 1, -1, 1
        self.add_plot([-1, 1], [0, 0], linestyle=axis_linestyle, color=axis_color, with_grid=with_grid, grid_style=grid_style, subtitle=subtitle, multi_index=multi_index)  # affichage du repère horizontal
        self.add_plot([0, 0], [-1, 1], linestyle=axis_linestyle, color=axis_color, with_grid=with_grid, grid_style=grid_style, subtitle=subtitle, multi_index=multi_index)  # affichage du repère vertical
        if df_variables_correlation.shape[0] < 30 :  # affichage des flèches ; s'il y a plus de 30 flèches, on n'affiche que les lignes
            self.add_fleches(np.zeros(df_variables_correlation.shape[0]), np.zeros(df_variables_correlation.shape[0]), df_variables_correlation.iloc[:,idx1].values, df_variables_correlation.iloc[:,idx2].values, color=color, alpha=alpha, multi_index=multi_index)
        else:
            lines = [[[0,0],[x,y]] for x,y in df_variables_correlation.iloc[:,[idx1,idx2]].values]
            self.objets.append(self.ax[multi_index-1].add_collection(LineCollection(lines, axes=self.ax[multi_index-1], alpha=alpha, color=color)))
            self.objets_desc.append("Lignes - axe {}".format(multi_index-1))
        if show_labels:  # affichage des noms des variables
            for i, (x, y) in enumerate(df_variables_correlation.iloc[:,[idx1,idx2]].values):
                if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                    if abs(x) > abs(y):
                        ha = 'left' if x > 0 else 'right'
                        va = 'center'
                    else:
                        ha = 'center'
                        va = 'bottom' if y > 0 else 'top'
                    self.add_text(x + np.sign(x)*(xmax - xmin)*0.005, y + np.sign(y)*(ymax - ymin)*0.005, df_variables_correlation.index[i], fontsize=labels_fontsize, ha=ha, va=va, rotation=labels_rotation, color=labels_color, alpha=labels_alpha, backgroundcolor=(1,1,1), backgroundalpha=0, multi_index=multi_index)
        self.add_cercle(0, 0, 1, facecolor='none', edgecolor='b', multi_index=multi_index)  # affichage du cercle
        self.set_axe_x(label=variables_projected_index[idx1], tick_min=xmin, tick_max=xmax, multi_index=multi_index)
        self.set_axe_y(label=variables_projected_index[idx2], tick_min=ymin, tick_max=ymax, multi_index=multi_index)
        if limits is None:
            self.ax[multi_index-1].axis('scaled')
        self.ax[multi_index-1].spines['left'].set_visible(False)
        self.ax[multi_index-1].spines['bottom'].set_visible(False)
    
    """
    ACP : factorial planes (data or centroids projection)
    """
    def graph_factorial_planes(self, df_data_projected, marker='x', marker_alpha=0.75, marker_color=None, idx1=0, idx2=1, show_labels=False, labels_fontsize=None, labels_rotation=0, labels_color="blue", \
                               labels_alpha=0.5, hue=None, hue_legend_title=None, hue_color_base_index=None, limits=None, axis_linestyle='--', axis_color='grey', with_grid='both', grid_style=':', subtitle=None, multi_index=1):
        if limits is not None:
            xmin, xmax, ymin, ymax = limits
        else:
            boundary = np.max(np.abs(df_data_projected.iloc[:, [idx1, idx2]].values)) * 1.1
            xmin, xmax, ymin, ymax = -boundary, boundary, -boundary, boundary
        self.add_plot([xmin, xmax], [0, 0], linestyle=axis_linestyle, color=axis_color, with_grid=with_grid, grid_style=grid_style, subtitle=subtitle, multi_index=multi_index)  # affichage du repère horizontal
        self.add_plot([0, 0], [ymin, ymax], linestyle=axis_linestyle, color=axis_color, with_grid=with_grid, grid_style=grid_style, subtitle=subtitle, multi_index=multi_index)  # affichage du repère vertical
        if hue is None:
            self.add_plot(df_data_projected.iloc[:, idx1].values, df_data_projected.iloc[:, idx2].values, marker=marker, color=marker_color, alpha=marker_alpha, linestyle='', with_grid=with_grid, grid_style=grid_style, subtitle=subtitle, multi_index=multi_index)
        else:
            hue_values = np.array(df_data_projected[hue].values)
            cpt = 0
            for value in np.unique(hue_values):
                selected = np.where(hue_values == value, True, False)
                if hue_color_base_index is None:
                    color = self.liste_couleurs[(cpt + multi_index - 1) % len(self.liste_couleurs)]
                else:
                    color = self.liste_couleurs[(cpt + hue_color_base_index) % len(self.liste_couleurs)]
                self.add_plot(df_data_projected[selected].iloc[:, idx1].values, df_data_projected[selected].iloc[:, idx2].values, label=value, marker=marker, color=color, alpha=marker_alpha, linestyle='', with_grid=with_grid, grid_style=grid_style, subtitle=subtitle, legend=True, multi_index=multi_index)
                cpt = cpt + 1
            if hue_legend_title is not None:
                self.set_legend(title=hue_legend_title, multi_index=multi_index)
        if show_labels:  # affichage des labels des points
            for i, (x, y) in enumerate(df_data_projected.iloc[:,[idx1,idx2]].values):
                if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                    if abs(x) > abs(y):
                        ha = 'left' if x > 0 else 'right'
                        va = 'center'
                    else:
                        ha = 'center'
                        va = 'bottom' if y > 0 else 'top'
                    self.add_text(x + np.sign(x)*(xmax - xmin)*0.005, y + np.sign(y)*(ymax - ymin)*0.005, df_data_projected.index[i], fontsize=labels_fontsize, ha=ha, va=va, rotation=labels_rotation, color=labels_color, alpha=labels_alpha, backgroundcolor=(1,1,1), backgroundalpha=0, multi_index=multi_index)
        self.set_axe_x(label=df_data_projected.columns[idx1], tick_min=xmin, tick_max=xmax, multi_index=multi_index)
        self.set_axe_y(label=df_data_projected.columns[idx2], tick_min=ymin, tick_max=ymax, multi_index=multi_index)
        
    """
    Clustering : choix n_clusters basé sur l'inertie
    """
    def graph_choix_nclusters_inertia(self, nclusters_list, inertia_list):
        self.add_plot(nclusters_list, inertia_list, with_grid="y", grid_style=':')
        self.set_axe_x(label="Nombre de clusters", tick_min=nclusters_list[0]-0.5, tick_max=nclusters_list[-1]+0.5)
        tick_step = 1
        if nclusters_list[-1] - nclusters_list[0] + 1 > 10:
            tick_step = int(round((nclusters_list[-1] - nclusters_list[0] + 1.0) / 10))
        self.set_axe_x(tick_min=nclusters_list[0], tick_max=nclusters_list[-1], tick_step=tick_step, tick_labels_format=':.0f')
        self.set_axe_y(label="Inertie intra-classe")
    
    """
    Clustering : choix n_clusters basé sur le coefficient de silhouette
    """
    def graph_choix_nclusters_silhouette(self, nclusters_list, silhouette_avg_score_list):
        self.add_barv(nclusters_list, silhouette_avg_score_list)
        self.set_axe_x(label="Nombre de clusters", tick_min=nclusters_list[0]-0.5, tick_max=nclusters_list[-1]+0.5)
        tick_step = 1
        if nclusters_list[-1] - nclusters_list[0] + 1 > 10:
            tick_step = int(round((nclusters_list[-1] - nclusters_list[0] + 1.0) / 10))
        self.set_axe_x(tick_min=nclusters_list[0], tick_max=nclusters_list[-1], tick_step=tick_step, tick_labels_format=':.0f')
        self.set_axe_y(label="Coefficient de silhouette moyen")
        
    """
    Clustering : détail des coefficients de silhouette
    """
    def graph_silhouette_detail(self, nclusters, cluster_labels, silhouette_avg_score, sample_silhouette_values,
                                min_silhouette=-1, max_silhouette=1, multi_index=1):
        y_lower = 10
        nb_colors = len(self.liste_couleurs)
        for i in range(nclusters):
            cluster_silhouette_values = sample_silhouette_values[cluster_labels==i].copy()
            cluster_size = cluster_silhouette_values.shape[0]
            y_upper = y_lower + cluster_size
            self.add_area(np.arange(y_lower, y_upper), 
                          0,
                          sorted(cluster_silhouette_values),
                          orientation='horizontal',
                          color=self.liste_couleurs[i % nb_colors],
                          alpha=0.7,
                          subtitle="{} clusters : {:.3f}".format(nclusters, silhouette_avg_score),
                          multi_index=multi_index)
            self.add_text(-0.05, y_lower + 0.5 * cluster_size, 
                          str(i),
                          backgroundalpha=0,
                          multi_index=multi_index)
            y_lower = y_upper + 10  # 10 for the 0 samples
        self.set_axe_x(label="Valeurs du coefficient de silhouette",
                       tick_min=min_silhouette, 
                       tick_max=max_silhouette,
                       multi_index=multi_index)
        self.set_axe_y(label="Nomnre d'individus",
                       multi_index=multi_index)
        self.add_line(silhouette_avg_score, 
                      color="red", 
                      linestyle="--",
                      multi_index=multi_index)
        
    """
    Clustering : analyse des coefficients de silhouette
    """
    def graph_silhouette_analysis(self, nclusters, cluster_labels, silhouette_avg_score, sample_silhouette_values,
                                  data_visualisation, min_silhouette=-1, max_silhouette=1, clust_figsize=(12,6), plt_mult_param=0.8):
        #graph de gauche : analyse des silhouettes
        y_lower = 10
        for i in range(nclusters):
            cluster_silhouette_values = sample_silhouette_values[cluster_labels==i].copy()
            cluster_size = cluster_silhouette_values.shape[0]
            y_upper = y_lower + cluster_size
            self.add_area(np.arange(y_lower, y_upper), 0, sorted(cluster_silhouette_values),
                           orientation='horizontal', color=self.liste_couleurs[i],
                           alpha=0.7, subtitle="Silhouette plot pour chaque cluster")
            self.add_text(-0.05, y_lower + 0.5 * cluster_size, str(i), backgroundalpha=0)
            y_lower = y_upper + 10  # 10 for the 0 samples
        self.set_axe_x(label="Valeurs du coefficient de silhouette", tick_min=min_silhouette, tick_max=max_silhouette)
        self.set_axe_y(label="Nombre d'individus")
        self.add_line(silhouette_avg_score, color="red", linestyle="--")
        #graph de droite : visualisation des clusters
        dataviz = data_visualisation.copy()
        dataviz["cluster"] = cluster_labels
        clusters_center = dataviz.groupby("cluster").mean()
        for i in range(nclusters):
            self.add_scatter(dataviz[dataviz.cluster==i].iloc[:,0], dataviz[dataviz.cluster==i].iloc[:,1],
                              marker='.', uniquecolor=self.liste_couleurs[i], size=30, alpha=0.7, multi_index=2)
        self.add_scatter(clusters_center.iloc[:, 0], clusters_center.iloc[:, 1], marker='o',
                          uniquecolor="white", size=200, subtitle="Visualisation des clusters", multi_index=2)
        for i in range(nclusters):
            self.add_text(clusters_center.iloc[i, 0], clusters_center.iloc[i, 1], str(i), 
                           color=self.liste_couleurs[i], fontsize=12, ha="center", va="center",
                           backgroundalpha=0, multi_index=2)
        self.set_axe_x(label="Premier axe de projection des données", multi_index=2)
        self.set_axe_y(label="Deuxième axe de projection des données", multi_index=2)
        
    """
    si multi_index == 0, on ajoute du text à fig ; sinon le texte concerne l'axe d'indice multi_index-1
    """
    def add_text(self, x_coord, y_coord, text, ha='center', va='top', color='black', fontsize=None, rotation=0, alpha=1, backgroundcolor=(0.95, 0.95, 0.95), backgroundalpha=1, multi_index=1):
        if multi_index == 0:
            self.objets.append(self.fig.text(x_coord, y_coord, text, horizontalalignment=ha, verticalalignment=va, color=color, fontsize=fontsize, rotation=rotation, alpha=alpha, backgroundcolor=backgroundcolor, bbox={'alpha':backgroundalpha}))
            self.objets_desc.append("Texte - figure")
        else:
            self.objets.append(self.ax[multi_index-1].text(x_coord, y_coord, text, horizontalalignment=ha, verticalalignment=va, color=color, fontsize=fontsize, backgroundcolor=backgroundcolor, bbox={'alpha':backgroundalpha}))
            self.objets_desc.append("Texte - axe {}".format(multi_index-1))
        
    def add_line(self, coord, vertical=True, label=None, line_min=0, line_max=1, linestyle="-", color=None, alpha=1, linewidth=1.5, legend=False, multi_index=1):
        if vertical:
            self.objets.append(self.ax[multi_index-1].axvline(x=coord, ymin=line_min, ymax=line_max, label=label, linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth))
            self.objets_desc.append("Ligne verticale - axe {}".format(multi_index-1))
        else:
            self.objets.append(self.ax[multi_index-1].axhline(y=coord, xmin=line_min, xmax=line_max, label=label, linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth))
            self.objets_desc.append("Ligne horizontale - axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, None, False)
        
    def add_rectangle(self, coord_min, coord_max, vertical=True, label=None, rect_min=0, rect_max=1, linestyle="-", color=None, alpha=1, linewidth=1.5, fill=False, multi_index=1):
        if vertical:
            self.objets.append(self.ax[multi_index-1].axvspan(xmin=coord_min, xmax=coord_max, ymin=rect_min, ymax=rect_max, label=label, linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, fill=fill))
            self.objets_desc.append("Rectangle vertical - axe {}".format(multi_index-1))
        else:
            self.objets.append(self.ax[multi_index-1].axhspan(ymin=coord_min, ymax=coord_max, xmin=rect_min, xmax=rect_max, label=label, linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth, fill=fill))
            self.objets_desc.append("Rectangle horizontal - axe {}".format(multi_index-1))
            
    def add_cercle(self, x_center, y_center, radius, edgecolor=None, facecolor=None, alpha=1, linestyle='-', linewidth=1.5, multi_index=1):
        self.objets.append(self.ax[multi_index-1].add_artist(plt.Circle((x_center, y_center), radius=radius, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha, linestyle=linestyle, linewidth=linewidth)))
        self.objets_desc.append("Cercle - axe {}".format(multi_index-1))
        
    def add_fleches(self, x_origins, y_origins, x_arrows, y_arrows, units='width', angles='xy', scale_units='xy', scale=1, color=None, alpha=1, linestyle='-', multi_index=1):
        self.objets.append(self.ax[multi_index-1].quiver(x_origins, y_origins, x_arrows, y_arrows, units=units, angles=angles, scale_units=scale_units, scale=scale, color=color, alpha=alpha, linestyle=linestyle))
        self.objets_desc.append("Flèches - axe {}".format(multi_index-1))
        
    
    """
    si multi_index == 0, on ajoute une légende à fig ; sinon la légende concerne l'axe d'indice multi_index-1
    """
    def set_legend(self, loc=None, bbox_to_anchor=None, ncol=1, title=None, fontsize=None, multi_index=1):
        if multi_index == 0:
            self.fig.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol, fontsize=fontsize, title=title)
        else:
            self.ax[multi_index - 1].legend(loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol, fontsize=fontsize, title=title)
    
    """
    label_format : par exemple : ':.0%' ; ':.2%' ; ':,.0f', ':.3f'
    """    
    def set_axe_x(self, label=None, label_position=(0.5,0.5), tick_min=None, tick_max=None, tick_step=None, tick_labels=None, tick_labels_format=None, tick_dash=False, color=None, rotation=None, ha=None, va=None, multi_index=1):
        if label is not None:
            self.ax[multi_index - 1].set_xlabel(label, position=label_position)
        if tick_min is None:
            tick_min = self.ax[multi_index - 1].get_xlim()[0]
        if tick_max is None:
            tick_max = self.ax[multi_index - 1].get_xlim()[1]
        if tick_step is not None:
            self.ax[multi_index - 1].xaxis.set_ticks(np.arange(tick_min, tick_max + (tick_step/10), tick_step))
            self.ax[multi_index - 1].set_xticklabels(np.arange(tick_min, tick_max + (tick_step/10), tick_step))
        else:
            if (tick_min is not None) | (tick_max is not None):
                self.ax[multi_index - 1].set_xlim([tick_min, tick_max])
        if tick_labels is not None:
            self.ax[multi_index - 1].set_xticklabels(tick_labels)
        if tick_labels_format is not None:
            myformat = '{x' + tick_labels_format + '}'
            self.ax[multi_index - 1].xaxis.set_major_formatter(plticker.StrMethodFormatter(myformat))
        if tick_dash:
            self.ax[multi_index - 1].xaxis.set_tick_params(bottom=True)
        if (color is not None) | (rotation is not None) | (ha is not None) | (va is not None):
            for label in self.ax[multi_index - 1].get_xticklabels():
                if color is not None:
                    label.set_color(color)
                if rotation is not None:
                    label.set_rotation(rotation)
                if ha is not None:
                    label.set_ha(ha)
                if va is not None:
                    label.set_va(va)
    
    def set_axe_y(self, label=None, label_position=(0.5,0.5), tick_min=None, tick_max=None, tick_step=None, tick_labels=None, tick_labels_format=None, tick_dash=False, color=None, rotation=None, ha=None, va=None, multi_index=1, twinx=False):
        if label is not None:
            self.ax[multi_index - 1].set_ylabel(label, position=label_position)
        if tick_min is None:
            tick_min = self.ax[multi_index - 1].get_ylim()[0]
        if tick_max is None:
            tick_max = self.ax[multi_index - 1].get_ylim()[1]
        if tick_step is not None:
            self.ax[multi_index - 1].yaxis.set_ticks(np.arange(tick_min, tick_max + (tick_step/10), tick_step))
            self.ax[multi_index - 1].set_yticklabels(np.arange(tick_min, tick_max + (tick_step/10), tick_step))
            #self.ax[multi_index - 1].set_ylim([tick_min, tick_max])
        else:
            if (tick_min is not None) | (tick_max is not None):
                self.ax[multi_index - 1].set_ylim([tick_min, tick_max])
        if tick_labels is not None:
            self.ax[multi_index - 1].set_yticklabels(tick_labels)
        if tick_labels_format is not None:
            myformat = '{x' + tick_labels_format + '}'
            self.ax[multi_index - 1].yaxis.set_major_formatter(plticker.StrMethodFormatter(myformat))
        if tick_dash:
            if not twinx:
                self.ax[multi_index - 1].yaxis.set_tick_params(left=True)
            else:
                self.ax[multi_index - 1].yaxis.set_tick_params(right=True)
        if (color is not None) | (rotation is not None) | (ha is not None) | (ha is not None):
            for label in self.ax[multi_index - 1].get_yticklabels():
                if color is not None:
                    label.set_color(color)
                if rotation is not None:
                    label.set_rotation(rotation)
                if ha is not None:
                    label.set_ha(ha)
                if va is not None:
                    label.set_va(va)


def graph_densite_remplissage(df, title=None, y_title=1.05, label_fontsize=14, black_color=False, color_palette=None, color_index=0, figsize=(12,8)):
    if title is None:
        title = "Densité des données renseignées"
    graph= MyGraph(title, y_title=y_title, color_palette=color_palette, figsize=figsize)
    if black_color:
        color = (0.25, 0.25, 0.25)
    else:
        color = graph.liste_couleurs[color_index]
    msno.matrix(df, color=color, sparkline=False, ax=graph.ax[0])
    ticklabels = graph.ax[0].xaxis.get_ticklabels()
    graph.ax[0].xaxis.set_ticks_position('bottom')
    graph.ax[0].xaxis.set_ticks_position('none')
    graph.ax[0].set_xticklabels(ticklabels, rotation=45, ha='right', fontsize=label_fontsize)
    graph.fig.tight_layout()
    return graph.fig, graph.ax

def graph_analyse_variable_quant(data_serie, nom_variable, label_x_boxplot, title=None, color_palette=None, figsize=(12,9)):
    if title is None:
        title = "Distribution de la variable {}".format(nom_variable)
    graph = MyGraph(title, nblin=3, nbcol=2, color_palette=color_palette, figsize=figsize, gridspec=True)
    graph.ax.append(graph.fig.add_subplot(graph.gs[0, :]))
    graph.ax.append(graph.fig.add_subplot(graph.gs[1:, 0]))
    graph.ax.append(graph.fig.add_subplot(graph.gs[1:, 1]))
    graph.add_boxplot([data_serie], [nom_variable], means=True, vertical=False, subtitle="Boxplot", multi_index=1)
    graph.set_axe_x(label=label_x_boxplot, multi_index=1)
    graph.graph_droite_henry(data_serie, subtitle="Droite de Henry", multi_index=2)
    graph.graph_compar_distrib_normal(data_serie, nom_variable, subtitle="Histogramme de distribution", show_mean=False, multi_index=3)
    return graph.fig, graph.ax

def graph_analyse_variable_categ(data_serie, nom_variable, title=None, index_as_int=False, alternatif_value_counts=None, color_palette=None, ticklabels_fontsize=None, ticklabels_rotation=None, force_pie=False, force_startangle=None, figsize=(12,8)):
    if title is None:
        title = "Distribution de la variable {}".format(nom_variable)
    if alternatif_value_counts is None:
        distrib_values = data_serie.value_counts()
    else:
        distrib_values = alternatif_value_counts
    if index_as_int:
        labels = distrib_values.index.values.astype(int)
    else:
        labels = distrib_values.index
    if force_startangle is None:
        force_startangle = 90
    if ticklabels_rotation is None:
        ticklabels_rotation = 30
    if (ticklabels_rotation > 0) & (ticklabels_rotation < 90):
        ha='right'
    else:
        ha='center'
    graph = MyGraph(title, color_palette=color_palette, figsize=figsize)
    if (len(distrib_values) <= 5) | force_pie:
        graph.add_pie(distrib_values.values, labels, startangle=force_startangle, counterclock=False)
    else:
        graph.add_barv(labels, distrib_values.values)
        graph.set_axe_x(rotation=ticklabels_rotation, ha=ha)
        graph.set_axe_y(label="Nombre d'enregistrements")
        if ticklabels_fontsize is not None:
            ticklabels = graph.ax[0].xaxis.get_ticklabels()
            graph.ax[0].set_xticklabels(ticklabels, fontsize=ticklabels_fontsize)
    return graph.fig, graph.ax

"""
hue must be a string value (not numeric nor boolean)
avec version 0.10, ajouter corner
"""
def sns_pairplot(data, hue=None, title=None, y_title=1.0, palette=None, kind='scatter', diag_kind='auto', indiv_height=2.5):
    sns.set_style("white")
    mydefault_plt_parameters()
    liste_couleurs = set_colors(False, palette)
    g = sns.pairplot(data, hue=hue, palette=palette, kind=kind, diag_kind=diag_kind, height=indiv_height)
    fig = g.fig
    if title is not None:
        fig.suptitle(title, y=y_title)
    return fig, g

def sns_jointplot(x, y, data, title=None, palette=None, kind='reg', color_index=0, height=8):
    sns.set_style("white")
    mydefault_plt_parameters()
    liste_couleurs = set_colors(False, palette)
    g = sns.jointplot(x=x, y=y, data=data, kind=kind, color=liste_couleurs[color_index], height=height)
    fig = g.fig
    if title is not None:
        fig.suptitle(title, y=1.05)
    return fig, fig.axes

def sns_catplot(x, y, hue, data, title=None, palette=None, kind='strip', order=None, hue_order=None, orient=None, legend=True, legend_out=True, figsize=(12,8)):
    height = figsize[1]
    aspect = figsize[0] / figsize[1]
    sns.set_style("white")
    mydefault_plt_parameters()
    liste_couleurs = set_colors(False, palette)
    g = sns.catplot(x, y, hue, data, kind=kind, order=order, hue_order=hue_order, orient=orient, height=height, aspect=aspect, legend=legend, legend_out=legend_out)
    fig = g.fig
    if title is not None:
        fig.suptitle(title, y=1.05)
    return fig, fig.axes
    

class FullGraphAuto:
    
    """
    checkgraph = sfg.MyGraph(title, nblin=max_diff+1, nbcol=2, figsize=(12,hauteur_fig))
    sfg.FullGraph(title, figsize=(12,hauteur_fig))
    checkgraph.graph_sarimax_autocorrelations(df_diff, max_diff=max_diff, lags=lags, season_freq=season_freq)
    """
    def graph_sarimax_autocorrelations(self, my_serie, max_diff=2, lags=40, season_freq=0):
        # initialisation du graph
        nblin = max_diff + 1
        nbcol = 2
        self.fig = plt.figure()
        cpt = 0
        self.fig.subplots_adjust(wspace=0.25, hspace=0.25)
        for lin in np.arange(nblin):
            for col in np.arange(nbcol):
                cpt += 1
                self.ax.append(plt.subplot(nblin, nbcol, cpt))
        self.fig.suptitle(self.title, y=1.05)
        # préparation des données
        df_diff = my_serie.copy()
        for i in np.arange(1, max_diff+1):
            df_diff["diff{}".format(i)] = np.append(np.zeros(i), np.diff(df_diff[df_diff.columns[0]].values, n=i))
        if season_freq <= 0:
            season_freq = 0
        else:
            for i in np.arange(max_diff+1):
                df_diff[df_diff.columns[i]] = df_diff[df_diff.columns[i]].diff(season_freq)
        # génération des charts
        for i in np.arange(max_diff+1):
            sm.graphics.tsa.plot_acf(df_diff[df_diff.columns[i]].iloc[i+season_freq:], lags=lags, title="", ax=self.ax[i*2])
            self.ax[i*2].spines['right'].set_visible(False)
            self.ax[i*2].spines['top'].set_visible(False)
            self.ax[i*2].grid(b=True, axis='x')
            self.ax[i*2].yaxis.set_tick_params(left=True)
            self.set_axe('y', label="diff={}".format(i), multi_index=i*2+1)
            if i == max_diff:
                self.set_axe('x', label="lags", label_position=(1.1,0.5), multi_index=i*2+1)
            sm.graphics.tsa.plot_pacf(df_diff[df_diff.columns[i]].iloc[i+season_freq:], lags=lags, title="", ax=self.ax[i*2+1])
            self.ax[i*2+1].spines['right'].set_visible(False)
            self.ax[i*2+1].spines['top'].set_visible(False)
            self.ax[i*2+1].grid(b=True, axis='x')
            self.ax[i*2+1].yaxis.set_tick_params(left=True)
        self.ax[0].set_title("Autocorrélations simples", fontweight='normal')
        self.ax[1].set_title("Autocorrélations partielles", fontweight='normal')
        self.fig.tight_layout()

    """
    checkgraph = sfg.MyGraph(title, nblin=3, nbcol=2, figsize=(12,9), gridspec=True)  
    sfg.FullGraph(title, figsize=(12,9))
    checkgraph.graph_sarimax_analysis(my_serie, my_index, centered=centered)        
    """
    def graph_sarimax_analysis(self, my_serie, my_index, centered=False, lags=40):
        # initialisation du graph
        nblin = 3
        nbcol = 2
        self.fig = plt.figure()
        self.fig.subplots_adjust(wspace=0.25, hspace=0.25)
        self.gs = plt.GridSpec(nblin, nbcol, figure=self.fig)
        self.fig.suptitle(self.title, y=1.05)
        self.ax.append(self.fig.add_subplot(self.gs[0, :]))
        self.ax.append(self.fig.add_subplot(self.gs[1, :-1]))
        self.ax.append(self.fig.add_subplot(self.gs[1, -1]))
        self.ax.append(self.fig.add_subplot(self.gs[2, :-1]))
        self.ax.append(self.fig.add_subplot(self.gs[2, -1]))
        # chart 1
        self.add_plot(my_index, my_serie, label="", subtitle="Graphique des valeurs", with_grid='x', multi_index=1)
        if centered:
            self.add_line(0, vertical=False, color=self.liste_couleurs[1], multi_index=1)
            tick_min = self.ax[0].get_ylim()[0]
            tick_max = self.ax[0].get_ylim()[1]
            absmax = abs(max(tick_min, tick_max))
            self.ax[0].set_ylim([-absmax, absmax])
        self.ax[0].yaxis.set_tick_params(left=True)
        # chart 2
        sm.graphics.tsa.plot_acf(my_serie, lags=lags, ax=self.ax[1])
        self.ax[1].set_title("Autocorrélations simples", fontweight='normal')
        self.ax[1].spines['right'].set_visible(False)
        self.ax[1].spines['top'].set_visible(False)
        self.ax[1].grid(b=True, axis='x')
        self.ax[1].yaxis.set_tick_params(left=True)
        # chart 3
        sm.graphics.tsa.plot_pacf(my_serie, lags=lags, ax=self.ax[2])
        self.ax[2].set_title("Autocorrélations partielles", fontweight='normal')
        self.ax[2].spines['right'].set_visible(False)
        self.ax[2].spines['top'].set_visible(False)
        self.ax[2].grid(b=True, axis='x')
        self.ax[2].yaxis.set_tick_params(left=True)
        # chart 4
        sm.qqplot(my_serie, fit=True, markeredgecolor=self.liste_couleurs[0], markerfacecolor=self.liste_couleurs[0], alpha=0.5, ax=self.ax[3])
        ymin, ymax = self.ax[3].get_ylim()
        xmin, xmax = self.ax[3].get_xlim()
        data_min = min(xmin, ymin)
        data_max = max(xmax, ymax)
        self.add_plot([data_min, data_max], [data_min, data_max], label="", color=self.liste_couleurs[1], multi_index=4)
        self.ax[3].set_xlabel('')
        self.ax[3].set_ylabel('')
        self.ax[3].set_title("QQ Plot", fontdict={'fontweight': 'normal'})
        self.ax[3].xaxis.set_tick_params(bottom=True)
        self.ax[3].yaxis.set_tick_params(left=True)
        # chart 5
        self.graph_compar_distrib_normal(my_serie.values, label_data="Distribution", color_mean=self.liste_couleurs[2], subtitle="Histogramme de distribution", legend=False, multi_index=5)
        self.ax[4].xaxis.set_tick_params(bottom=True)
        self.ax[4].yaxis.set_tick_params(left=True)
        self.fig.tight_layout()