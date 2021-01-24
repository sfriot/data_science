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
import seaborn as sns
import scipy.stats as st

# changes in matplotlib default parameters
def mydefault_plt_parameters(figsize=(12,8)):
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

def set_colors(is_mono):
    if is_mono:
        sns.set_palette(sns.light_palette("navy"))
    else:
        sns.set_palette("Set2")
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
def calcul_lorenz_gini(listdata):
    lorenz = np.cumsum(np.sort(listdata)) / listdata.sum()
    lorenz = np.append([0], lorenz)
    aire_sous_courbe = lorenz[:-1].sum() / len(lorenz)
    gini = 2 * (0.5 - aire_sous_courbe)
    return lorenz, gini
"""

def calcul_lorenz_gini(listdata):
    lorenz = np.append([0], np.cumsum(np.sort(listdata)) / listdata.sum())
    aire_ss_courbe = lorenz[:-1].sum()/len(lorenz)  # aire sous la courbe de Lorenz. La dernière valeur ne participe pas à l'aire
    gini = 2 * (0.5 - aire_ss_courbe)  # deux fois l'aire entre la 1e bissectrice et la courbe de Lorenz
    return lorenz, gini


class MyGraph:
    """
    with_grid : None, 'both', 'x', 'y'
    """
    def __init__(self, title, nblin=1, nbcol=1, is_mono=False, figsize=(12,8), frameon=True, polar=False):
        sns.set_style("white")
        if (nblin > 1) & (figsize == (12,8)):
            figsize = (12, 6 * nblin)
        self.mult_param = mydefault_plt_parameters(figsize=figsize)
        self.liste_couleurs = set_colors(is_mono)
        self.polar = polar
        self.frameon = frameon
        self.fig = plt.figure(frameon=frameon)
        self.ax = []
        self.ax_desc = []
        self.objets = []
        self.objets_desc = []
        if (nblin == 1) & (nbcol == 1):
            self.__init_graph_simple(title=title, polar=polar)
        else:
            self.__init_graph_multiple(title=title, nblin=nblin, nbcol=nbcol, frameon=frameon, polar=polar)
    
    def __init_graph_simple(self, title, polar):
        self.ax.append(plt.axes(polar=polar))
        self.ax_desc.append("Axe principal - index 1")
        self.fig.suptitle(title, y=1.025)
        self.fig.tight_layout()

    def __init_graph_multiple(self, title, nblin, nbcol, frameon, polar):
        cpt = 0
        self.fig.subplots_adjust(wspace=0.25, hspace=0.25)
        for lin in np.arange(nblin):
            for col in np.arange(nbcol):
                cpt += 1
                self.ax.append(plt.subplot(nblin, nbcol, cpt, frameon=frameon, polar=polar))
                self.ax_desc.append("Axe principal du subplot {} - index {}".format(cpt, cpt))
        self.fig.suptitle(title, y=1.05)
        self.fig.tight_layout()
        
    def __add_grid_twinx(self, multi_index, with_grid, grid_style="-", twinx=False):
        if twinx:
            self.ax[multi_index-1].spines['right'].set_visible(True)
            self.ax.append(self.ax[multi_index-1].twinx())
            self.ax_desc.append("Axe secondaire du subplot {} - index {}".format(multi_index, len(self.ax)))
            multi_index = len(self.ax)
        else:
            if with_grid == 'both':
                self.ax[multi_index-1].grid(axis='both', linestyle=grid_style)
            elif with_grid == 'x':
                self.ax[multi_index-1].grid(axis='x', linestyle=grid_style)
            elif with_grid == 'y':
                self.ax[multi_index-1].grid(axis='y', linestyle=grid_style)
            self.ax[multi_index-1].spines['right'].set_visible(False)
            self.ax[multi_index-1].spines['top'].set_visible(False)
        return multi_index
    
    def __add_legend_subtitle(self, multi_index, legend, subtitle, twinx=False):
        if legend:
            self.ax[multi_index-1].legend()
        if (subtitle is not None) & (not twinx):
            self.ax[multi_index-1].set_title(subtitle, fontweight='regular')
        self.fig.tight_layout()
    
    def add_histogramme(self, values, labels=None, bins=20, density=True, stacked=False, bar_width_percent=None, color=None, alpha=1, subtitle=None, legend=False, with_grid='both', grid_style="-", multi_index=1, twinx=False):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style, twinx)
        self.objets.append(self.ax[multi_index-1].hist(values, bins=bins, label=labels, density=density, stacked=stacked, rwidth=bar_width_percent, color=color, alpha=alpha))
        self.objets_desc.append("Histogramme de l'axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, subtitle, twinx)
    
    def add_bar(self, orientation, with_grid, grid_style, x_values, y_height, label, y_bottom=None, bar_width=0.8, color=None, alpha=1, subtitle=None, legend=False, multi_index=1, twinx=False):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style, twinx)
        self.objets.append(self.ax[multi_index-1].bar(x_values, y_height, bottom=y_bottom, width=bar_width, color=color, orientation=orientation, label=label, alpha=alpha))
        self.objets_desc.append("Barre de l'axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, subtitle, twinx)
            
    def add_barv(self, x_values, y_height, label, y_bottom=None, bar_width=0.8, color=None, alpha=1, subtitle=None, legend=False, with_grid='y', grid_style=":", multi_index=1, twinx=False):
        orientation = 'vertical'
        self.add_bar(orientation, with_grid, grid_style, x_values, y_height, label, y_bottom, bar_width, color, alpha, subtitle, legend, multi_index, twinx)
    
    def add_barh(self, x_values, y_height, label, y_bottom=None, bar_width=0.8, color=None, alpha=1, subtitle=None, legend=False, with_grid='x', grid_style=":", multi_index=1, twinx=False):
        orientation = 'horizontal'
        self.add_bar(orientation, with_grid, grid_style, x_values, y_height, label, y_bottom, bar_width, color, alpha, subtitle, legend, multi_index, twinx)

    def add_plot(self, x_values, y_values, label, marker=None, linestyle="-", color=None, markeredgecolor=None, markerfacecolor=None, alpha=1, linewidth=1.5, subtitle=None, legend=False, with_grid=None, grid_style="-", multi_index=1, twinx=False):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style, twinx)
        self.objets.append(self.ax[multi_index-1].plot(x_values, y_values, label=label, marker=marker, linestyle=linestyle, color=color, markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor, alpha=alpha, linewidth=linewidth))
        self.objets_desc.append("Plot de l'axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, subtitle, twinx)
        
    def add_scatter(self, x_values, y_values, label, marker=None, size=None, color=None, alpha=1, subtitle=None, legend=False, with_grid='both', grid_style="-", multi_index=1, twinx=False):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style, twinx)
        self.objets.append(self.ax[multi_index-1].scatter(x_values, y_values, label=label, marker=marker, s=size, c=color, alpha=alpha))
        self.objets_desc.append("Scatter de l'axe {}".format(multi_index-1))
        self.__add_legend_subtitle(multi_index, legend, subtitle, twinx)
        
    def add_pie(self, values, labels, explodes=None, colors=None, autopct='%1.1f%%', startangle=90, shadow=False, subtitle=None, legend=False, multi_index=1):
        # pie ne peut pas prendre twinx et n'affiche pas de grille -> pas d'appel à __add_grid_twinx
        self.objets.append(self.ax[multi_index-1].pie(values, labels=labels, autopct=autopct, startangle=startangle, explode=explodes, colors=colors, shadow=shadow))
        self.objets_desc.append("Camembert de l'axe {}".format(multi_index-1))
        self.ax[multi_index-1].axis('equal')
        self.__add_legend_subtitle(multi_index, legend, subtitle, False)
    
    def add_boxplot(self, values, cat_labels=None, color=None, color_base_index=None, outliers=True, mult_iq=1.5, means=False, width=0.7, vertical=True, alpha=1, subtitle=None, legend=False, grid_style=":", multi_index=1):
        if vertical:
            with_grid = 'y'
        else:
            with_grid = 'x'
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style)
        bplot = self.ax[multi_index-1].boxplot(values, labels=cat_labels, showfliers=outliers, whis=mult_iq, showmeans=means, widths=width, vert=vertical, patch_artist=True)
        if color is None:
            color = self.liste_couleurs
        nb_color = len(color)
        if cat_labels is None:
            nb_cat_labels = 1
        else:
            nb_cat_labels = len(cat_labels)
        for cpt in np.arange(nb_cat_labels):
            if color_base_index is None:
                colorindex = (cpt + multi_index - 1) % nb_color
            else:
                colorindex = (cpt + color_base_index) % nb_color
            bplot['boxes'][cpt].set_facecolor(color[colorindex])
            bplot['boxes'][cpt].set_alpha(alpha)
            bplot['medians'][cpt].set_color('black')
            if outliers:
                bplot['fliers'][cpt].set_markeredgecolor('grey')
                bplot['fliers'][cpt].set_markerfacecolor(color[colorindex])
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
        
    """
    hue must be a string value (not numeric nor boolean)
    """
    def add_pairplot(self, data, hue, multi_index=1):
        if multi_index == 1:
            #self.ax[0].spines['left'].set_visible(False)
            #self.ax[0].spines['top'].set_visible(False)
            #self.ax[0].spines['right'].set_visible(False)
            #self.ax[0].spines['bottom'].set_visible(False)
            sns.pairplot(data, hue=hue)
            self.fig.tight_layout()
            
            
    def add_regplot(self, x, y, subtitle=None, show_labels=True, multi_index=1):
        sns.regplot(x=x, y=y, ax=self.ax[multi_index-1])
        if subtitle is not None:
            self.ax[multi_index-1].set_title(subtitle, fontweight='regular')
        if show_labels == False:
            self.set_axe('x', label="", multi_index=multi_index)
            self.set_axe('y', label="", multi_index=multi_index)
        self.ax[multi_index-1].spines['right'].set_visible(False)
        self.ax[multi_index-1].spines['top'].set_visible(False)
        
    def graph_concentration(self, lorenz, gini, x_label, y_label, subtitle=None, with_grid='both', grid_style=":", print_gini=True, multi_index=1):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style)
        self.objets.append(self.ax[multi_index-1].plot(np.linspace(0,1,len(lorenz)), label="Première bissectrice"))
        self.objets.append(self.ax[multi_index-1].plot(lorenz, label="Courbe de Lorenz"))
        self.objets_desc.append("Plot de la première bissectrice - axe {}".format(multi_index-1))
        self.objets_desc.append("Plot de la courbe de Lorenz - axe {}".format(multi_index-1))
        self.set_axe(multi_index=multi_index, axe='x', label=x_label, tick_min=-0.025*len(lorenz), tick_max=1.025*len(lorenz))
        self.set_axe(multi_index=multi_index, axe='y', label=y_label, tick_min=-0.025, tick_max=1.025, tick_labels_format=':.0%')
        self.ax[multi_index-1].legend(loc="upper left")
        if print_gini:
            self.add_text(0.16*len(lorenz), 0.8, " Indice de Gini = {:.2f} ".format(gini), multi_index=multi_index)
        if subtitle is not None:
            self.ax[multi_index-1].set_title(subtitle, fontweight='regular')
        self.fig.tight_layout()

    def graph_compar_distrib_normal(self, data, label_data, bins=20, color_distrib=None, alpha_distrib=1, linewidth=1.5, color_loi_normale=None, color_mean=None, show_median=False, color_median=None, subtitle=None, with_grid=None, grid_style=":", multi_index=1):
        multi_index = self.__add_grid_twinx(multi_index, with_grid, grid_style)
        self.objets.append(self.ax[multi_index-1].hist(data, bins=bins, density=True, label=label_data, color=color_distrib, alpha=alpha_distrib))
        self.objets_desc.append("Histogramme des données - axe {}".format(multi_index-1))
        x_theo = np.arange(data.min(), data.max(), 0.01*(data.max()-data.min()))
        data_mean = data.mean()
        data_std = data.std(ddof=1)
        self.objets.append(self.ax[multi_index-1].plot(x_theo, st.norm.pdf(x_theo, loc=data_mean, scale=data_std), color=color_loi_normale, linewidth=linewidth, label="Distribution de la loi normale"))
        self.objets_desc.append("Distribution théorique de la loi normale - axe {}".format(multi_index-1))
        self.add_line(coord=data_mean, color=color_mean, linewidth=linewidth, label="Moyenne des données", multi_index=multi_index)
        if show_median:
            data_median = data.median()
            self.add_line(coord=data_median, color=color_median, linewidth=linewidth, label="Médiane des données", multi_index=multi_index)
        self.ax[multi_index-1].legend(loc='upper right')
        if subtitle is not None:
            self.ax[multi_index-1].set_title(subtitle, fontweight='regular')
        self.fig.tight_layout()
        
    def graph_compar_distrib_binomial(self, freq_empirique, n, p, label_data, color_distrib=None, alpha_distrib=1, linewidth=1.5, color_loi_binomiale=None, color_mean=None, subtitle=None, with_grid=None, grid_style=":", multi_index=1):
        self.add_barv(range(n+1), freq_empirique, label=label_data, color=color_distrib, alpha=alpha_distrib, subtitle=subtitle, with_grid=with_grid, grid_style=grid_style, multi_index=multi_index)
        x_theo = range(n+1)
        freq_theo = st.binom.pmf(x_theo, n, p) * freq_empirique.sum()
        data_mean = 0.0
        for i in x_theo:
            data_mean += (freq_empirique[i] * i)
        data_mean = data_mean / freq_empirique.sum()
        self.objets.append(self.ax[multi_index-1].plot(x_theo, freq_theo, color=color_loi_binomiale, linewidth=linewidth, label="Distribution de la loi binomiale"))
        self.objets_desc.append("Distribution théorique de la loi binomiale - axe {}".format(multi_index-1))
        self.add_line(coord=data_mean, color=color_mean, linewidth=linewidth, label="Moyenne des données", multi_index=multi_index)
        self.ax[multi_index-1].legend(loc='upper right')
        
    """
    si multi_index == 0, on ajoute du text à fig ; sinon le texte concerne l'axe d'indice multi_index-1
    """
    def add_text(self, x_coord, y_coord, text, horizontalalignment='center', verticalalignment='top', color='black', backgroundcolor=(0.95, 0.95, 0.95), multi_index=1):
        if multi_index == 0:
            self.objets.append(self.fig.text(x_coord, y_coord, text, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, color=color, backgroundcolor=backgroundcolor))
            self.objets_desc.append("Texte - figure")
        else:
            self.objets.append(self.ax[multi_index-1].text(x_coord, y_coord, text, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, color=color, backgroundcolor=backgroundcolor))
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
        
    def set_axe(self, axe, label=None, label_position=(0.5,0.5), tick_min=None, tick_max=None, tick_step=None, tick_labels=None, tick_labels_format=None, tick_dash=False, color=None, rotation=None, ha=None, va=None, multi_index=1, twinx=False):
        """
        axe : 'x' ou 'y'
        label_format : par exemple : ':.0%' ; ':.2%' ; ':,.0f', ':.3f'
        """
        if axe == 'x':
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
        if axe == 'y':
            if label is not None:
                self.ax[multi_index - 1].set_ylabel(label, position=label_position)
            if tick_min is None:
                tick_min = self.ax[multi_index - 1].get_ylim()[0]
            if tick_max is None:
                tick_max = self.ax[multi_index - 1].get_ylim()[1]
            if tick_step is not None:
                self.ax[multi_index - 1].yaxis.set_ticks(np.arange(tick_min, tick_max + (tick_step/10), tick_step))
                self.ax[multi_index - 1].set_yticklabels(np.arange(tick_min, tick_max + (tick_step/10), tick_step))
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

    """
    si multi_index == 0, on ajoute une légende à fig ; sinon la légende concerne l'axe d'indice multi_index-1
    """
    def set_legend(self, loc=None, bbox_to_anchor=None, ncol=1, title=None, multi_index=1):
        if multi_index == 0:
            self.fig.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol, title=title)
        else:
            self.ax[multi_index - 1].legend(loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol, title=title)
