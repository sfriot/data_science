# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 2019
@author: Sylvain Friot
Content: inferential statistics : estimators, confidence intervals and tests
"""
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.stats as sm
import statsmodels.stats.api as sma
import logging as lg
import sys


"""
Class GenericFunctions
Parent class with generic functions used by the other classes
Includes some functions to manage input errors and to write comments on the results of estimates and tests
"""
class GenericFunctions:
    
    def checktype_ndarray(self, x, name):
        if not isinstance(x, (list,pd.Series,np.ndarray)):
            raise TypeError("Type Error : input data {} must be a list, an numpy ndarray or pandas Series (not DataFrame).".format(name))
    
    def checknumber_equal_value(self, number, value, name, notequal=False):
        if notequal:
            if number == value:
                raise ValueError("Value Error : {} must be != {}".format(name, value))
        else:
            if number != value:
                raise ValueError("Value Error : {} must be = {}".format(name, value))
                
    def checknumber_sup_value(self, number, value, name, orequal=True):
        if orequal:
            if number < value:
                raise ValueError("Value Error : {} must be >= {}".format(name, value))
        else:
            if number <= value:
                raise ValueError("Value Error : {} must be > {}".format(name, value))
        
    def checknumber_n1_equal_n2(self, number1, name1, number2, name2, notequal=False):
        if notequal:
            if number1 == number2:
                raise ValueError("Value Error : {} must be different than {}".format(name1, name2))
        else:
            if number1 != number2:
                raise ValueError("Value Error : {} must be equal to {}".format(name1, name2))
        
    def generic_shortcomment(self, estim_object, tested_value, p_value, alpha, alternative):
        # remarque : les tests pour toujours sur du féminin : proportion, moyenne, variance et différence.
        # nous n'avons que ratio comme mot masculin. Et il est toujours au singulier. Donc les -> féminin pluriel
        if estim_object[0:2] == "la":
            verbe = "est"
            accord = "e"
        elif estim_object[0:3] == "les":
            verbe = "sont"
            accord = "es"
        else:
            verbe = "est"
            accord = ""
        if p_value > alpha:
            comment = "Le test conclut que {} {} égal{} à {}.".format(estim_object, verbe, accord, tested_value)
        else:
            if alternative == "two-sided":
                comment = "Le test conclut que {} {} différent{} de {}.".format(estim_object, verbe, accord, tested_value)
            elif alternative == "smaller":
                comment = "Le test conclut que {} {} inférieur{} à {}.".format(estim_object, verbe, accord, tested_value)
            else:
                comment = "Le test conclut que {} {} supérieur{} à {}.".format(estim_object, verbe, accord, tested_value)
        return comment
    
    def generic_longcomment(self, test_object, stat_value, stat_law, estim_object, estim_value, cint, tested_value, p_value, alpha, alternative, precision_txt=None):
        confidence_level = 1 - alpha
        if estim_object[0:2] == "la":
            verbe = "est"
            accord = "e"
        elif estim_object[0:3] == "les":
            verbe = "sont"
            accord = "es"
        else:
            verbe = "est"
            accord = ""
        comment = "On teste {} = {} avec la {}.\n".format(test_object, stat_value, stat_law)
        if precision_txt is not None:
            if precision_txt[0:2] == "la":
                precision_verbe = "est"
                precision_accord = "e"
            elif precision_txt[0:3] == "les":
                precision_verbe = "sont"
                precision_accord = "es"
            else:
                precision_verbe = "est"
                precision_accord = ""
            comment += "{} {} estimé{} à {} et {} compris{} dans l'intervalle {} avec un niveau de confiance de {:.2%}.\n".format(precision_txt.capitalize(), precision_verbe, precision_accord, estim_value, precision_verbe, precision_accord, cint, confidence_level)
        else:
            comment += "{} {} estimé{} à {} et {} compris{} dans l'intervalle {} avec un niveau de confiance de {:.2%}.\n".format(estim_object.capitalize(), verbe, accord, estim_value, verbe, accord, cint, confidence_level)
        if p_value > alpha:
            comment += "La p_value {:.3f} est supérieure à alpha {:.3f}.".format(p_value, alpha)
            comment += " L'hypothèse H0 que {} {} égal{} à {} est acceptée au niveau de test de {:.2%}".format(estim_object, verbe, accord, tested_value, alpha)
            if alternative == "two-sided":
                comment += " (contre l'hypothèse alternative que {} {} différent{} de {}).".format(estim_object, verbe, accord, tested_value)
            elif alternative == "smaller":
                comment += " (contre l'hypothèse alternative que {} {} inférieur{} à {}).".format(estim_object, verbe, accord, tested_value)
            else:
                comment += " (contre l'hypothèse alternative que {} {} supérieur{} à {}).".format(estim_object, verbe, accord, tested_value)
        else:
            comment += "La p_value {:.3f} est inférieure à alpha {:.3f}.".format(p_value, alpha)
            comment += " L'hypothèse H0 que {} {} égal{} à {} est rejetée avec un niveau de risque de {:.2%}.".format(estim_object, verbe, accord, tested_value, alpha)
            if alternative == "two-sided":
                comment += " Le test conclut que {} {} différent{} de {}.".format(estim_object, verbe, accord, tested_value)
            elif alternative == "smaller":
                comment += " Le test conclut que {} {} inférieur{} à {}.".format(estim_object, verbe, accord, tested_value)
            else:
                comment += " Le test conclut que {} {} supérieur{} à {}.".format(estim_object, verbe, accord, tested_value)
        return comment
    
    def testlaw_comment(self, tested_law, stat_name, stat_value, p_value, alpha, short_comment, complement_loi=None):
        if short_comment:
            if p_value > alpha:
                comment = "Le test conclut que la distribution de l'échantillon suit une loi {}.".format(tested_law)
            else:
                comment = "Le test conclut que la distribution de l'échantillon ne suit pas une loi {}.".format(tested_law)
        else:
            if complement_loi is None:
                comment = "La statistique testée est {} = {:.3f}\n".format(stat_name, stat_value)
            else:
                comment = "La statistique testée est {} = {:.3f} avec la {}\n".format(stat_name, stat_value, complement_loi)
            if p_value > alpha:
                comment += "La p_value {:.3f} est supérieure à alpha {:.3f}.".format(p_value, alpha)
                comment += " L'hypothèse que la distribution de l'échantillon suit une loi {} est acceptée au niveau de test de {:.2%}.".format(tested_law, alpha)
            else:
                comment += "La p_value {:.3f} est inférieure à alpha {:.3f}.".format(p_value, alpha)
                comment += " L'hypothèse que la distribution de l'échantillon suit une loi {} est rejetée avec un niveau de risque de {:.2%}.".format(tested_law, alpha)
                comment += " Le test conclut que la distribution de l'échantillon ne suit pas une loi {}.".format(tested_law)
        return comment
    
    def test2laws_comment(self, stat_value, p_value, alpha, short_comment):
        if short_comment:
            if p_value > alpha:
                comment = "Le test conclut que les deux échantillons ont une distribution similaire et que leurs médianes peuvent être considérées comme identiques."
            else:
                comment = "Le test conclut que les deux échantillons n'ont pas une distribution similaire et que leurs médianes peuvent être considérées comme différentes."
        else:
            if self.paired:
                comment = "La statistique testée est W = {:.3f}\n".format(stat_value)
            else:
                comment = "La statistique testée est U = {:.3f}\n".format(stat_value)
            if p_value > alpha:
                comment += "La p_value {:.3f} est supérieure à alpha {:.3f}.".format(p_value, alpha)
                comment += " L'hypothèse que les deux échantillons ont une distribution similaire est acceptée au niveau de test de {:.2%}.".format(alpha)
                comment += " Nous pouvons considérer que leurs médianes sont identiques."
            else:
                comment += "La p_value {:.3f} est inférieure à alpha {:.3f}.".format(p_value, alpha)
                comment += " L'hypothèse que les deux échantillons ont une distribution similaire est rejetée avec un niveau de risque de {:.2%}.".format(alpha)
                comment += " Nous pouvons considérer que leurs médianes sont différentes."
        return comment

"""
Class : Proportion
Calculates estimate, confidence_interval and test on one proportion
Attributes
    n_success: number of successes
    n_trials: number of trials
    estimate : estimation of the proportion
Methods
    confidence_interval : calculates the centered confidence interval of the proportion
    test_value = tests if the proportion is equal to a target value
    comment_estimate
    comment_confidence_interval
    comment_test_value
"""
class Proportion(GenericFunctions):
    
    def __init__(self, n_successes, n_trials):
        GenericFunctions.__init__(self)
        self.n_successes = int(n_successes)
        self.n_trials = int(n_trials)
        self.estimate = n_successes / n_trials
    
    """
    Intervalle de confiance symétrique pour la proportion
    Paramètres
        confidence_level: niveau de confiance pour l'intervalle de confiance
        method_forced = détermine la loi à utiliser pour l'intervalle de confiance :
            - None : par défaut en fonction de la taille de l'échantillon : test exact avec la loi binomiale si n_trials < 30 ; test avec la loi normale sinon
            - 'normal' : loi normale quelque soit la taille de l'échantillon
            - 'binomial' ou 'binom_test' : loi binomiale quelque soit la taille de l'échantillon
            - peut aussi être 'agresti_coull','beta','wilson','jeffreys' (cf doc statsmodels)
    """
    def confidence_interval(self, confidence_level=0.95, method_forced=None):
        alpha = 1 - confidence_level
        if method_forced == 'binomial':
                method_forced = 'binom_test'
        if method_forced is None:
            if self.n_trials < 30:
                method_forced = 'binom_test'
            else:
                method_forced = 'normal'
        confidence_interval = sm.proportion.proportion_confint(self.n_successes, self.n_trials, alpha=alpha, method=method_forced)
        return confidence_interval
    
    """
    Teste si la proportion est égale à la valeur tested_p
    Paramètres
        tested_p: valeur de p testée ( H0 : p == tested_p )
        alpha: niveau de risque de première espèce du test (rejet de H0 à tort)
        alternative = détermine l'hypothèse alternative. 3 alternatives possibles :
            - 'two-sided' : H1 : p != tested_p
            - 'smaller' : H1 : p < tested_p
            - 'larger' : H1 : p > tested_p
        method_forced = détermine la loi à utiliser pour le test :
            - None : par défaut en fonction de la taille de l'échantillon : test exact avec la loi binomiale si n_trials < 30 ; test avec la loi normale sinon
            - 'normal' : loi normale quelque soit la taille de l'échantillon
            - 'binomial' : loi binomiale quelque soit la taille de l'échantillon
    Valeurs retournées
        zstat : statistique évakuée par le test (n'est calculée qu'avec le test par la loi normale)
        p_value : p_value du test
        confidence_interval : intervalle de confiance pour p (symétrique ou asymétrique selon H1)
    """
    def test_value(self, tested_p, alpha=0.05, alternative="two-sided", method_forced=None):
        confidence_level = 1 - alpha
        verif_appro_normale = self.n_trials * self.estimate * (1 - self.estimate)
        if verif_appro_normale < 5:
            if (method_forced is None) | (method_forced == 'binomial'):
                zstat, p_value, confidence_interval = self.test_binomial(tested_p, alpha, alternative)
            else:
                zstat, p_value, confidence_interval = self.test_normal(tested_p, confidence_level, alternative)
        else:
            if method_forced == 'binomial':
                zstat, p_value, confidence_interval = self.test_binomial(tested_p, alpha, alternative)
            else:
                zstat, p_value, confidence_interval = self.test_normal(tested_p, confidence_level, alternative)
        return zstat, p_value, confidence_interval
        
    # comment on estimate
    def comment_estimate(self):
        comment = "La taille de l'échantillon est {}.\n".format(self.n_trials)
        comment += "Le nombre de succès est {}.\n".format(self.n_successes)
        comment += "L'estimateur de la proportion de succès est p = {}.".format(self.estimate)
        return comment
    
    # comment on confidence_interval
    def comment_confidence_interval(self, confidence_level=0.95, method_forced=None):
        cint = self.confidence_interval(confidence_level=confidence_level, method_forced=method_forced)
        comment = "L'intervalle de confiance de la proportion de succès p est {} au niveau de confiance {:.2%}.".format(cint, confidence_level)
        return comment
    
    """
    comment on test_value
    Parameter short_comment
        True : one-sentence summary of the test result
        False (default) : details on the test and its results
    """
    def comment_test_value(self, tested_p, alpha=0.05, alternative="two-sided", method_forced=None, short_comment=False):
        zstat, p_value, cint = self.test_value(tested_p, alpha=alpha, alternative=alternative, method_forced=method_forced)
        if short_comment:
            comment = self.generic_shortcomment(estim_object="la proportion de succès p", tested_value=tested_p, p_value=p_value, alpha=alpha, alternative=alternative)
        else:
            verif_appro_normale = self.n_trials * self.estimate * (1 - self.estimate)
            if verif_appro_normale < 5:
                if (method_forced is None) | (method_forced == 'binomial'):
                    comment = "Test binomial exact de la proportion de succès d'un échantillon de taille {}.\n".format(self.n_trials)
                    comment += self.generic_longcomment(test_object="la proportion p", stat_value=tested_p, stat_law="loi binomiale p={} et n={}".format(self.estimate, self.n_trials),
                              estim_object="la proportion de succès p", estim_value=self.estimate, cint=cint, tested_value=tested_p, p_value=p_value, alpha=alpha, alternative=alternative)
                else:
                    comment = "Test basé sur la loi normale de la proportion de succès d'un échantillon de taille {}. ATTENTION, pour ce cas où n*p*(1-p) < 5, il est conseillé d'utiliser le test exact binomial.\n".format(self.n_trials)
                    comment += self.generic_longcomment(test_object="la statistique z", stat_value=zstat, stat_law="loi normale centrée réduite",
                              estim_object="la proportion de succès p", estim_value=self.estimate, cint=cint, tested_value=tested_p, p_value=p_value, alpha=alpha, alternative=alternative)
            else:
                if method_forced == 'binomial':
                    comment = "Test binomial exact de la proportion de succès d'un échantillon de taille {}. ATTENTION, pour ce cas où n*p*(1-p) >= 5, il est d'usage d'utiliser le test asymptotique avec la loi normale.\n".format(self.n_trials)
                    comment += self.generic_longcomment(test_object="la proportion p", stat_value=tested_p, stat_law="loi binomiale p={} et n={}".format(self.estimate, self.n_trials),
                              estim_object="la proportion de succès p", estim_value=self.estimate, cint=cint, tested_value=tested_p, p_value=p_value, alpha=alpha, alternative=alternative)
                else:
                    comment = "Test basé sur la loi normale de la proportion de succès d'un échantillon de taille {}.\n".format(self.n_trials)
                    comment += self.generic_longcomment(test_object="la statistique z", stat_value=zstat, stat_law="loi normale centrée réduite",
                              estim_object="la proportion de succès p", estim_value=self.estimate, cint=cint, tested_value=tested_p, p_value=p_value, alpha=alpha, alternative=alternative)
        return comment
    
    # test exact de la proportion avec la loi binomiale
    def test_binomial(self, tested_p, alpha, alternative):
        zstat = np.nan
        p_value = sm.proportion.binom_test(self.n_successes, self.n_trials, prop=tested_p, alternative=alternative)
        if alternative == "two-sided":
            if self.n_successes == 0:
                cint_low = 0
                cint_high = st.beta.ppf(1 - (alpha / 2), self.n_successes + 1, self.n_trials - self.n_successes)
            elif self.n_successes == self.n_trials:
                cint_low = st.beta.ppf((alpha / 2), self.n_successes, self.n_trials - self.n_successes + 1)
                cint_high = 1
            else:
                cint_low = st.beta.ppf((alpha / 2), self.n_successes, self.n_trials - self.n_successes + 1)
                cint_high = st.beta.ppf(1 - (alpha / 2), self.n_successes + 1, self.n_trials - self.n_successes)
        elif alternative == "smaller":
            cint_low = 0
            if self.n_successes == self.n_trials:
                cint_high = 1
            else:
                cint_high = st.beta.ppf(1 - alpha, self.n_successes + 1, self.n_trials - self.n_successes)
        else:
            cint_high = 1
            if self.n_successes == 0:
                cint_low = 0
            else:
                cint_low = st.beta.ppf(alpha, self.n_successes, self.n_trials - self.n_successes + 1)
        p_confidence_interval = (cint_low * 100 , cint_high * 100)
        return zstat, p_value, p_confidence_interval
    
    # test de la proportion avec la loi normale
    def test_normal(self, tested_p, confidence_level, alternative):
        zstat, p_value = sm.proportion.proportions_ztest(self.n_successes, self.n_trials, value=tested_p, alternative=alternative)
        if alternative == "two-sided":
            z = st.norm.ppf((1 + confidence_level) / 2)
        else:
            z = st.norm.ppf(confidence_level)
        z22n = z**2 / (2 * self.n_trials)
        inter_cint = z * np.sqrt((self.estimate * (1 - self.estimate) / self.n_trials) + (z22n / (2 * self.n_trials)))
        cint_low = (self.estimate + z22n - inter_cint) / (1 + (2 * z22n))
        cint_high = (self.estimate + z22n + inter_cint) / (1 + (2 * z22n))
        if alternative == "two-sided":
            p_confidence_interval = (max(cint_low, 0), min(cint_high, 1))
        elif alternative == "smaller":
            p_confidence_interval = (0, min(cint_high, 1))
        else:
            p_confidence_interval = (max(cint_low, 0), 1)
        return zstat, p_value, p_confidence_interval
    
    
"""
Class : Mean
Calculates estimate, confidence_interval and test on the mean of a sample
Attributes
    data : one-dimension numpy array containing data to estimate or test (sample)
    size : number of data in the sample
    estimate : estimation of the mean of the sample
Methods
    confidence_interval : calculates the centered confidence interval of the mean
    test_value = tests if the mean is equal to a target value
    comment_estimate
    comment_confidence_interval
    comment_test_value
ATTENTION
    l'échantillon doit être iid dans tous les cas.
    si n<30, il faut vérifier que l'échantillon est gaussien (suit une loi normale)
    si n>=30, cette vérification n'est pas nécessaire
"""
class Mean(GenericFunctions):
    
    def __init__(self, x):
        GenericFunctions.__init__(self)
        try:
            self.checktype_ndarray(x, "x")
            x = np.array(x)
            self.checknumber_equal_value(x.ndim, 1, "the dimension of the array")
            x = x[~np.isnan(x)]
            self.checknumber_sup_value(len(x), 2, "the number of non-null data in the array", orequal=True)
        except (TypeError, ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error Mean init : {}".format(sys.exc_info()))
        else:
            self.data = x
            self.size = len(x)
            self.estimate = x.mean()
            
    """
    Intervalle de confiance symétrique pour la moyenne
    Paramètre
        confidence_level: niveau de confiance pour l'intervalle de confiance
    """
    def confidence_interval(self, confidence_level=0.95):
        if self.size < 30:
            lg.warning("Attention, l'échantillon ne contient que {} valeurs. Lorsque la taille de l'échantillon est inférieure à 30, il faut vérifier que sa distribution suit une loi normale.".format(self.size))
        moyenne, variance, ecart_type = st.bayes_mvs(self.data, alpha=confidence_level)
        confidence_interval = moyenne.minmax
        return confidence_interval
    
    """
    t-test pour déterminer si la moyenne de l'échantillon est égale à la valeur tested_mu        
    Paramètres
        tested_mu: valeur de mu testée ( H0 : mu == tested_mu )
        alpha: niveau de risque de première espèce du test (rejet de H0 à tort)
        alternative = détermine l'hypothèse alternative. 3 alternatives possibles :
            - 'two-sided' : H1 : mu != tested_mu
            - 'smaller' : H1 : mu < tested_mu
            - 'larger' : H1 : mu > tested_mu    
    Valeurs retournées
        tstat : statistique t de Student testée
        dof : degrés de liberté de la loi de Student
        p_value : p_value du test
        confidence_interval : intervalle de confiance pour mu (symétrique ou asymétrique selon H1)
    """
    def test_value(self, tested_mu, alpha=0.05, alternative="two-sided"):
        confidence_level = 1 - alpha
        vx = self.data.var(ddof=1)
        dof = self.size - 1
        stderr = np.sqrt(vx / self.size)
        tstat = (self.estimate - tested_mu) / stderr        
        if alternative == "smaller":
            p_value = st.t.cdf(tstat, dof)
            pre_confidence_interval = (-np.inf, tstat + st.t.ppf(confidence_level, dof))
        elif alternative == "larger":
            p_value = 1 - st.t.cdf(tstat, dof)
            pre_confidence_interval = (tstat - st.t.ppf(confidence_level, dof), np.inf)
        else:
            p_value = 2 * st.t.cdf(-np.abs(tstat), dof)
            tempinterval = st.t.ppf(1 - (alpha / 2), dof)
            pre_confidence_interval = (tstat - tempinterval, tstat + tempinterval)
        confidence_interval = (tested_mu + pre_confidence_interval[0] * stderr, tested_mu + pre_confidence_interval[1] * stderr)
        return tstat, dof, p_value, confidence_interval
        
    # comment on estimate
    def comment_estimate(self):
        comment = "L'estimateur de la moyenne est mu = {}.".format(self.estimate)
        return comment
    
    # comment on confidence_interval
    def comment_confidence_interval(self, confidence_level=0.95):
        cint = self.confidence_interval(confidence_level=0.95)
        comment = "L'intervalle de confiance de la moyenne mu est {} au niveau de confiance {:.2%}.".format(cint, confidence_level)
        return comment
    
    """
    comment on test_value
    Parameter short_comment
        True : one-sentence summary of the test result
        False (default) : details on the test and its results
    """
    def comment_test_value(self, tested_mu, alpha=0.05, alternative="two-sided", short_comment=False):
        tstat, dof, p_value, cint = self.test_value(tested_mu, alpha=alpha, alternative=alternative)
        if short_comment:
            comment = self.generic_shortcomment(estim_object="la moyenne mu", tested_value=tested_mu, p_value=p_value, alpha=alpha, alternative=alternative)
        else:
            comment = "T-test de la valeur de la moyenne mu d'un échantillon de taille {}.\n".format(self.size)
            if self.size < 30:
                comment += "ATTENTION, l'échantillon ne contient que {} valeurs. Lorsque la taille de l'échantillon est inférieure à 30, il faut vérifier que sa distribution suit une loi normale.\n".format(self.size)
            comment += self.generic_longcomment(test_object="la statistique T", stat_value=tstat, stat_law="loi t de Student à {} degrés de liberté".format(dof),
                          estim_object="la moyenne mu", estim_value=self.estimate, cint=cint, tested_value=tested_mu, p_value=p_value, alpha=alpha, alternative=alternative)
        return comment


"""
Class : Variance
Calculates estimate, confidence_interval and test on the variance of a sample
Attributes
    data : one-dimension numpy array containing data to estimate or test (sample)
    size : number of data in the sample
    estimate : estimation of the variance of the sample
Methods
    confidence_interval : calculates the centered confidence interval of the mean
    test_value = tests if the mean is equal to a target value
    comment_estimate
    comment_confidence_interval
    comment_test_value
ATTENTION
    l'échantillon doit être iid dans tous les cas.
    si n<30, il faut vérifier que l'échantillon est gaussien (suit une loi normale)
    si n>=30, cette vérification n'est pas nécessaire
"""
class Variance(GenericFunctions):
    
    def __init__(self, x):
        GenericFunctions.__init__(self)
        try:
            self.checktype_ndarray(x, "x")
            x = np.array(x)
            self.checknumber_equal_value(x.ndim, 1, "the dimension of the array")
            x = x[~np.isnan(x)]
            self.checknumber_sup_value(len(x), 2, "the number of non-null data in the array", orequal=True)
        except (TypeError, ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error Variance init: {}".format(sys.exc_info()))
        else:
            self.data = x
            self.size = len(x)
            self.estimate = x.var(ddof=1) # les données sont converties en numpy array -> on doit préciser ddof=1 pour avoir l'estimateur non-biaisé
            
    """
    Intervalle de confiance symétrique pour la variance
    Paramètre
        confidence_level: niveau de confiance pour l'intervalle de confiance
    """
    def confidence_interval(self, confidence_level=0.95):
        if self.size < 30:
            lg.warning("Attention, l'échantillon ne contient que {} valeurs. Lorsque la taille de l'échantillon est inférieure à 30, il faut vérifier que sa distribution suit une loi normale.".format(self.size))
        moyenne, variance, ecart_type = st.bayes_mvs(self.data, alpha=confidence_level)
        confidence_interval = variance.minmax
        return confidence_interval
    
    """
    Test du chi2 pour déterminer si la variance de l'échantillon est égale à la valeur tested_var
    Paramètres
        tested_var: valeur de s'2 testée ( H0 : s'2 == tested_var )
        alpha: niveau de risque de première espèce du test (rejet de H0 à tort)
        alternative = détermine l'hypothèse alternative. 3 alternatives possibles :
            - 'two-sided' : H1 : s'2 != tested_var
            - 'smaller' : H1 : s'2 < tested_var
            - 'larger' : H1 : s'2 > tested_var    
    Valeurs retournées
        chi2stat : statistique chi2 testée
        dof : degrés de liberté de la loi du chi2
        p_value : p_value du test
        confidence_interval : intervalle de confiance pour s'2 (symétrique ou asymétrique selon H1)
    """
    def test_value(self, tested_var, alpha=0.05, alternative="two-sided"):
        dof = self.size - 1
        chi2stat = dof * self.estimate / tested_var
        smaller_value = st.chi2.cdf(chi2stat, df = dof)
        larger_value = 1 - st.chi2.cdf(chi2stat, df = dof)
        if alternative == "two-sided":
            p_value = 2 * min(smaller_value, larger_value)
            confidence_interval = (dof * self.estimate / st.chi2.ppf(1-(alpha/2), df = dof), 
                                   dof * self.estimate / st.chi2.ppf(alpha/2, df = dof))
        elif alternative == "smaller":
            p_value = smaller_value
            confidence_interval = (0, dof * self.estimate / st.chi2.ppf(alpha, df = dof))
        else:
            p_value = larger_value
            confidence_interval = (dof * self.estimate / st.chi2.ppf(1-alpha, df = dof), np.inf)
        return chi2stat, dof, p_value, confidence_interval
            
    # comment on estimate
    def comment_estimate(self):
        comment = "L'estimateur de la variance est s'2 = {}.".format(self.estimate)
        return comment
    
    # comment on confidence_interval
    def comment_confidence_interval(self, confidence_level=0.95):
        cint = self.confidence_interval(confidence_level=0.95)
        comment = "L'intervalle de confiance de la variance s'2 est {} au niveau de confiance {:.2%}.".format(cint, confidence_level)
        return comment
    
    """
    comment on test_value
    Parameter short_comment
        True : one-sentence summary of the test result
        False (default) : details on the test and its results
    """
    def comment_test_value(self, tested_var, alpha=0.05, alternative="two-sided", short_comment=False):
        chi2stat, dof, p_value, cint = self.test_value(tested_var, alpha=alpha, alternative=alternative)
        if short_comment:
            comment = self.generic_shortcomment(estim_object="la variance s'2", tested_value=tested_var, p_value=p_value, alpha=alpha, alternative=alternative)
        else:
            comment = "Test du chi2 de la valeur de la variance s'2 d'un échantillon de taille {}.\n".format(self.size)
            if self.size < 30:
                comment += "ATTENTION, l'échantillon ne contient que {} valeurs. Lorsque la taille de l'échantillon est inférieure à 30, il faut vérifier que sa distribution suit une loi normale.\n".format(self.size)
            comment += self.generic_longcomment(test_object="la statistique chi2", stat_value=chi2stat, stat_law="loi du chi2 à {} degrés de liberté".format(dof),
                          estim_object="la variance s'2", estim_value=self.estimate, cint=cint, tested_value=tested_var, p_value=p_value, alpha=alpha, alternative=alternative)
        return comment


"""
Class : OneSample
One sample of continuous numeric data : 
    Accesses to Mean and Variance object, to estimate or test theoritical values. 
    Runs tests to check the distribution of data
Attributes
    data : one-dimension numpy array containing data of the sample
    size : number of data in the sample
    mean : Mean object to estimate or test the mean of the sample
    var : Variance object to estimate or test the variance of the sample
Methods
    test_loi_continue_ks : performs the Kolmogorov-Smirnov test for goodness of fit (any continuous distributions included in scipy.stats)
    test_loi_normale_shapiro : performs the Shapiro-Wilk test for normality.
    test_normalite : performs Kolmogorov-Smirnov or Shapiro-Wilk test for normality, according to the sample size
    comment_test_loi_continue_ks
    comment_test_loi_normale_shapiro
    comment_test_normalite
"""
class OneSample(GenericFunctions):
    
    def __init__(self, x):
        GenericFunctions.__init__(self)
        try:
            self.checktype_ndarray(x, "One Sample x")
            x = np.array(x)
            self.checknumber_equal_value(x.ndim, 1, "the dimension of the array")
            x = x[~np.isnan(x)]
            self.checknumber_sup_value(len(x), 3, "the number of non-null data in the array", orequal=True)
        except (TypeError, ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error OneSample init : {}".format(sys.exc_info()))
        else:
            self.data = x
            self.size = len(x)
            self.mean = Mean(x)
            self.var = Variance(x)
    
    """
    CHANGES TO DO : ADD ONE OPTIONAL OTHER PARAMETER (dof, alpha, etc) - ADD OPTIONAL FORCED MEAN AND OPTIONAL FORCED VAR
    Test de Kolmogorov Smirnov pour vérifier si la distribution de l'échantillon suit une loi continue définie
    Paramètres
        tested_law: loi continue testée. Nom des lois continues dispo avec scipy.stats
    Valeurs retournées
        dstat = valeur de la statistique D testée
        p_value = p-value du test
    """
    def test_loi_continue_ks(self, tested_law='norm', other_parameter=None, forced_mean=None, forced_var=None):
        if forced_mean is None:
            forced_mean = self.mean.estimate
        if forced_var is None:
            forced_var = self.var.estimate
        if other_parameter is None:
            dstat, p_value = st.kstest(self.data, tested_law, args=(forced_mean, np.sqrt(forced_var)))
        else:
            dstat, p_value = st.kstest(self.data, tested_law, args=(other_parameter, forced_mean, np.sqrt(forced_var)))
        return dstat, p_value

    """
    Test de Shapiro-Wilk pour vérifier si la distribution de l'échantillon suit une loi normale
    Valeurs retournées
        wstat = valeur de la statistique W testée
        p_value = p-value du test
    """
    def test_loi_normale_shapiro(self):
        wstat, p_value = st.shapiro(self.data)
        return wstat, p_value

    """
    Test de normalité
        si n<5000, on utilise par défaut le test de Shapiro-Wilk
        si n>=5000, on utilise par défaut le test de Kolmogorov-Smirnov
    Valeurs retournées
        test_stat = valeur de la statistique D ou W testée
        p_value = p-value du test
    """    
    def test_normalite(self):
        if self.size < 5000:
            test_stat, p_value = self.test_loi_normale_shapiro()
        else:
            test_stat, p_value = self.test_loi_continue_ks(tested_law='norm')
        return test_stat, p_value
    
    """
    Paramètres
        tested_law: loi continue testée. Nom des lois continues dispo avec scipy.stats
        alpha: niveau de risque de première espèce du test (rejet de H0 à tort)
        short_comment : conclusion du test en une ligne si True, résultats détaillés si False
    """
    def comment_test_loi_continue_ks(self, tested_law='norm', alpha=0.05, short_comment=False):
        dstat, p_value = self.test_loi_continue_ks(tested_law=tested_law)
        comment = self.testlaw_comment(tested_law, "D", dstat, p_value, alpha, short_comment)
        if not short_comment:
            if (tested_law == 'norm') & (self.size < 5000):
                comment = "Test de Kolmogorov-Smirnov de la loi de distribution {} pour l'échantillon de taille {}.\nATTENTION, pour tester la normalité d'un échantillon de moins de 5000 individus, il est conseillé d'utiliser le test de Shapiro-Wilk.\n".format(tested_law, self.size) + comment
            else:
                comment = "Test de Kolmogorov-Smirnov de la loi de distribution {} pour l'échantillon de taille {}.\n".format(tested_law, self.size) + comment
        return comment
    
    """
    Paramètres
        alpha: niveau de risque de première espèce du test (rejet de H0 à tort)
        short_comment : conclusion du test en une ligne si True, résultats détaillés si False
    """
    def comment_test_loi_normale_shapiro(self, alpha=0.05, short_comment=False):
        wstat, p_value = self.test_loi_normale_shapiro()
        comment = self.testlaw_comment("normale", "W", wstat, p_value, alpha, short_comment)
        if not short_comment:
            if self.size < 5000:
                comment = "Test de normalité de Shapiro-Wilk pour l'échantillon de taille {}.\n".format(self.size) + comment
            else:
                comment = "Test de normalité de Shapiro-Wilk pour l'échantillon de taille {}.\nATTENTION, pour tester la normalité d'un échantillon de plus de 5000 individus, il est conseillé d'utiliser le test de Kolmogorov-Smirnov.\n".format(self.size) + comment
        return comment
    
    """
    Paramètres
        alpha: niveau de risque de première espèce du test (rejet de H0 à tort)
        short_comment : conclusion du test en une ligne si True, résultats détaillés si False
    """
    def comment_test_normalite(self, alpha=0.05, short_comment=False):
        if self.size < 5000:
            comment = self.comment_test_loi_normale_shapiro(alpha=alpha, short_comment=short_comment)
        else:
            comment = self.comment_test_loi_continue_ks(tested_law='norm', alpha=alpha, short_comment=short_comment)
        return comment
        

"""
Class : TwoSamples
One sample of continuous numeric data : 
    Accesses to OneSample object, and all its possible estimates and tests. 
    Includes estimates and tests on mean_difference and variance_ratio
    Can handle paired_data
Attributes
    datax and datay : two One_Sample objects (datay is not available if data are paired)
    paired : boolean to indicate if data are paired (in this case, y is null and x is the difference between x and y)
    var_equal : facultative boolean. To indicate if the two samples have the same variance. if None, default value is true.
    mean_difference_estimate = estimate of the difference between the means of x and y
    variance_ratio_estimate = estimate of the ratio between the variances of x and y (not available if data are paired)
Methods
    change_var_equal : permet de changer la valeur de l'attribut var_equal
    mean_difference_confidence_interval : intervalle de confiance pour la différence entre les moyennes de x et de y
    mean_difference_ttest : test sur la différence de valeur entre les moyennes de x et de y
    variance_ratio_confidence_interval :  intervalle de confiance pour le ratio des variances de x et de y
    variance_ratio_fishertest : test sur la valeur du ratio des variances de x et de y
    comment_mean_difference_estimate
    comment_mean_difference_confidence_interval
    comment_mean_difference_ttest
    comment_variance_ratio_estimate
    comment_variance_ratio_confidence_interval
    comment_variance_ratio_fishertest
"""
class TwoSamples(GenericFunctions):
    
    def __init__(self, x, y, paired=False, var_equal=None):
        GenericFunctions.__init__(self)
        try:
            self.checktype_ndarray(x, "x")
            self.checktype_ndarray(y, "y")
            x = np.array(x)
            y = np.array(y)
            if paired:
                if len(x) != len(y):
                    raise ValueError("Value Error : x and y must be of the same size if they are paired.")
                x = x - y
                y = np.nan
            else:
                self.checknumber_equal_value(y.ndim, 1, "the dimension of the array y")
                y = y[~np.isnan(y)]
                self.checknumber_sup_value(len(y), 3, "the number of non-null data in the array", orequal=True)
            self.checknumber_equal_value(x.ndim, 1, "the dimension of the array x")
            x = x[~np.isnan(x)]
            self.checknumber_sup_value(len(x), 3, "the number of non-null data in the array", orequal=True)
        except (TypeError, ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()))
        else:
            self.datax = OneSample(x)
            self.paired = paired
            self.var_equal = var_equal
            if paired:
                self.datay = np.nan
                self.mean_difference_estimate = self.datax.mean.estimate
                self.variance_ratio_estimate = np.nan
            else:
                self.datay = OneSample(y)
                self.mean_difference_estimate = self.datax.mean.estimate - self.datay.mean.estimate
                self.variance_ratio_estimate = self.datax.var.estimate / self.datay.var.estimate
    
    def change_var_equal(self, var_equal):
        if (isinstance(var_equal, bool)) | (var_equal is None):
            self.var_equal = var_equal
    
    def __get_var_equal(self, forced_var_equal):
        if forced_var_equal is None:
            if self.var_equal is None:
                forced_var_equal = True
            else:
                forced_var_equal = self.var_equal
        return forced_var_equal
    
    """
    Intervalle de confiance symétrique pour la différence entre les moyennes
    Paramètre
        confidence_level: niveau de confiance pour l'intervalle de confiance
        forced_var_equal: permet de forcer l'égalité ou l'inégalité des variances pour le calcul de l'intervalle de confiance
    """
    def mean_difference_confidence_interval(self, confidence_level=0.95, forced_var_equal=None):
        var_equal = self.__get_var_equal(forced_var_equal)
        alpha = 1 - confidence_level
        if self.paired:
            dof = self.datax.size - 1
            stderr = np.sqrt(self.datax.var.estimate / self.datax.size)
            tstat = self.mean_difference_estimate / stderr
        else:
            if var_equal:
                dof = self.datax.size + self.datay.size - 2
                v = (self.datax.size - 1) * self.datax.var.estimate
                v = v + (self.datay.size - 1) * self.datay.var.estimate
                v = v / dof
                stderr = np.sqrt(v * ((1/self.datax.size) + (1/self.datay.size)) )
            else:
                stderrx = np.sqrt(self.datax.var.estimate / self.datax.size)
                stderry = np.sqrt(self.datay.var.estimate / self.datay.size)
                stderr = np.sqrt(stderrx**2 + stderry**2)
                dof = stderr**4 / ((stderrx**4 / (self.datax.size-1)) + (stderry**4 / (self.datay.size-1)))
            tstat = self.mean_difference_estimate / stderr
        confidence_interval = ((tstat - st.t.ppf(1 - (alpha / 2), dof)) * stderr, (tstat + st.t.ppf(1 - (alpha / 2), dof)) * stderr)
        return confidence_interval
    
    """
    Test de Student pour déterminer si deux moyennes sont égales (tested_difference = 0)
        Le test peut aller au-delà en testant d'autres niveaux de différence entre les deux moyennes
    Paramètres
        tested_difference : différence entre les moyennes testée  ( H0 : mu(X) - mu(Y) == tested_difference )
        alpha: niveau de risque de première espèce du test (rejet de H0 à tort)
        alternative = détermine l'hypothèse alternative. 3 alternatives possibles :
            - 'two-sided' : H1 : mu(X) - my(Y) != tested_difference
            - 'smaller' : H1 : mu(X) - my(Y) < tested_difference
            - 'larger' : H1 : mu(X) - my(Y) > tested_difference
        forced_var_equal = permet de forcer l'égalité ou l'inégalité des variances pour le test
    Valeurs retournées
        tstat : valeur de la statistique t testée
        dof : degrés de liberté de loi t de Student utilisée
        p_value : p_value du test
        confidence_interval : intervalle de confiance pour la différence des moyennes (symétrique ou asymétrique selon H1)
    """
    def mean_difference_ttest(self, tested_difference=0, alpha=0.05, alternative="two-sided", forced_var_equal=None):
        var_equal = self.__get_var_equal(forced_var_equal)
        if self.paired:
            dof = self.datax.size - 1
            stderr = np.sqrt(self.datax.var.estimate / self.datax.size)
            tstat = (self.mean_difference_estimate - tested_difference) / stderr
        else:
            if var_equal:
                dof = self.datax.size + self.datay.size - 2
                v = (self.datax.size - 1) * self.datax.var.estimate
                v = v + (self.datay.size - 1) * self.datay.var.estimate
                v = v / dof
                stderr = np.sqrt(v * ((1/self.datax.size) + (1/self.datay.size)) )
            else:
                stderrx = np.sqrt(self.datax.var.estimate / self.datax.size)
                stderry = np.sqrt(self.datay.var.estimate / self.datay.size)
                stderr = np.sqrt(stderrx**2 + stderry**2)
                dof = stderr**4 / ((stderrx**4 / (self.datax.size-1)) + (stderry**4 / (self.datay.size-1)))
            tstat = (self.mean_difference_estimate - tested_difference) / stderr
        confidence_level = 1 - alpha
        if alternative == "smaller":
            p_value = st.t.cdf(tstat, dof)
            pre_confidence_interval = (-np.inf, tstat + st.t.ppf(confidence_level, dof))
        elif alternative == "larger":
            p_value = 1 - st.t.cdf(tstat, dof)
            pre_confidence_interval = (tstat - st.t.ppf(confidence_level, dof), np.inf)
        else:
            p_value = 2 * st.t.cdf(-np.abs(tstat), dof)
            pre_confidence_interval = (tstat - st.t.ppf(1 - (alpha / 2), dof), tstat + st.t.ppf(1 - (alpha / 2), dof))
        confidence_interval = (tested_difference + pre_confidence_interval[0] * stderr, tested_difference + pre_confidence_interval[1] * stderr)
        return tstat, dof, p_value, confidence_interval

    """
    Intervalle de confiance symétrique pour le ratio entre les variances de datax et datay (non disponible si les données sont appariées)
    Paramètre
        confidence_level: niveau de confiance pour l'intervalle de confiance
    """
    def variance_ratio_confidence_interval(self, confidence_level=0.95):
        try:
            if self.paired:
                raise ValueError("The variance ratio is not available for paired samples")
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()))
        else:
            alpha = 1 - confidence_level
            numerator_dof = self.datax.size - 1
            denominator_dof = self.datay.size - 1
            confidence_interval = (self.variance_ratio_estimate / st.f.ppf(1 - (alpha / 2), numerator_dof, denominator_dof), 
                                   self.variance_ratio_estimate / st.f.ppf(alpha / 2, numerator_dof, denominator_dof))
            return confidence_interval
    
    """
    test de Fisher pour déterminer si deux variances sont égales (tested_ratio = 1) (non disponible si les données sont appariées)
    le test peut aller au-delà en testant d'autres niveaux de ratio entre les deux variances
    Paramètres
        tested_ratio = ratio cible des variances. ( H0 : var(X) / var(Y) == tested_ratio )
        alpha: niveau de risque de première espèce du test (rejet de H0 à tort)
        alternative = détermine l'hypothèse alternative. 3 alternatives possibles :
            - 'two-sided' : H1 : var(X) / var(Y) != tested_ratio
            - 'smaller' : H1 : var(X) / var(Y) < tested_ratio
            - 'larger' : H1 : var(X) / var(Y) > tested_ratio
    Valeurs retournées
        fisher_stat : valeur de la statistique de Fisher testée
        numerator_dof = degré de liberté du numérateur
        denominator_dof = degré de liberté du dénominateur
        p_value : p_value du test
        confidence_interval : intervalle de confiance du ratio (symétrique ou asymétrique selon H1)
    """
    def variance_ratio_fishertest(self, tested_ratio=1, alternative="two-sided", alpha=0.05):
        try:
            if self.paired:
                raise ValueError("The variance ratio is not available for paired samples")
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()))
        else:
            numerator_dof = self.datax.size - 1
            denominator_dof = self.datay.size - 1
            fisherstat = self.variance_ratio_estimate / tested_ratio
            smaller_value = st.f.cdf(fisherstat, numerator_dof, denominator_dof)
            larger_value =  1 - smaller_value
            if alternative == "two-sided":
                p_value = 2 * min(smaller_value, larger_value)
                beta = alpha / 2
                confidence_interval = (self.variance_ratio_estimate / st.f.ppf(1-beta, numerator_dof, denominator_dof), 
                                       self.variance_ratio_estimate / st.f.ppf(beta, numerator_dof, denominator_dof))
            elif alternative == "smaller":
                p_value = smaller_value
                confidence_interval = (0, self.variance_ratio_estimate / st.f.ppf(alpha, numerator_dof, denominator_dof))
            else:
                p_value = larger_value
                confidence_interval = (self.variance_ratio_estimate / st.f.ppf(1-alpha, numerator_dof, denominator_dof), np.inf)
            return fisherstat, numerator_dof, denominator_dof, p_value, confidence_interval
    
    
    def non_parametric_test(self):
        if self.paired:
            test_stat, p_value = st.wilcoxon(self.datax.data)
        else:
            test_stat, p_value = st.mannwhitneyu(self.datax.data, self.datay.data)
        return test_stat, p_value
    
    # comment on mean difference estimate
    def comment_mean_difference_estimate(self):
        comment = "L'estimateur de la différence entre les moyennes de x et de y est mean-difference = {}.".format(self.mean_difference_estimate)
        return comment
    
    # comment on mean difference confidence interval
    def comment_mean_difference_confidence_interval(self, confidence_level=0.95, forced_var_equal=None):
        cint = self.mean_difference_confidence_interval(confidence_level=0.95, forced_var_equal=None)
        comment = "L'intervalle de confiance de la différence entre les moyennes de x et de y est {} au niveau de confiance {:.2%}.".format(cint, confidence_level)
        return comment
    
    """
    comment on mean_difference ttest
    Parameter short_comment
        True : one-sentence summary of the test result
        False (default) : details on the test and its results
    """
    def comment_mean_difference_ttest(self, tested_difference=0, alternative="two-sided", alpha=0.05, short_comment=False, forced_var_equal=None):
        var_equal = self.__get_var_equal(forced_var_equal)
        tstat, dof, p_value, cint = self.mean_difference_ttest(tested_difference=tested_difference, alternative=alternative, alpha=alpha, forced_var_equal=var_equal)
        if self.paired:
            estim_object = "la moyenne de x-y"
            text_difference = tested_difference
            precision_txt = "la moyenne de x-y"
        else:
            precision_txt = "la différence entre les moyennes de x et de y"
            if tested_difference == 0:
                estim_object = "la moyenne de x"
                text_difference = "la moyenne de y"
            else:
                estim_object = "la différence entre les moyennes de x et de y"
                text_difference = tested_difference    
        if short_comment:
            comment = self.generic_shortcomment(estim_object=estim_object, tested_value=text_difference, p_value=p_value, alpha=alpha, alternative=alternative)
        else:
            if self.paired:
                comment = "Méthode = test t de student pour comparer les moyennes de deux échantillons appariés (dépendants).\n"
            else:
                if var_equal:
                    comment = "Méthode = test t de student pour comparer les moyennes de deux échantillons avec la même variance.\n"
                else:
                    comment = "Méthode = test t de Welch pour comparer les moyennes de deux échantillons avec des variances différentes.\n"
            comment += self.generic_longcomment(test_object="la statistique T", stat_value=tstat, stat_law="loi de Student à {:.2f} degrés de liberté".format(dof),
                  estim_object=estim_object, estim_value=self.mean_difference_estimate, cint=cint, tested_value=text_difference, p_value=p_value, alpha=alpha, alternative=alternative, precision_txt=precision_txt)
        return comment
    
    # comment on variance ratio estimate
    def comment_variance_ratio_estimate(self):
        if not self.paired:
            comment = "L'estimateur du ratio entre les variances de x et de y est variance_ratio = {}.".format(self.variance_ratio_estimate)
        else:
            comment = "No variance ratio calculation for paired samples"
        return comment
    
    # comment on variance ratio confidence interval
    def comment_variance_ratio_confidence_interval(self, confidence_level=0.95):
        if not self.paired:
            cint = self.variance_ratio_confidence_interval(confidence_level=0.95)
            comment = "L'intervalle de confiance du ratio des variances de x et de y est {} au niveau de confiance {:.2%}.".format(cint, confidence_level)
        else:
            comment = "No variance ratio calculation for paired samples"
        return comment
    
    """
    comment on variance ratio fisher test
    Parameter short_comment
        True : one-sentence summary of the test result
        False (default) : details on the test and its results
    """
    def comment_variance_ratio_fishertest(self, tested_ratio=1, alternative="two-sided", alpha=0.05, short_comment=False):
        fisherstat, numerator_dof, denominator_dof, p_value, cint = self.variance_ratio_fishertest(tested_ratio=tested_ratio, alpha=alpha, alternative=alternative)
        if tested_ratio == 1:
            estim_object = "la variance de x"
            text_difference = "la variance de y"
        else:
            estim_object = "le ratio des deux variances"
            text_difference = tested_ratio
        if short_comment:
            comment = self.generic_shortcomment(estim_object=estim_object, tested_value=text_difference, p_value=p_value, alpha=alpha, alternative=alternative)
        else:
            precision_txt = "le ratio des deux variances"
            comment = "Test F de Fisher pour comparer les variances de deux échantillons de taille {} et {}.\n".format(self.datax.size, self.datay.size)
            comment += self.generic_longcomment(test_object="la statistique F", stat_value=fisherstat, stat_law="loi de Fisher à {} et {} degrés de liberté".format(numerator_dof, denominator_dof),
                          estim_object=estim_object, estim_value=self.variance_ratio_estimate, cint=cint, tested_value=text_difference, p_value=p_value, alpha=alpha, alternative=alternative, precision_txt=precision_txt)
        return comment
    
    def comment_non_parametric_test(self, alpha=0.05, short_comment=False):
        test_stat, p_value = self.non_parametric_test()
        comment = self.test2laws_comment(test_stat, p_value, alpha, short_comment)
        if not short_comment:
            if self.paired:
                if (self.datax.size > 20) & (self.datay.size > 20):
                    comment = "Test non paramétrique de Wilcoxon pour les échantillons de taille {} et {}.\n".format(self.datax.size, self.datay.size) + comment
                else:
                    comment = "Test non paramétrique de Wilcoxon pour les échantillons de taille {} et {}.\nATTENTION, pour ce test, il est conseillé que la taille de chaque échantillon soit strictement supérieure à 20.\n".format(self.datax.size, self.datay.size) + comment
            else:
                if (self.datax.size > 20) & (self.datay.size > 20):
                    comment = "Test non paramétrique de Mann-Whitney pour les échantillons de taille {} et {}.\n".format(self.datax.size, self.datay.size) + comment
                else:
                    comment = "Test non paramétrique de Mann-Whitney pour les échantillons de taille {} et {}.\nATTENTION, pour ce test, il est conseillé que la taille de chaque échantillon soit strictement supérieure à 20.\n".format(self.datax.size, self.datay.size) + comment
        return comment


"""
Class : FreqEmpirique
Empirical frequencies for categorical or discrete numeric data. Includes tests for goodness of fit with binomial theoritical frequencies, or any other theoritical frequencies table
Attributes
    possible_values : tableau des valeurs possibles pour la variable
    freq_empirique : tableau des fréquences empiriques constatées
    n_trials : nombre d'expériences menées
    n_modalites : nombre de modalités différentes (résultats différents)
Methods
    test_chi2_loi_binomiale : test pour vérifier si les fréquences empiriques correspondent à la distribution d'une loi binomiale
    test_chi2_loi_discrete : test pour vérifier si les fréquences empiriques correspondent à la distribution d'une loi dont l'on donne les fréquences théoriques
    comment_test_chi2_loi_binomiale
    comment_test_chi2_loi_discrete
"""
class FreqEmpirique(GenericFunctions):
    
    def __init__(self, valeurs_possibles, freq_empiriques):
        GenericFunctions.__init__(self)
        try:
            self.checktype_ndarray(valeurs_possibles, "valeurs_possibles")
            self.checktype_ndarray(freq_empiriques, "freq_empiriques")
            valeurs_possibles = np.array(valeurs_possibles)
            freq_empiriques = np.array(freq_empiriques)
            self.checknumber_equal_value(valeurs_possibles.ndim, 1, "the dimension of valeurs_possibles")
            self.checknumber_equal_value(freq_empiriques.ndim, 1, "the dimension of freq_empiriques")
            valeurs_possibles = valeurs_possibles[~np.isnan(valeurs_possibles)]
            freq_empiriques = freq_empiriques[~np.isnan(freq_empiriques)]
            self.checknumber_n1_equal_n2(len(valeurs_possibles), "the size of possible values", len(freq_empiriques), "the size of empirical frequencies")
            self.checknumber_sup_value(len(freq_empiriques), 3, "the number of possible values", orequal=True)
        except (TypeError, ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()))
        else:
            self.possible_values = valeurs_possibles
            self.freq_empirique = freq_empiriques
            self.n_trials = freq_empiriques.sum()
            self.n_modalites = len(freq_empiriques)
    
    """
    Test du chi2 pour déterminer si la distribution d'un échantillon suit une loi binomiale d'arguments n et p 
    Paramètres
        n: nombre d'expériences de la loi binomiale
        p: probabilité de succès pour chaque expérience de la loi binomiale
        group_modalites: tableau de listes des modalités à regrouper. ex:[(0,1)] ou [(0,1),(11,12,13)]
    Valeurs retournées
        f_obs : fréquence empirique avec les regroupements éventuels
        f_loi : fréquence théorique avec les regroupements éventuels
        chi2stat = valeur de la statistique chi2 testée
        dof = degré de liberté de la loi du chi2 utilisée pour le test
        p_value = p-value du test
    """
    def test_chi2_loi_binomiale(self, n, p, group_modalites=None):
        try:
            self.checknumber_n1_equal_n2(self.n_modalites, "the number of possible values", (n + 1), "(n + 1)")
            freq_theorique = st.binom.pmf(self.possible_values, n, p) * self.n_trials
            f_obs, f_loi = self.__group_levels(group_modalites, freq_theorique)
            self.checknumber_sup_value(len(f_obs), 3, "the number of grouped levels", orequal=True)
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()))
        else:
            dof = len(f_obs) - 1
            chi2stat, p_value = st.chisquare(f_obs, f_exp=f_loi)
            return f_obs, f_loi, chi2stat, dof, p_value

    """
    Test du chi2 pour déterminer si la distribution d'un échantillon suit une loi discrète dont la distribution théorique est donnée en argument
    Paramètres
        freq_theorique: distribution théorique selon la loi testée
        group_modalites: tableau de liste(s) des modalités à regrouper. ex:[(0,1)] ou [(0,1),(11,12,13)]
    Valeurs retournées
        f_obs : fréquence empirique avec les regroupements éventuels
        f_loi : fréquence théorique avec les regroupements éventuels
        chi2stat = valeur de la statistique chi2 testée
        dof = degré de liberté de la loi du chi2 utilisée pour le test
        p_value = p-value du test
    """
    def test_chi2_loi_discrete(self, freq_theorique, group_modalites=None):
        try:
            self.checknumber_n1_equal_n2(self.n_modalites, "the number of levels in the empirical frequency", len(freq_theorique), "the number of levels in the theoretical frequency")
            f_obs, f_loi = self.__group_levels(group_modalites, freq_theorique)
            self.checknumber_sup_value(len(f_obs), 3, "the number of grouped levels", orequal=True)
        except (ValueError) as e:
            lg.error(e)
        except:
            lg.critical("Unexpected error : {}".format(sys.exc_info()))
        else:
            dof = len(f_obs) - 1
            chi2stat, p_value = st.chisquare(f_obs, f_exp=f_loi)
            return f_obs, f_loi, chi2stat, dof, p_value
        
    """
    comment on test chi2 loi binomiale
    Parameter short_comment
        True : one-sentence summary of the test result
        False (default) : details on the test and its results
    """
    def comment_test_chi2_loi_binomiale(self, n, p, group_modalites=None, alpha=0.05, short_comment=False):
        f_obs, f_loi, chi2stat, dof, p_value = self.test_chi2_loi_binomiale(n=n, p=p, group_modalites=group_modalites)
        comment = self.testlaw_comment("binomiale", "chi2", chi2stat, p_value, alpha, short_comment, complement_loi="loi du chi2 à {} degrés de liberté".format(dof))
        if not short_comment:
            n_alert = False
            for i in np.arange(dof + 1):
                if (f_obs[i] < 5) | (f_loi[i] < 5):
                    n_alert = True
            if n_alert:
                comment = "Test du chi2 de distribution binomiale pour l'échantillon de taille {}.\nATTENTION, CERTAINES VALEURS DES FREQUENCES OBSERVEES OU THEORIQUES SONT TROP PETITES (<5). IL FAUT LES REGROUPER.\n".format(self.n_trials) + comment
            else:
                comment = "Test du chi2 de distribution binomiale pour l'échantillon de taille {}.\n".format(self.n_trials) + comment
        return comment
    
    """
    comment on test chi2 loi discrete
    Parameter short_comment
        True : one-sentence summary of the test result
        False (default) : details on the test and its results
    """
    def comment_test_chi2_loi_discrete(self, freq_theorique, group_modalites=None, alpha=0.05, short_comment=False):
        f_obs, f_loi, chi2stat, dof, p_value = self.test_chi2_loi_discrete(freq_theorique=freq_theorique, group_modalites=group_modalites)
        comment = self.testlaw_comment("discrète conforme à la fréquence théorique donnée", "chi2", chi2stat, p_value, alpha, short_comment, complement_loi="loi du chi2 à {} degrés de liberté".format(dof))
        if not short_comment:
            n_alert = False
            for i in np.arange(dof + 1):
                if (f_obs[i] < 5) | (f_loi[i] < 5):
                    n_alert = True
            if n_alert:
                comment = "Test du chi2 de distribution discrète selon une fréquence théorique donnée pour l'échantillon de taille {}.\nATTENTION, CERTAINES VALEURS DES FREQUENCES OBSERVEES OU THEORIQUES SONT TROP PETITES (<5). IL FAUT LES REGROUPER.\n".format(self.n_trials) + comment
            else:
                comment = "Test du chi2 de distribution discrète selon une fréquence théorique donnée pour l'échantillon de taille {}.\n".format(self.n_trials) + comment
        return comment
        
    """
    Permet de regrouper les niveaux où le nombre d'observations n'est pas suffisant
    ATTENTION : on ne peut regrouper que des niveaux adjacents
    """
    def __group_levels(self, group_modalites, freq_theorique):
        if group_modalites is None:
            f_obs = self.freq_empirique
            f_loi = freq_theorique
        else:
            f_obs = []
            f_loi = []
            cpt = -1
            current_group = -1
            group_exist = False
            for i in np.arange(self.n_modalites):
                cpt += 1
                i_grouped = False
                check_group = -1
                for modalites in group_modalites:
                    check_group += 1
                    for modalite in modalites:
                        if modalite == i:
                            i_grouped = True
                            if current_group != check_group:
                                current_group = check_group
                                group_exist = False
                            else:
                                group_exist = True
                if not i_grouped:
                    f_obs.append(self.freq_empirique[i])
                    f_loi.append(freq_theorique[i])
                    group_exist = False
                else:
                    if not group_exist: # si le groupe n'existe pas, on le crée
                        f_obs.append(0)
                        f_loi.append(0)
                    else: # si le groupe existe déjà, il faut diminuer cpt de 1 pour ajouter les fréquences au groupe
                        cpt = cpt - 1
                    f_obs[cpt] = f_obs[cpt] + self.freq_empirique[i]
                    f_loi[cpt] = f_loi[cpt] + freq_theorique[i]
        return f_obs, f_loi
    

"""
Class : TwoProportions
Calculates estimate, confidence_interval and test on the difference between two proportions
Attributes
    proportion1 : object Proportion
    proportion2 : object Proportion
    p_difference_estimate : estimate of the difference between proportion 1 and porportion 2
Methods
    p_difference_confidence_interval
    p_difference_chi2test : test pour vérifier si la différence entre les 2 proportions est égale à une valeur donnée
    comment_p_difference_chi2test
"""
class TwoProportions(GenericFunctions):
    
    def __init__(self, n_successes_1, n_trials_1, n_successes_2, n_trials_2):
        GenericFunctions.__init__(self)
        self.proportion1 = Proportion(n_successes_1, n_trials_1)
        self.proportion2 = Proportion(n_successes_2, n_trials_2)
        self.p_difference_estimate = self.proportion1.estimate - self.proportion2.estimate
    
    """
    Intervalle de confiance symétrique pour la différence entre les deux proportions
    Paramètre
        confidence_level: niveau de confiance pour l'intervalle de confiance
    """
    def p_difference_confidence_interval(self, confidence_level=0.95):
        stderr = np.sqrt((self.proportion1.estimate * (1 - self.proportion1.estimate) / self.proportion1.n_trials) + 
                         (self.proportion2.estimate * (1 - self.proportion2.estimate) / self.proportion2.n_trials)) 
        z = st.norm.ppf((1 + confidence_level) / 2)
        confidence_interval = (self.p_difference_estimate - (z * stderr), self.p_difference_estimate + (z * stderr))
        return confidence_interval

    """
    Test avec la loi normale pour déterminer si deux proportions sont égales (tested_delta_p = 0)
        Le test peut aller au-delà en testant d'autres niveaux de différence entre les deux proportions
    Paramètres
        tested_delta_p : niveau de différence entre les 2 proportions testé ( H0 : p1 - p2 == tested_delta_p )
        alpha: niveau de risque de première espèce du test (rejet de H0 à tort)
        alternative = détermine l'hypothèse alternative. 3 alternatives possibles:
            - 'two-sided' : H1 : p1 - p2 != tested_delta_p
            - 'smaller' : H1 : p1 - p2 < tested_delta_p
            - 'larger' : H1 : p1 - p2 > tested_delta_p
    Valeurs retournées
        zstat_test : valeur de la statistique z testée
        p_value : p_value du test
        confidence_interval : intervalle de confiance pour la différence des proportions (symétrique ou asymétrique selon H1)
    """
    def p_difference_chi2test(self, tested_delta_p=0, alpha=0.05, alternative="two-sided"):
        stderr = np.sqrt((self.proportion1.estimate * (1 - self.proportion1.estimate) / self.proportion1.n_trials) + 
                         (self.proportion2.estimate * (1 - self.proportion2.estimate) / self.proportion2.n_trials)) 
        confidence_level = 1 - alpha
        if alternative == "two-sided":
            z = st.norm.ppf((1 + confidence_level) / 2)
            confidence_interval = (self.p_difference_estimate - (z * stderr), self.p_difference_estimate + (z * stderr))
        else:
            z = st.norm.ppf(confidence_level)
            if alternative == "smaller":
                confidence_interval = (-1, self.p_difference_estimate + (z * stderr))
            else:
                confidence_interval = (self.p_difference_estimate - (z * stderr), 1)
        zstat, p_value = sm.proportion.proportions_ztest([self.proportion1.n_successes, self.proportion2.n_successes], 
                 [self.proportion1.n_trials, self.proportion2.n_trials], value=tested_delta_p, alternative=alternative)
        return zstat, p_value, confidence_interval
        
    # comment on estimate
    def comment_p_difference_estimate(self):
        comment = "L'estimateur de la différence entre les deux proportions de succès est p-difference = {}.".format(self.p_difference_estimate)
        return comment
    
    # comment on confidence_interval
    def comment_p_difference_confidence_interval(self, confidence_level=0.95):
        cint = self.p_difference_confidence_interval(confidence_level=0.95)
        comment = "L'intervalle de confiance de la différence entre les deux proportions de succès est {} au niveau de confiance {:.2%}.".format(cint, confidence_level)
        return comment
    
    """
    comment on p_difference chi2 test
    Parameter short_comment
        True : one-sentence summary of the test result
        False (default) : details on the test and its results
    """
    def comment_p_difference_chi2test(self, tested_delta_p=0, alpha=0.05, alternative="two-sided", short_comment=False):
        zstat, p_value, cint = self.p_difference_chi2test(tested_delta_p=tested_delta_p, alpha=alpha, alternative=alternative)
        if tested_delta_p == 0:
            estim_object = "la proportion 1"
            text_difference = "la proportion 2"
        else:
            estim_object = "la différence entre les proportions 1 et 2"
            text_difference = tested_delta_p
        if short_comment:
            comment = self.generic_shortcomment(estim_object=estim_object, tested_value=text_difference, p_value=p_value, alpha=alpha, alternative=alternative)
        else:
            comment = "Test de la différence entre deux proportions.\n"
            comment += self.generic_longcomment(test_object="la statistique z", stat_value=zstat, stat_law="loi normale centrée réduite",
                  estim_object=estim_object, estim_value=self.p_difference_estimate, cint=cint, tested_value=text_difference, p_value=p_value, alpha=alpha, alternative=alternative)
        return comment
