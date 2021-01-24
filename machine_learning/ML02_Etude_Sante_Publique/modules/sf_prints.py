# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:32:58 2020

@author: Sylvain Friot

Content: print results functions
"""

def print_one_test(test_object, h0, h1, p_value, stat_law=None, stat_value=None, alpha=0.05):
    print("\033[1m" + test_object + " (niveau de risque = {:.0%})".format(alpha) + "\033[0m")
    print("   H0: {}".format(h0))
    print("   H1: {}".format(h1))
    if stat_law is not None:
        if stat_value is None:
            print("Statistique étudiée : {}".format(stat_law))
        else:
            print("Statistique étudiée : {} = {:.3f}".format(stat_law, stat_value))
    print("p-value = {:.3f}".format(p_value))
    print("\033[1m" + "Accept H0 : {}".format(p_value >= alpha) + "\033[0m")
    print("")
    
def print_df_tests(test_object, h0, h1, df_p_values, df_index=None, alpha=0.05, with_transpose=True):
    print("\033[1m" + test_object + " (niveau de risque = {:.0%})".format(alpha) + "\033[0m")
    print("   H0: {}".format(h0))
    print("   H1: {}".format(h1))
    print("")
    if df_index is not None:
        df_p_values.set_index(df_index, inplace=True)
    if with_transpose:
        df_print = df_p_values.T.to_string(justify='center')
    else:
        df_print = df_p_values.to_string(justify='center')
    print(df_print)
    print("")
    
def print_df_h0h1(test_object, df_h0h1, df_index=None, alpha=0.05):
    print("\033[1m" + test_object + " (niveau de risque = {:.0%})".format(alpha) + "\033[0m")
    df_h0h1["Accept H0"] = df_h0h1.p_value >= alpha
    if df_index is not None:
        df_h0h1.set_index(df_index, inplace=True)
    print(df_h0h1.T)
    print("")
    
def print_estimation(estim_object, estim_value, cint_value, confidence_level):
    print("\033[1m" + estim_object + " (niveau de confiance = {:.0%})".format(confidence_level) + "\033[0m")
    print("   Estimation = {}".format(estim_value))
    print("   Intervalle de confiance = [{} ; {}]".format(cint_value[0], cint_value[1]))
    print("")
    