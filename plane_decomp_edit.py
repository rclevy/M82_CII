#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 12:20:09 2018

@author: Eliz
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


def plane_direct(x, y, z, sigx, sigy, sigz, n, guess):

    def chi_2var(const, x, y, z):
        a = const[0]
        b = const[1]
        return np.nansum((z - a*x - b*y)**2/(sigz**2 + a**2*sigx**2 + b**2*sigy**2))


    result = opt.minimize(chi_2var, guess, args = (x, y, z), method = 'L-BFGS-B', options={'disp': False})

    a = result.x[0]
    b = result.x[1]

    hess = result.hess_inv.todense()
    err = np.sqrt(np.diag(hess))
    success = result.success

    a_err = err[0]
    b_err = err[1]

    chisq = chi_2var([a, b], x, y, z)/(n-2)

    if  not success:
        print 'Two comp Failed! ', result.message
        print 'Chisq: ', chisq
        print 'a: ', a

        a = np.nan
        b = np.nan
        a_err = np.nan
        b_err = np.nan
        chisq = np.nan


    ans = {'a': a, 'b': b, 'a_err': a_err, 'b_err': b_err, 'chisq': chisq, 'success': success}

    return ans
