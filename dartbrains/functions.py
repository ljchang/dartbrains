# -*- coding: utf-8 -*-

'''
dartbrains functions
===================

functions to help perform data analysis

'''

import numpy as np
import pandas as pd

def find_spikes(data, global_spike_cutoff=3, diff_spike_cutoff=3):
    '''Function to identify spikes from fMRI Time Series Data

        Args:
            data: Brain_Data instance
            global_spike_cutoff: (int,None) cutoff to identify spikes in global signal
                                 in standard deviations, None indicates do not calculate.
            diff_spike_cutoff: (int,None) cutoff to identify spikes in average frame difference
                                 in standard deviations, None indicates do not calculate.
        Returns:
            pandas dataframe with spikes as indicator variables
    '''

    if (global_spike_cutoff is None) & (diff_spike_cutoff is None):
        raise ValueError('Did not input any cutoffs to identify spikes in this data.')

    if global_spike_cutoff is not None:
        global_mn = np.mean(data.data, axis=1)
        global_outliers = np.append(np.where(global_mn > np.mean(global_mn) + np.std(global_mn) * global_spike_cutoff),
                                    np.where(global_mn < np.mean(global_mn) - np.std(global_mn) * global_spike_cutoff))

    if diff_spike_cutoff is not None:
        frame_diff = np.mean(np.abs(np.diff(data.data, axis=0)), axis=1)
        frame_outliers = np.append(np.where(frame_diff > np.mean(frame_diff) + np.std(frame_diff) * diff_spike_cutoff),
                                   np.where(frame_diff < np.mean(frame_diff) - np.std(frame_diff) * diff_spike_cutoff))
   # build spike regressors
    outlier = pd.DataFrame([x+1 for x in range(len(global_mn))],columns=['TR'])
    if (global_spike_cutoff is not None):
        for i, loc in enumerate(global_outliers):
            outlier['global_spike' + str(i + 1)] = 0
            outlier['global_spike' + str(i + 1)].iloc[int(loc)] = 1

    # build FD regressors
    if (diff_spike_cutoff is not None):
        for i, loc in enumerate(frame_outliers):
            outlier['diff_spike' + str(i + 1)] = 0
            outlier['diff_spike' + str(i + 1)].iloc[int(loc)] = 1
    return outlier
