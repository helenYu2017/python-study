# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Time: 2018/09/07 18:40
# @author: Siyuan

import pandas as pd
import numpy as np
from genpro.PublicUtils import check_error
from genpro.PublicUtils.check_error import GenError


def main_logical_not(data):
    return genpro_logical_not(data)


def genpro_logical_not(data):
    """
    Add the input bool_array

    :param bool_array: array
    :return: the "not" result of a bool_array, DataFrame
    """
    
    check_error.check_df_type(data, -1)
    check_error.check_data_empty(data, -2)
    check_error.check_df_numeric(data, -8)

    result = np.logical_not(data.values)
    return  pd.DataFrame(result)

