# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:28:40 2020

@author: elisa
"""

import pandas as pd
from pathlib import Path

#%%
path = Path(__file__).parent.parent.parent
#%%
tmp = pd.read_pickle(path/'Data/Raw/raw_trivia.pickle')