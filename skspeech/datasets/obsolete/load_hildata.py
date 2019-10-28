# Hillenbrand Database 
#
# a simple Hillenbrand dat load version returning full dataset
# for more options look at hillenbrand()
#- ...
#
import pandas as pd
from sklearn.datasets.base import Bunch

def load_hildata():
   
    hildata = pd.read_csv('ftp://ftp.esat.kuleuven.be/psi/speech/data/hillenbrand/vowdata.csv',index_col=0)        
    return hildata