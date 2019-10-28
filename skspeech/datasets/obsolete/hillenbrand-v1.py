# Hillenbrand Database 
#
#The function load_hillenbrand() loads the database in a similar way as the databases supported by the sklearn load_datasets.
#However, there are arguments that lets one select parts of the database for further use.
#The subset selection is specified by the arguments  (genders, vowels, features, targets)
#
#The output is provided in a Bunch object with following parameters
#- data: raw data for selected features
#- target: classification labels (possibly multi-class)
#- ...
#

import pandas as pd
from sklearn.datasets.base import Bunch

def load_hildata():
    ''' 
    The function load_hildata() loads the full Hillenbrand dataset in a panda's dataframe
    A more selective version of the same is found in load_hillenbrand():
    '''
    hildata = pd.read_csv('ftp://ftp.esat.kuleuven.be/psi/speech/data/hillenbrand/vowdata.csv',index_col=0)  
    return hildata
    
def load_hillenbrand(genders=[],vowels=[],features=[],targets=[],Debug=False,Output='Bunch'):

    '''
    The function load_hillenbrand() loads the Hillenbrand dataset in a similar way as the datasets in sklearn.
    There are extra arguments that lets one select parts of the database for further use.
    The subset selection is specified by the arguments  (genders, vowels, features, targets)

    The Hillenbrand dataset is a 1995 repeat and extension of the classic Peterson-Barney(1953) experiment
    in which Formants are established as compact and highly discriminative features for vowel recognition
    (c) 1995 James Hillenbrand
    https://homepages.wmich.edu/~hillenbr/voweldata.html
    
    The interface provided here reads from an ftp copy of the data at ESAT stored in a more 
    convenient csv formant and in which the 0 values (not available) are replaced by #N/A

    =================   ==============
    Classes (genders)                4 (m,w,b,g)
    Classes (vowels)                12 (ae,ah,aw,eh,er,ei,ih,iy,oa,oo,uh,uw)
    Samples per class (genders)     (50,50,29,22)
    Samples per class (vowels)       1 per speaker (100 for adults, 50 for children)
    Samples total                 1668
    Dimensionality                  19
    Features            real, positive
    =================   ==============
    
    With this interface you can load the specific parts of the database (subset of speakers, vowels and features)
    that you want to use
   
    Parameters
    ----------
        genders:  list of selected genders (empty=all)
        vowels:   list of selected vowels (empty=all)
        features: list of selected features (empty=all)
        targets:  list of targets (empty is equivalent to ['vid','gid'])
        Output:   Bunch(def) or DataFrame
        Debug:    False(def) or True
        
    Returns
    -------
        data : Bunch
        Dictionary-like object, with attributes:
            'data', the data to learn, 
            'target', the classification labels,
            'target_names', the meaning of the columns in target
            'feature_names', the meaning of the features
    
    '''
    
    # STEP 1: READ ALL THE DATA
    ###########################
    # the dataframe 'columns' contain the fields in our dataset, the first 3 fields are identifiers 
    # for vowel, gender and speaker, the remaining 19 are datafields
    # print(hildata.columns)
    # the dataframe 'index' allows for accessing the records by name, it is the file-id in the first column
    # print(hildata.index)
    #    
    hildata = pd.read_csv('ftp://ftp.esat.kuleuven.be/psi/speech/data/hillenbrand/vowdata.csv',index_col=0)        

    allgenders = list(hildata['gid'].unique())
    allvowels = list(hildata['vid'].unique())
    allspeakers = list(hildata['sid'].unique())
    allfeatures = list(hildata.columns[3:].values)
    alltargets = ['vid','gid','sid']
    if(Debug):
        print(hildata.head())
        print('Genders(%3d) :' % len(allgenders),type(allgenders),allgenders)
        print('Vowels(%3d)  :' % len(allvowels),type(allvowels),allvowels)
        print('Features(%3d):' % len(allfeatures),type(allfeatures),allfeatures) 
        
    # STEP 2: select the appropriate data records
    #############################################
    if type(genders) is str:
        if genders == 'adults':
            genders = ['m','w']
        elif genders == 'children':
            genders = ['b','g']
        elif genders == 'male':
            genders = ['m','b']
        elif genders == 'female':
            genders = ['w','g']
        else:
            genders = []
                
    if type(vowels) is str:
        if vowels == 'vowels6':
            vowels = ['aw','eh','er','ih','iy','uw']
        elif vowels == 'vowels3':
            vowels = ['aw','iy','uw']
        else:
            vowels = []
            
    if(len(genders) == 0):
        genders = allgenders        
    if not set(genders) <= set(allgenders):
        print("load_hillenbrand(): GENDER List Error")
        return
    
    if(len(vowels) == 0):
        vowels = allvowels
    if not set(vowels) <= set(allvowels):
        print("load_hillenbrand(): VOWEL List Error")
        return
    
    if(len(features) ==0):
        features = allfeatures
    if not set(features) <= set(allfeatures):
        print("load_hillenbrand(): FEATURE List Error")
        return
    
    if(len(targets) == 0):
        targets = ['vid','gid']
    if not set(targets) <= set(alltargets):
        print("load_hillenbrand(): TARGET List Error")
        return
         
    indx = hildata['gid'].isin(genders) & hildata['vid'].isin(vowels)
    if(Debug):
        print('Selected Genders(%3d):' % len(genders),genders)
        print('Selected Vowels(%3d):' % len(vowels),vowels)
        print(type(indx),type(indx.values))
        print(indx.values)
        
    # STEP 4: Select data and unpack for desired output format
    target = hildata.loc[indx,targets].values
    data = hildata.loc[indx,features].values

    if Output == 'DataFrame':
        df = hildata.loc[indx,targets+features]
        return df
    else:
        return Bunch(data=data,target=target, target_names=targets, feature_names=features)