"""
File Utilities for accessing TIMIT style files
including conversion of TIMIT phone sets
"""


import os,sys
import numpy as np
import pandas as pd

def read_xlat_dic(fname,icol=0,ocol=1):
    """
    make a translation dictionary for mapping values from source column (icol) to 
    values in target column (ocol)
    """
    
    xlat_dic = {}
    with open(fname) as fp:
        for line in fp:
            w = line.strip().split()
            xlat_dic[w[icol]] = w[ocol]
    return(xlat_dic)

def xlat_seg(isegdf,xlat_dic,MERGE_CLOSURES=False):
    """
    convert input segmentation to an output segmentation
    and merge identical labels
    this means that glottal closures are mapped to intra-word silence segments, this is OK for frame based analysis
    for segmental analysis it may be better to group the glottal closures with their respective plosive part
    inputs:
        isegdf: input segmentation in panda dataframe format
        xlat_dic:  phone translation dictionary
        MERGE_CLOSURES: flag 
            False: convert segments 1-on-1 and merge segments with identical labels
            True:  merge glottal closures with adjoining plosive first    
    """
    if MERGE_CLOSURES:
        print("ERROR(xlat_seg): Closure Merging not supported Yet")
        exit(-1)
        
    oseg = 0
    iseg = 0
    ww=isegdf.seg
    t0=isegdf.t0
    t1=isegdf.t1
    cnt = len(t0)
    xww = []
    xt0 = []
    xt1 = []
    while iseg < cnt:
        xww.append(xlat_dic[ww[iseg]])
        xt0.append(t0[iseg])
        Merge = False
        if iseg != cnt-1:
            if(xlat_dic[ww[iseg+1]] == xlat_dic[ww[iseg]]):
                Merge = True
        if Merge:
            xt1.append(t1[iseg+1])
            iseg = iseg+2
        else:
            xt1.append(t1[iseg])
            iseg += 1
        oseg +=1
    return(pd.DataFrame({'t0':xt0,'t1':xt1,'seg':xww}))
    

def read_seg_file(fname,downsample=160):
    """
    input: 
        TIMIT style segmentation file consisting of lines
            first_frame   last_frame+1    seg_name
        downsample
            integer downsampling
            default = 160  (i.e. downsampling from 16kHz sample rate to 100Hz frame rate)
    output:
        panda's datastructure with columns [seg,t0,t1]
            seg = segmentation name
            t0  = starting frame (counting starting at 0)
            t1  = last frame +1 (python style ranges)
    """
    ww = []
    t0 = []
    t1 = []
    # read input file without doing conversions
    with open(fname) as fp:
        cnt = 0
        for line in fp:
            w = line.strip().split()
            t0.append(round(int(w[0])/downsample))
            t1.append(round(int(w[1])/downsample))
            ww.append(w[2])
            cnt += 1
    #cnt,len(t0),len(t1),len(ww)
    fp.close()
    df = pd.DataFrame({'t0':t0,'t1':t1,'seg':ww})
    return(df)

def write_seg_file(fname,segdf):
    """
    write a TIMIT style segmentation to file
    """
    nseg = len(segdf)
    fp = open(fname,"w")
    for i in range(nseg):
        fp.write('{:6d} {:6d} {:10s} \n'.format(segdf['t0'][i],segdf['t1'][i],segdf['seg'][i]) )
    fp.close()
 