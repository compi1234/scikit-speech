import os, sys
import numpy as np
import pandas as pd
import scipy.io as sio

# General Purpose open/read/write routines for 
# a number of common filetypes I use in speech processing
#
 
def read(fname,filetype='RAW',datatype='LIST',encoding='latin1',dtype=[],args={},header=False):
    ''' reads data till the end of file for a number of different file and datatypes
        input arguments:
            fname           file name (string)
            encoding        latin1 (default), utf8, ..            
            filetype        RAW (default), SPRAAK, MATLAB
            datatype        LIST (default), DICT, WAV, FRAME, SPRSEG, SEG
            dtype           string, float32, int32, segdf, ...   (numpy dtype)
            args            dictionary for passing on optional arguments
            header          TRUE, FALSE   (returns data+header in tuple when true)
        return values:
            data (,hdr)     the data and optionally the hdr (when header=TRUE is specified)
            '''
      
    if not os.path.isfile(fname):
       print("File path {} does not exist. Exiting...".format(fname))
       sys.exit()
       
       
    hdr = {}
    if( filetype == 'MATLAB' ):
        data = sio.loadmat(fname,squeeze_me=True)
        return data
    elif( filetype == 'SPRAAK'):
        fp = open(fname,'r',encoding=encoding)        
        fp, hdr = read_spr_hdr(fp)
    else:
        fp = open(fname,'r',encoding=encoding)
  
    if( 'datatype' in hdr.keys() ): datatype = hdr['datatype'] 
    if( 'dtype'  in hdr.keys() ): dtype = hdr['dtype'] 

    if( datatype =='FRAME' ):
        data = np.fromfile(fp,dtype=dtype,count=-1)
        if( 'DIM1' in hdr.keys() and 'DIM2' in hdr.keys() ):
            nfr = int(hdr['DIM1'])
            nparam = int(hdr['DIM2'])            
            data = np.reshape(data,(nfr,nparam))
    elif( datatype == 'SPRSEG'):
        data = read_spr_segdata(fp,hdr,args)
    elif( datatype == 'SEG'):
        data = read_segdata(fp,args)
    elif( datatype == 'DICT'):
        data = read_dict(fp,args)                
    else:
        data = fp.read()
        
        
    if header:
        return data, hdr
    else:
        return data

def decode_args(args,defaults):
    '''
    decoding of optional / default arguments
        defaults: is a dict containing the values of parameter defaults 
        args:     is a dict containing the optional arguments  (possibly more than applicable here)
    returns
        params:  is a dict with locally relevant parameters set to the correct default / optional values
    '''
        
    params = defaults
    for key in args.keys():
        if key in params.keys():  params[key] = args[key]
    return(params)

def read_dict(fp,args={}):
    '''
    Reads a datafile directly into a dictionary structure, with params:
        ckey = column containing the keys (default = 0)
        cvalue = column containing the values (default = 1)
        maxsplit = max number of splits (default = -1)
    
    '''
    params = decode_args(args,{'ckey':0,'cvalue':1,'maxsplit':-1})

    dic = {}
    for line in fp:
        w = line.strip().split(maxsplit=params['maxsplit'])
        dic[w[params['ckey']]] = w[params['cvalue']]
    return(dic)

def read_spr_hdr(fp):
    '''
    Reads the header of a SPRAAK file with .spr, .key or ASCII header
    i.e. explicitly assumes that the file has header data till a line starting with "#" 
    and that data after that
    
    The header consists with multiple lines consisting of  KEY VALUE pairs where
            the first word is the KEY and the REMAINDER the VALUE

    returns
        fp:   file pointer at the beginning of the data section
        hdr:  header as a Python dictionary
    '''
    hdr = {}
    first_time = True
    while(1):
        line = fp.readline()
        # print("reading: ",line)
        line = line.strip()
        # determine the header type of the file
        if ( first_time ):
            first_time = False
            if( line != ".key"   and  line != ".spr"):
                # assuming ascii header if neither .key or .spr found
                hdr['.ascii'] = None
        # continue reading header KEY VALUE pairs till EOH is detected
        if len(line) == 0: 
            continue
        elif line[0]=="#": 
            break
        else:
            w = line.split(None,1)
            if len(w) == 1: hdr[w[0]] = None
            else: hdr[w[0]] = w[1]    
    # print("last line in loop: ",line)            
    # convert and overwrite certain header keys
    if 'DATA' in hdr.keys():
        if hdr['DATA'] == 'TRACK': hdr['datatype'] = 'FRAME'
        if hdr['DATA'] == 'SEG':   hdr['datatype'] = 'SPRSEG'
    if 'TYPE' in hdr.keys():
        if hdr['TYPE'] == 'F32': hdr['dtype'] = 'float32'
        if hdr['TYPE'] == 'INT': hdr['dtype'] = 'int32'
        
    return fp, hdr

def read_segdata(fp,args={}):
    """
    input: 
        segmentation file consisting of lines
            first_frame   last_frame+1    seg_name
        params (dict):
            segtype    = CONTINOUS, DISCRETE (default)
            frameshift = 1 (i.e. frame based segmentation, can be used for downsampling from sample segs )
            col_t0     = column containing t0 (default=0)
            col_t1     = column containing t1 (default=1)
            col_seg     = column containing seg (default=2)
    output:
        panda's datastructure with columns [seg,t0,t1]
            seg = segment name
            t0  = starting frame (counting starting at 0)
            t1  = last frame +1 (python style ranges)
    """

    params = decode_args(args,{'frameshift':1,'col_t0':0,'col_t1':1,'col_seg':2})
    
    ww = []
    t0 = []
    t1 = []
    # read input file without doing conversions
    cnt = 0
    for line in fp:
        w = line.strip().split()
        t0.append(round(int(w[params['col_t0']])/params['frameshift']))
        t1.append(round(int(w[params['col_t1']])/params['frameshift']))
        ww.append(w[params['col_seg']])
        cnt += 1
    #cnt,len(t0),len(t1),len(ww)
    fp.close()
    df = pd.DataFrame({'t0':t0,'t1':t1,'seg':ww})
    return(df)


def read_spr_segdata(fp,hdr,params={}):
    """
    input: 
        fp points to SPRAAK style segmentation file consisting of lines
            entry_name    seg_name    begin    end/nfr
                    entry_name   is a file reference or a '-' as continuation sign from previous line
                    seg_name     is a word/phone/state reference
                    begin        begin_time for CONTINOUS,  first_frame for DISCRETE (counting from 0)
                    end/nfr      end_time for CONTINOUS, n_frames for DISCRETE
        params:
            frameshift    continous time frameshift to be applied for converting continous to discrete (default=0.01)
    output:
        a dictionary with   keys=entry_names   values=segmenations as panda datastructure
        panda datastructures have columns [seg,t0,t1]
            seg = segment name
            t0  = starting frame (counting starting at 0)
            t1  = last frame +1 (python style ranges)
    """

    C2D = False
    if( 'TIMEBASE' in hdr.keys() ):
        if ( hdr['TIMEBASE'] == "CONTINUOUS" ):
            if ('frameshift' in params.keys() ):   frameshift = params['frameshift']
            else: frameshift = 0.01
            C2D = True
                
    First_time = True
    
    segdata = {}
    segname = ""
    ww = []
    t0 = []
    t1 = []
    cnt = 0
    for line in fp:
        w = line.strip().split()
        if w[0] != "-": 
            if First_time: First_time = False
            else: segdata[segname] = pd.DataFrame({'t0':t0,'t1':t1,'seg':ww})
            ww = []
            t0 = []
            t1 = []
            cnt = 0
            segname=w[0]

        # process segmentation
        if C2D:
            i0 = round(float(w[2])/frameshift)
            i1 = round(float(w[3])/frameshift)
            t0.append(i0)
            t1.append(i1)
        else:
            i0 = int(w[2])
            i1 = int(w[3])
            t0.append(i0)
            t1.append(i0+i1)
        ww.append(w[1])
        cnt+=1
    
    # still need to end last entry to output                                            
    segdata[segname] = pd.DataFrame({'t0':t0,'t1':t1,'seg':ww}) 
    
    return(segdata)
    
def write_timit_seg_file(fname,segdf):
    """
    write a TIMIT style segmentation to file
    """
    nseg = len(segdf)
    fp = open(fname,"w")
    for i in range(nseg):
        fp.write('{:6d} {:6d} {:10s} \n'.format(segdf['t0'][i],segdf['t1'][i],segdf['seg'][i]) )
    fp.close()
 
def xlat_seg(isegdf,xlat_dic,MERGE_CLOSURES=False):
    """
    convert alphabets between input segmentation and output segmentations
    optionally merge identical labels
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