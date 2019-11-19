# Hidden Markov Models - libhmm
"""
The modules in `libhmm.py` contain a basic implementation of hidden Markov models
The main purpose is didactic demonstrations of (very) small systems 
with applications such as speech recognition in mind. 
Efficiency of the code is not a major concern

Author: Dirk Van Compernolle
Modification History:
12/12/2018 Created
11/02/2019 All list-like objects (states, labels, end_states, ... ) are now uniformly numpy arrays allowing for list indexing
14/11/2019: changed 'backtrace' to 'alignment'

libhmm vs. hmmlearn    
====================
hmmlearn is an existing package that is often referenced
however, it is not really suited as a demonstration package for speech recognition
    - training procedure and model knowledge are coupled in ways
    - hmmlearn is more a clustering than a learning toolkit in which 
            you can not put much prior topological knowledge
in order to make migration easy for users of hmmlearn, we use the same nomenclature and conventions as in hmmlearn
    
libhmm details
==============
    - the classes are all derived from master class _BaseHMM and just describe the model
    - DHMM and GHMM are the most simple implementations for single discrete and continuous densities 
    - a set of training and initializiation procedures are available
        
"""
import sys, os
from math import ceil, log10, pow
import numpy as np
import pandas as pd
from IPython.display import display
# put the spchlib library on the path
# sys.path.append(os.path.join('C:\\users\\compi\\Nextcloud\\Jupyter'))
from . import utils as u

__all__ = ["DHMM","DirectHMM"]
#__all__ = ["DHMM", "CHMM", "GMMHMM",  "MultinomialHMM"]
COVARIANCE_TYPES = frozenset(( "diag", "spherical", "tied"))
HMM_CLASSES = frozenset(("DHMM","DirectHMM"))
PROB_FLOOR = u.__EPS__


# HMM Master Class
class _BaseHMM():
    """ 
    The _BaseHMM() class is the generic class for HMM models
    Subclasses are created for different emission models and feature types
    The base class contains the common framework for the state network
    
       + DirectHMM()   the input features are the observation probabilities
       + DHMM()        is a discrete density with n_features=1 mandatory
     
     Important NOMENCLATURE used throughout:
         - emission probabilities: probabilities in the model
         - observation probabilities: probabilities of the observation stream for the respective states
    
    Attributes common to ALL instantiations of  _BaseHMM()
    ======================================================

    n_states  : int
        Number of states
    n_features: int
        Number of feature streams
    transmat  : array, shape (from_states, to_states)
        Matrix of transition probabilities between states, indexed 
        (from_state,to_state)
    startprob : array, shape (n_states, )
        Initial state occupancy distribution.
        
    hmm_class  : string, any of HMM_CLASSES    
    prob_style : string, "lin", "log" or "log10"  (some backwards compatibility for "prob" and "logprob")
        Do computations with probabilities or log-probabilities
    prob_floor: probability flooring, set by default to PROB_FLOOR
        flooring is used when converting to logprobs
    
    Variables associated with an HMM 
    ================================
    
    X         : array, shape(n_samples, n_features)
        Feature vector of length n_samples

    trellis   : array, shape(n_samples, n_states)
        Trellis containing cummulative probs up to (i_sample, j_state)
        The same arrangement applies to a number of other variables: posteriors, backpointers, ...


    Methods 
    =======
    
    init_topology()
        initializes the state diagram
        
    set_probstyle()
        allows for choice between prob's and logprob's for computations
        
    compute_frameprobs()
        compute the observation probabilities for a given stream of features
    
    viterbi_trellis()
        compute a full trellis using the Viterbi algorithm
    
    viterbi_step()
        compute a single time step in a Trellis based Viterbi recursion
        
    print_state_model()
        pretty prints HMM init/transition model parameters
        
    print_model()
        pretty prints HMM transition and observation model parameters

    """

    def __init__(self,n_states=1,transmat=None,startprob=None,states=None,prob_style="lin",prob_floor=PROB_FLOOR):

        self.hmm_class = ""
        self._Debug = False
        # map old-fashioned synonyms  logprob -> log and prob -> lin
        if prob_style == "logprob": prob_style = "log"
        elif prob_style == "prob":  prob_style = "lin"
        
        self.prob_style = prob_style
        self.prob_floor = prob_floor
            
        # initialize everything at the state level
        self.n_states = n_states
        if (states is None):
            self.states = np.array(['S'+str(i) for i in range(0,self.n_states)],dtype='str')
        else:
            self.states = states
        self.end_states = np.array([i for i in range(0,self.n_states)],dtype='int')
        if (transmat is None):
            self.transmat = np.eye(self.n_states)
            if(self.prob_style == "log"):
                self.transmat = u.logf(self.transmat,eps=self.prob_floor)
            elif(self.prob_style == "log10"):
                self.transmat = u.log10f(self.transmat,eps=self.prob_floor)
        else:
            self.transmat = transmat

        if (startprob is None):
            self.startprob = np.zeros(self.n_states)
            self.startprob[0] = 1.0
            if(self.prob_style == "log"):
                self.startprob = u.logf(self.startprob,eps=self.prob_floor)
            elif(self.prob_style == "log10"):
                self.startprob = u.log10f(self.startprob,eps=self.prob_floor)
        else:
            self.startprob = startprob   


    def set_probstyle(self,prob_style):
        
        if prob_style == "logprob": prob_style = "log"
        elif prob_style == "prob":  prob_style = "lin"
        if(self.prob_style == prob_style):  return
        
        self.transmat = u.convertf(self.transmat,iscale=self.prob_style,oscale=prob_style)
        self.startprob = u.convertf(self.startprob,iscale=self.prob_style,oscale=prob_style)
        self.set_obs_probstyle(prob_style)       
        self.prob_style = prob_style
           
    def set_obs_probstyle(self,prob_style):
        return
    
    def init_topology(self,type="lr",selfprob=0.5):
        if(type == "lr"):
            self.end_states = np.array([ self.n_states-1 ],'int')
            self.transmat = np.eye(self.n_states)
            for j in range(0,self.n_states-1):
                self.transmat[j,j]=selfprob    
                self.transmat[j,j+1]=1.0-selfprob
        if(self.prob_style == "logprob"):
            self.transmat = np.log(u.floor(self.transmat,self.prob_floor))
        
      
        
    def print_state_model(self):
        """ 
        print_model(self) 
        is a pretty print routine showing INITIAL and TRANSITION PROBABILITY matrices in an HMM
        """
        tostates = np.array(['to-'+self.states[i] for i in range(0,self.n_states)],dtype='str')
        dfstates = pd.DataFrame(self.transmat,index=self.states,columns=tostates)
        dfinit = pd.DataFrame(self.startprob,index=self.states,columns=["INIT"])
        print("INITIAL and TRANSITION PROBABILITIES (from,to)")
        display(pd.concat([dfinit,dfstates],axis=1))  
         
                
    def print_observation_model(self):
        """
        Prints observation model parameters
        """

    def print_model(self):
        self.print_state_model()
        self.print_observation_model()
  
        
    def compute_frameprobs(self,X):
        """Computes per-component log probability under the model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        Returns
        -------
        prob : array, shape (n_samples, n_states)
            probability of each sample in ``X`` for each of the model states.
        """

        
    def viterbi_step(self, prev_buffer, frameprob):
        """ Viterbi Processing over one single frame
        """
        if(self.prob_style == "log" or self.prob_style == "log10"):
            buffer = u.logf(np.zeros(prev_buffer.shape))
            backptr = np.zeros(self.n_states)-1
            for to_state in range(0,self.n_states):
                for from_state in range(0,self.n_states):
                    new = self.transmat[from_state,to_state] + prev_buffer[from_state]
                    if( new > buffer[to_state] ):
                        buffer[to_state] = new
                        backptr[to_state] = from_state
                buffer[to_state] = buffer[to_state] + frameprob[to_state]
        else:
            buffer = np.zeros(prev_buffer.shape)
            backptr = np.zeros(self.n_states)-1
            for to_state in range(0,self.n_states):
                for from_state in range(0,self.n_states):
                    new = self.transmat[from_state,to_state] * prev_buffer[from_state]
                    if( new > buffer[to_state] ):
                        buffer[to_state] = new
                        backptr[to_state] = from_state
                buffer[to_state] = buffer[to_state] * frameprob[to_state]
        return(buffer,backptr)
        
    
    def viterbi_trellis(self,X,col_norm=False):
        """Compute full trellis with Viterbi Algorithm
        col_norm when True normalizes column by column in the trellis 
            to keep values close to 1.0

            - when probstyle = lin: values are normalized with best power of 10
            - when probstyle = log or log10: max value is set to 0.0
            ++ returns an extra vector 'scale_vec'
        """

        # compute frameprobs
        n_samples = len(X)
        if self.prob_style == "log" or self.prob_style == "log10":
            scale_vec = np.zeros(n_samples)
        else:
            scale_vec = np.ones(n_samples)
        frameprobs = self.compute_frameprobs(X)

        # assign working arrays for trellis and backptrs
        trellis = np.ndarray(frameprobs.shape)
        backptrs = np.zeros(trellis.shape,dtype='int')-1
        
        # initialize at frame 0
        if(self.prob_style == "log" or self.prob_style=="log10"):
            trellis[0,:] = self.startprob + frameprobs[0,:]
        else:
            trellis[0,:] = self.startprob * frameprobs[0,:]
        backptrs[0,:] = np.arange(0,self.n_states)

        # forward pass
        for i in range(1,n_samples):
            (trellis[i,:], backptrs[i,:]) = self.viterbi_step(trellis[i-1,:],frameprobs[i,:])
            if col_norm :
                max_col = np.max(trellis[i,:])
                if(self.prob_style == "log" or self.prob_style == "log10"):
                    trellis[i,:] = trellis[i,:] - max_col
                    scale_vec[i] = scale_vec[i-1] + max_col
                else:
                    sc = pow(10,ceil(log10(max_col)))
                    trellis[i,:] = trellis[i,:] / sc
                    scale_vec[i] = scale_vec[i-1] * sc                    

        # backtracking    
        alignment = np.ndarray(n_samples,'int')
        # determine best admissible endstate
        endprobs = trellis[n_samples-1,:]
        best_end_state = self.end_states[np.argmax( endprobs[self.end_states] )]
        alignment[n_samples-1] = best_end_state
        for i in range(n_samples-1,0,-1):
            alignment[i-1]=backptrs[i,alignment[i]]

        if col_norm:
            return frameprobs, trellis, backptrs, alignment, scale_vec 
        else:
            return frameprobs, trellis, backptrs, alignment
        
    
class DirectHMM(_BaseHMM):
    """Direct Hidden Markov Model,
        i.e. observations are used as the emission probabilities 
        
    Parameters
    ----------

    n_features:  : number of features 

    topology related parameters have their default definitions and initializations:
        n_states, transmat, startprob 
    
    """       
    def __init__(self,n_states=1,n_features=2,transmat=None,startprob=None,states=None,prob_style="lin",prob_floor=PROB_FLOOR):
        _BaseHMM.__init__(self,n_states=n_states,transmat=transmat,startprob=startprob,states=states,prob_style=prob_style,prob_floor=prob_floor)
        self.hmm_class = "DirectHMM"  

    def print_observation_model(self): 
        print("OBSERVATION MODEL: Direct Input of Probability Stream\n")  
        
    def compute_frameprobs(self,X):
        return X
    
        
class DHMM(_BaseHMM):
    """Discrete Density Hidden Markov Model,
        i.e. with a single multinomial (discrete) emission
        
    Parameters
    ----------

    n_features:  : number of features (right now this should be 1 !)

    n_symbols    : array(n_features) 
        Number of possible symbols emitted by the model (in the samples).
        The symbol set for feature ii is [0, 1, ... ,n_symbols(ii)-1]
 
    emissionprob : array, shape (n_states, n_symbols)
        Probability of emitting a given symbol when in each state.

    labels:  array, shape(n_symbols)      ... sometimes a list would be handy instead

    topology related parameters have their default definitions and initializations:
        n_states, transmat, startprob 
    
    """


    def __init__(self,n_states=1,transmat=None,startprob=None,states=None,prob_style="lin",prob_floor=PROB_FLOOR,n_features=1,emissionprob=None,n_symbols=None,labels=None):
        _BaseHMM.__init__(self,n_states=n_states,transmat=transmat,startprob=startprob,states=states,prob_style=prob_style,prob_floor=prob_floor)

        self.hmm_class = "DHMM"
        # initialize everything at the feature/emission level
        self.n_features = n_features
        if (n_symbols is None):
            self.n_symbols = 2*np.ones(n_features,int)
        else:
            self.n_symbols = n_symbols
            
        # needs further adjustment for multi-stream input !!
        if self.n_features > 1:
            print("only supporting single stream ...")
            exit(1)
        if (labels is None):
            self.labels = np.array(['L'+str(i) for i in range(0,self.n_symbols)])
        else: 
            self.labels = np.array(labels)
            
        if (emissionprob is None) :
            self.emissionprob =   np.zeros((self.n_states,self.n_symbols),dtype=float)
            for j in range(0,n_states):
                for k in range(0,self.n_symbols):
                    self.emissionprob[j,k]= 1.0/float(self.n_symbols)
            if(self.prob_style == "logprob"):
                self.emissionprob = np.log(u.floor(self.emissionprob,self.prob_floor))
        else:
            self.emissionprob = emissionprob

    def set_obs_probstyle(self,prob_style):  
        # WARNING: set_obs_probstyle does not update self.prob_style !!
        if prob_style == "logprob": prob_style = "log"
        elif prob_style == "prob":  prob_style = "lin"
        if(self.prob_style == prob_style):  return
        self.emissionprob = u.convertf(self.emissionprob,iscale=self.prob_style,oscale=prob_style,eps=self.prob_floor)        

       
    def compute_frameprobs(self,X):
        return self.emissionprob[:, X].T

    def print_observation_model(self,labels=None): 
        """ 
        print_observation_model(self) 
        is a pretty print routine showing INITIAL and TRANSITION PROBABILITY matrices in an HMM
        """
        if (labels is None):
            columns = self.labels
        else:
            columns = labels
        print("OBSERVATION PROBABITIES for DISCRETE DENSITY MODEL")
        display(pd.DataFrame(self.emissionprob,index=self.states,columns=columns))                                  
 
class Trellis():
        def __init__(self,n_samples=1,n_states=1,prob_style="lin"):
            """
            Create a trellis of size(n_samples, n_states) 
            
            Attributes of a trellis are:
                prob_style      probability style     lin, log, log10
                frameprobs      frame probabilities   float(n_samples,n_states)
                probs           trellis cell values   float(n_samples,n_states)
                cellcompute     cell computations     float(n_samples,n_states,n_states)
                backptrs        back pointers         int(n_samples,n_states)
                alignment       Viterbi Alignment     int(n_samples)
                scale_vec       suggested scaling value per sample        float(n_samples)
                end_state       best admissible end state
                end_prob        cumulated prob in end_state
                method          method applied        None, Viterbi, Forward
            """               
                
            self.prob_style = prob_style;
            self.n_samples = n_samples
            self.n_states = n_states
            self.reset()
            self.method = "None"
            self._Debug = False
            
        def reset(self):
            siz = (self.n_samples,self.n_states)
            self.frameprobs = np.zeros(siz,dtype='float')
            self.probs = np.zeros(siz,dtype='float')
            self.backptrs = np.zeros(siz,dtype='int') - 1
            self.alignment = np.zeros(self.n_samples,dtype='int') - 1 
            if self.prob_style == "lin":
                self.scale_vec = np.ones(self.n_samples,dtype='float')
            else:
                self.scale_vec = np.zeros(self.n_samples,dtype='float')
                
        def viterbi_pass(self,X,hmm):

            self.method = "Viterbi"
            self.frameprobs = hmm.compute_frameprobs(X)
            
            if (self._Debug):
                print('Shape of Frameprobs:',self.frameprobs.shape)
            # initialize at frame 0
            if(self.prob_style == "lin"):
                self.probs[0,:] = hmm.startprob * self.frameprobs[0,:]
            else:
                self.probs[0,:] = hmm.startprob + self.frameprobs[0,:]
                
            self.backptrs[0,:] = np.arange(0,self.n_states)
    
            # forward pass
            for i in range(1,self.n_samples):
                (self.probs[i,:], self.backptrs[i,:]) = hmm.viterbi_step(self.probs[i-1,:],self.frameprobs[i,:])
                max_col = np.max(self.probs[i,:])
                if(self.prob_style == "lin"):
                    self.scale_vec[i] = pow(10,ceil(log10(max_col)))
                else:
                    self.scale_vec[i] = max_col
                  
            # terminate
            endprobs = self.probs[self.n_samples-1,:]
            self.end_state = hmm.end_states[np.argmax(endprobs[hmm.end_states])]
            self.end_prob = self.probs[self.n_samples-1,self.end_state]
            
            # Find alignment via backtracking
            self.alignment = self.backtrack(endstate=self.end_state)
            
            #self.alignment[self.n_samples-1] = self.end_state
            #for i in range(self.n_samples-1,0,-1):
            #    self.alignment[i-1]=self.backptrs[i,self.alignment[i]]
            

        def backtrack(self,endstate=None):
            ''' Compute the backtracking from a Viterbi Trellis
            
            Parameters:            
                endstate: state to backtrack from 
                            (default: last state in trellis)

            Return:
                the alignment     (n_samples,) * int
                
            '''
            
            if( endstates == None ): endstate = self.n_states-1
            
            # Find alignment via backtracking
            alignment[self.n_samples-1] = end_state
            for i in range(self.n_samples-1,0,-1):
                alignment[i-1]=self.backptrs[i,alignment[i]]
            return alignment
            
            
        def print(self,Xlabels=[],Slabels=[],col_norm=False):
            S_index = ['S'+str(i) for i in range(0,self.n_states)]
            T_index = [i for i in range(0,self.n_samples)]
            X_index = ['X'+str(i) for i in range(0,self.n_samples)]

            if not list(Xlabels):
                Xlabels = X_index
            if not list(Slabels):
                Slabels = S_index
        
            print("FRAME PROBABILIITIES and TRELLIS")
            display(pd.DataFrame(self.frameprobs.T,index=Slabels,columns=Xlabels))
            
            dftrellis = pd.DataFrame(self.probs.T,index=Slabels,columns=Xlabels)
            if col_norm:
                dfscale = pd.DataFrame(self.scale_vec.reshape(1,self.n_samples),index=["SCALE"],columns=Xlabels)
                display(pd.concat([dftrellis,dfscale]))
            else:
                display(dftrellis)
        #        display(HTML(dftrellis.to_html(float_format=float_format)))
            
            print("BACKPOINTERS")
            df1 = pd.DataFrame(self.backptrs.T,index=Slabels,columns=Xlabels)
            display(df1)
            print("ALIGNMENT")
            bt = [Slabels[self.alignment[ii]] for ii in range(0,len(self.alignment))]
            df2 = pd.DataFrame([list(Xlabels), list(bt)],index=["X","BT"],columns=T_index)
            display(df2)