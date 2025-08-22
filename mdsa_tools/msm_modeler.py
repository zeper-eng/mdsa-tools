import numpy as np
import os
import pandas as pd

class MSM_Modeller():

    def __init__(self,labels,centers,reduced_coordinates,frame_scale):
        '''A module for evaluating and modelling the candidate states and subsequent MSM of an emebddingspace.

        
        Parameters
        ----------


        Returns
        -------


        Notes
        -----


        Examples
        --------

        '''
        self.labels=labels if labels is not None else None 
        self.centers=centers if centers is not None else None 
        self.frame_scale=frame_scale if frame_scale is not None else None 
        self.reduced_coordinates=reduced_coordinates if reduced_coordinates is not None else None

    def rmsd_from_centers(self, X, labels, centers):
        results = []
        for k in np.unique(labels):
            m = (labels == k)  # mask frames belonging to cluster k
            d = np.linalg.norm(X[m] - centers[int(k)], axis=1)
            rmsd = float(np.sqrt(np.mean(d**2)))
            results.append((int(k), rmsd))
        results=np.array(results)
        return results

    def evaluate_cohesion_slidingwindow(self,labels=None,centers=None,reduced_coordinates=None,frame_scale=None,step_size=None):
        '''evaluate whether trajectories are temporally settling into the candidate states
        
        Paramters
        ---------
        candidatestates=arraylike,default=mdsa_tools.Analysis.cluster_embeddingspace(),shape=(number_of_systems_)
            A list of arrays holding, each array in every system contains the cluster assignments and labels returned
            from the system analysis module's preform_clust_opt() operation.
        
        reduced coordinates =arraylike,shape=(n_samples,2)
            The results of either Principal Components Analysis or UMAP reduction to 2 new dimensions.
        
        frame_scale:list of int, optional
            A list holding integer counts of the number of frames in each replicate. 
            Default is (([80] * 20) + ([160] * 10)) * 2.
        
        
            
        
        Returns
        -------



        Notes
        -----




        Examples
        --------



        
        '''
        reduced_coordinates=reduced_coordinates if reduced_coordinates is not None else self.reduced_coordinates
        frame_scale=frame_scale if frame_scale is not None else self.frame_scale
        step_size = step_size if step_size is not None else 10
        labels = labels if labels is not None else self.labels        
        centers = centers if centers is not None else self.centers


        slidingwindow=0
        window_df_all=[]
        for j in range(1,(np.max(frame_scale)//step_size)+1):
            print(f"shrink: {j}")

            mask=[]

            #iterate through reps and make mask
            for rep_length in frame_scale:
                
                if slidingwindow>rep_length:
                    replicate_bools = np.full(rep_length,False)
                    mask.append(replicate_bools)
                    continue

                replicate_bools = np.full(rep_length,False)
                replicate_bools[slidingwindow:slidingwindow+step_size]=True
                mask.append(replicate_bools)
            

            slidingwindow+=step_size#increase creep

            #apply mask save current window as a pd 
            window_mask=np.concatenate(mask)
            window_labels=labels[window_mask]
            window_coordinates=reduced_coordinates[window_mask,:]

            rmsd_results = self.rmsd_from_centers(window_coordinates,window_labels,centers)
            windowdf=pd.DataFrame(rmsd_results,columns=('cluster','rmsd'))
            windowdf['window'] = j
            
            window_df_all.append(windowdf)

                
        #concatenate pd and return
        window_df_all=pd.concat(window_df_all)
        

        
        return window_df_all

    def evaluate_cohesion_shrinkingwindow(self,labels=None,centers=None,reduced_coordinates=None,frame_scale=None,step_size=None):
        '''shrinking window version of slidingwindow
        '''
        reduced_coordinates=reduced_coordinates if reduced_coordinates is not None else self.reduced_coordinates
        frame_scale=frame_scale if frame_scale is not None else self.frame_scale
        step_size = step_size if step_size is not None else 10

        labels = labels if labels is not None else self.labels        
        centers = centers if centers is not None else self.centers

        creepingstart=0
        window_df_all=[]
        for j in range(1,(np.max(frame_scale)//step_size)+1):
            print(f"shrink: {j}")

            mask=[]

            #iterate through reps and make mask
            for rep_length in frame_scale:
                if creepingstart>rep_length:
                    replicate_bools = np.full(rep_length,False)
                    mask.append(replicate_bools)
                    continue
                
                replicate_bools = np.full(rep_length,True)
                replicate_bools[0:creepingstart]=False
                mask.append(replicate_bools)


            #apply mask save currenti window as a pd 
            window_mask=np.concatenate(mask)

            window_labels=labels[window_mask]
            window_coordinates=reduced_coordinates[window_mask,:]

            rmsd_results = self.rmsd_from_centers(window_coordinates,window_labels,centers)
            windowdf=pd.DataFrame(rmsd_results,columns=('cluster','rmsd'))
            windowdf['window'] = j
            
            creepingstart+=step_size#oincrease creep

            window_df_all.append(windowdf)

            
        #concatenate pd and return
        window_df_all=pd.concat(window_df_all)
        print(window_df_all)
   
        return window_df_all



    def create_transition_probability_matrix(self,labels=None,frame_list=None,lag=None):
        '''Create probability matrix from input data (returns, and updates class attribute)

        Parameters
        ----------
        labels:arraylike,shape=(n_labels,)
            A list of labels pertaining to frames of molecular dynamics trajectories assigned particular substates

        frame_list: listlike,shape=(data,)
            A list of integers representing the number of frames present in each replicate. This should be in the order
            of which the various versions of the system, and replicates where concatenated. 

        
        Returns
        -------
        transition_probability_matrix:arraylike;shape=(n_states+1,n_states+1)
            A transition probability matrix created from the list of labels. Diagonals indicate
            if it is likely to stay in the same state and off diagonals mark probabilities of transitions



        
        Notes
        -----
        Much in the spirit of our original matrices the first row and column of theese matrices contain
        indexes mainly for ease of use and manipulation. Yes, in theory pandas dataframes could streamline this process
        but, numpy arrays are just that much more efficient in most use cases,



        Examples
        --------

        

        '''


        labels=labels if labels is not None else self.labels
        frame_list=frame_list if frame_list is not None else self.frame_list
        lag=lag if lag is not None else 1

        #extract unique states and initiate transiiton probability matrix
        unique_states=np.unique(labels)
        number_of_states=len(unique_states)
        transtion_prob_matrix=np.zeros(shape=(number_of_states,number_of_states))
        
        iterator=0
        for trajectory_length in frame_list: # iterate through 
            current_trajectory=labels[iterator:iterator+trajectory_length]
            iterator=iterator+trajectory_length #update this 

            for i in range(current_trajectory.shape[0]-lag):
                current_state=current_trajectory[i]
                next_state = current_trajectory[i+lag]
                transtion_prob_matrix[current_state, next_state] += 1

        row_sums = transtion_prob_matrix.sum(axis=1, keepdims=True)

        transition_probs = np.divide(
                    transtion_prob_matrix, row_sums,
                    out=np.zeros_like(transtion_prob_matrix), #because we dont want to divide by zero!
                    where=row_sums>0
                )

        final_transition_prob_matrix=np.zeros(shape=(number_of_states+1,number_of_states+1))
        final_transition_prob_matrix[1:,1:]=transition_probs
        final_transition_prob_matrix[0,1:],final_transition_prob_matrix[1:,0]=unique_states,unique_states

        self.transition_probability_matrix=final_transition_prob_matrix

        return final_transition_prob_matrix
    
    def evaluate_Chapman_Kolmogorov(self,transition_probability_matrix=None,n=None,labels=None,original_lag=None):
        '''evaluate if the chapman kolmogorov test evaluates to true

        Parameters
        ----------
        n:int,default=4
            The original number of lags we used to compute the transition probability matrix
        
        transition_proability_matrix:arraylike,shape=(n_states+1,n_states+1),

        n:int,default=4
            The time lag we are using to compute our labels

        labels:arraylike,default=self.labels
            The list of labels we are using for the labeling of data from trajectories. 
        
        original_lag:int:default=1


        Notes
        -----

        
        Returns
        -------



        Examples
        --------
        
        
        '''

        transition_probability_matrix=transition_probability_matrix if transition_probability_matrix is not None else self.create_transition_probability_matrix()
        original_lag=original_lag if original_lag is not None else 1
        n = n if n is not None else 4
        labels=labels if labels is not None else self.labels

        transition_prob_data=transition_probability_matrix[1:,1:]
        post_timestep_data=np.linalg.matrix_power(transition_prob_data,n)
        transition_probability_matrix[1:,1:]=post_timestep_data

        total_lag=original_lag*n
        matrix_from_total_lag = self.create_transition_probability_matrix(lag=total_lag)
        diff=matrix_from_total_lag[1:,1:]-transition_probability_matrix[1:,1:]
        frob = np.linalg.norm(diff, ord='fro')

        return frob