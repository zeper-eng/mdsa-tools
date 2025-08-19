import numpy as np
import os

class MSM_Modeller():

    def __init__(self,candidate_states):
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
        self.candidate_states=candidate_states
        self.transition_probability_matrix=None
        self.reduced_coordinates=None
        self.lag=None

    def rmsd_from_centers(self, X, labels, centers):
        results = []
        for k in np.unique(labels):
            m = (labels == k)  # mask frames belonging to cluster k
            d = np.linalg.norm(X[m] - centers[int(k)], axis=1)
            rmsd = float(np.sqrt(np.mean(d**2)))
            results.append((int(k), rmsd))
        return results

    def evaluate_cohesion(candidate_states,reduced_coordinates):
        '''evaluate whether trajectories are temporally settling into the candidate states
        
        candidatestates=arraylike,default=mdsa_tools.Analysis.cluster_embeddingspace(),shape=(number_of_systems_)
            A list of arrays holding, each array in every system contains the cluster assignments and labels returned
            from the system analysis module's preform_clust_opt() operation.
        
        '''
        
        savepath=savepath if savepath is not None else os.getcwd()
        frames_per_sys=np.array_split(reduced_coordinates,len(candidate_states))

        for i in range(len(candidate_states)):
            labels,_ = candidate_states[i][0],candidate_states[i][1]


    
        return

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