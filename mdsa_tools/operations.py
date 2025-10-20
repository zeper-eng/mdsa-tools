import numpy as np

def edgelist_single_frame(self,topology=None,granularity=None):
        '''Create an upper‑triangle residue–residue edge template for one frame.

        Parameters
        ----------
        topology : mdtraj.Topology or str or pathlib.Path or None, optional
        Topology object or path. If ``None``, uses ``self.topology``.
        granularity : {'residue'}, optional
        Placeholder for future atom‑ or group‑level variants. Only residue‑
        level edges are constructed at the moment.

        Returns
        -------
        np.ndarray of int, shape (E, 2)
        Each row is a pair ``(i, j)`` with ``i < j`` using 0‑based MDTraj
        residue indices. The set corresponds to the upper triangle of an
        ``n_residues × n_residues`` matrix.

        Notes
        -----
        Use :meth:`lookup_table_from_edgelist` to convert ``(i, j)`` pairs to
        contiguous row indices for vectorized time‑series storage.

        '''

        topology = topology if topology is not None else self.topology

        granularity = granularity if granularity is not None else 'residue'

        #Make atom to residue dictionary 

        #Create adjacency matrix, set first row and column as residue indices, and multiply to match the number of frames
        
        row_indexes, column_indexes = np.triu_indices(topology.n_residues, k=1)

        # 1-based residue labels (since original indexing is 1..N)
        edge_table = np.column_stack([row_indexes, column_indexes])   # shape (E, 3)
        
        return edge_table

def lookup_table_from_edgelist(self,edge_list_template=None):
        '''Build a fast ``(i, j) → row`` lookup for the edge template.

        Parameters
        ----------
        edge_list_template : np.ndarray of int or None, shape (E, 2)
        Output of :meth:`edgelist_single_frame`. If ``None``, a template is
        generated from the current topology.

        Returns
        -------
        np.ndarray of int, shape (n_residues, n_residues)
        Dense table ``pair2row`` where ``pair2row[i, j]`` gives the row index
        into the edge list for pair ``(i, j)`` (0‑based). Symmetric with
        diagonal set to ``-1`` as a sentinel for "no mapping".
        '''

        edge_list_template=edge_list_template if edge_list_template is not None else self.edgelist_single_frame()
        #print(edge_list_template.shape)
        #print(edge_list_template)
        #grab residue indexes as int bc we need int
        res1 = edge_list_template[:, 0].astype(np.int32)
        res2 = edge_list_template[:, 1].astype(np.int32)
        idx = np.arange(res1.size)

        #we can now initiate a table of empty -1s and then fill in the row index for pairwise comparisons so we can easily grab row indexes for comparisons
        #it really does not mean much we used -1, just decent convention for missing value, NAN, zeroes etc would be the same but since we are dealing
        #with indexes -1 is a nice simple flag for grabbing thingsok b
        pair2row = -np.ones((self.topology.n_residues+1, self.topology.n_residues+1), dtype=np.int32)
        pair2row[0,0]=0
        #print(self.res_of_interest)
        pair2row[0,1:]=self.res_of_interest
        pair2row[1:,0]=self.res_of_interest

        #pulling out just data
        subset=pair2row[1:,1:]
        subset[res1, res2] = idx
        subset[res2, res1] = idx  # undirected

        #adding it back in
        pair2row[1:,1:]=subset

        
        return pair2row