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