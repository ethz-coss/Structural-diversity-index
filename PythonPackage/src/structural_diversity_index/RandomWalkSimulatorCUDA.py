import cupy as cp  # noqa
import cupyx.scipy.sparse as sparse  # noqa
import numpy as np
import scipy.sparse
from tqdm import tqdm
import torch
from torch.utils.dlpack import to_dlpack
from typing import Tuple


class RandomWalkSimulator:
    """

    The class RandomWalkSimulator is designed to run a fast simulations of a random walk on a graph
    and compute the meeting times of two walks

    """

    def __init__(self, random_walk_matrix: sparse.csr_matrix):
        """

        Initialises a RandomWalkSimulator

        Args:
            random_walk_matrix (cupyx.scipy.sparse.csr_matrix): the random walk transition matrix of the graph on which you want to simulate the random walk

        """

        # Device name
        self.n_nodes = random_walk_matrix.shape[0]

        # Random walk matrix
        self.P = random_walk_matrix

    ###################################################################### PUBLIC METHODS ###############################################################################

    def get_meeting_times(self, max_time_steps: int, n_samples: int) -> np.ndarray:
        """

        Gets the meeting times necessary to compute delta_g, i.e., it computes n_samples of the meeting time of two randomly started walks.

        Args:
            max_time_steps (int): the number of time steps for which you want to simulate the random walks at most
            n_samples (int): the number of samples of the meeting time that you want

        Returns:
            (1D np.ndarray): a 1D np.ndarray in which each entry is one sample of the meeting time of two randomly started walks. If the walks met after max_time_steps the entry is equal to -1 by default.

        """

        start_position = cp.random.randint(low=0, high=self.n_nodes, size=[n_samples])
        meeting_times = self._get_meeting_times(max_time_steps=max_time_steps, start_position=start_position)

        meeting_times_flat = np.ndarray.flatten(meeting_times)
        meeting_times_flat_without_diagonal = np.delete(meeting_times_flat,
                                                        range(0, len(meeting_times_flat), len(meeting_times_flat) + 1),
                                                        0)

        return meeting_times_flat_without_diagonal

    @staticmethod
    def random_walk_matrix_from_adjacency(adjacency: scipy.sparse.csr_matrix, degrees: np.ndarray) -> sparse.csr_matrix:
        """

        Returns the transition matrix of the random walk

        Args:
            adjacency (scipy.sparse.csr_matrix): the adjacency matrix of the graph for which we want to build the transition matrix
            degrees (np.array): the degrees of the nodes in the graph

        Returns:
            (cupy.scipy.sparse.csr_matrix):  the transition matrix of the random walk. This is defined as P(i,j) = 1/deg(i)

        """

        # Build the transition matrix
        A = sparse.csr_matrix(adjacency)

        one_over_degrees = cp.array(1 / degrees)
        ind = cp.arange(one_over_degrees.shape[0])
        D = sparse.csr_matrix((one_over_degrees, (ind, ind)), shape=(ind.shape[0], ind.shape[0]))

        P = D * A
        return P

    ###################################################################### PRIVATE METHODS ###############################################################################

    def _get_meeting_times(self, max_time_steps: int, start_position: cp.ndarray) -> cp.ndarray:
        """

        This method simulates random walks started at start_position and keeps track of their meeting times.
        Then it returns these meeting times, once all the walks have met or max_time_steps has expired.

        Args:
            max_time_steps (int): the number of time steps for which you want to simulate the random walks at most
            start_position (1D cp.ndarray, optional): a 1D cp.ndarray in which each entry is the starting positions of one sample of the random walk.

        Returns:
            (2D cp.ndarray): a 2D cp.ndarray with entry m,n being the meeting time between sample walk m and sample walk n. \
            If two walks never meet the value of the m,n entry is -1

        """

        n_samples = len(start_position)

        meeting_times = -1 * cp.ones([n_samples, n_samples])

        # Fix the starting position and check if walks meet at the starting position
        start_pos, array_start = self._start_position(n_samples=n_samples, start_position=start_position)
        meeting_bool = self._check_meetings(current_pos=start_pos, meeting_times=meeting_times)
        meeting_times = meeting_times + meeting_bool

        # Run the walks and at each time step check if some walks have met. Also, end the loop if all walks have met.
        # Here a while condition would look better, but we could not use tqdm to time it
        array_current = array_start
        for t in tqdm(range(max_time_steps)):
            # Find next position of the walks
            next_pos, array_next = self._next_step(array_current=array_current)
            array_current = array_next

            # Check meetings at this round
            meeting_bool = self._check_meetings(current_pos=next_pos, meeting_times=meeting_times)
            meeting_times = meeting_times + (t + 1) * meeting_bool

            # Verify if all walks have met
            complete = self._check_complete(meeting_times=meeting_times)
            if complete:
                return cp.asnumpy(meeting_times)

        return cp.asnumpy(meeting_times)

    @staticmethod
    def _check_meetings(current_pos: cp.ndarray, meeting_times: cp.ndarray) -> cp.ndarray:
        """


        This method checks if the walks meet for the first time at the current step.

        Args:
            current_pos (1D cp.ndarray): the current position of the walks (i.e. the names of the vertices at which the walks are)
            meeting_times (2D cp.ndarray): the matrix of meeting times. Entry i, j is the meeting time of walk i with walk j. \
            The default value of -1 is set when the walks have not met yet.

        Returns:
            (2D cp.ndarray of boolean values): A matrix with entry (i,j) being true if walk i has met walk j for the first time at current_position \
             and false otherwise (i.e. either the walks are not at the same position, or they have met before).

        """

        # Check if the walks are at the same position
        auxiliary_1 = current_pos.reshape(-1, 1) * cp.ones((current_pos.shape[0], current_pos.shape[0]))
        same_pos_bool = (current_pos == auxiliary_1)

        # Check if the walks have met before
        auxiliary_2 = -1 * cp.ones((meeting_times.shape[0], meeting_times.shape[1]))
        not_already_met_bool = (meeting_times == auxiliary_2)

        # If walk i,j are at the same place and have never met before set entry i,j = True, else set entry i,j = False.
        first_meeting_bool = same_pos_bool * not_already_met_bool

        return first_meeting_bool

    @staticmethod
    def _check_complete(meeting_times: cp.ndarray) -> bool:
        """

        This method check if all the walks have met

        Args:
            meeting_times (2D cp.ndarray): the matrix of meeting times. Entry i, j is the meeting time of walk i with walk j. \
            The default value of -1 is set when the walks have not met yet.

        Returns:
            (bool):  True if all the walks have met and False otherwise

        """

        # Compares an auxiliary matrix of all -1 with the meeting time matrix
        auxiliary = -1 * cp.ones((meeting_times.shape[0], meeting_times.shape[1]))
        not_already_met_bool = (meeting_times == auxiliary)

        # Counts how many non-zero entries are in the matrix not_already_met_bool
        nb_not_met = int(cp.count_nonzero(not_already_met_bool))

        if nb_not_met == 0:
            return True
        else:
            return False

    def _next_step(self, array_current: sparse.csr_matrix) -> Tuple[cp.ndarray, sparse.csr_matrix]:
        """

         Given the current step of all the samples of the random walk, computes the next step for all the samples of the random walk

        Args:
            array_current (cupyx.scipy.sparse.csr_matrix): A matrix with rows indicating the current position of the random walk. \
            Each row is of the form [0,0, ..., 0, 1, 0, ..., 0]. The entry 1 indicates the current position of the random walk.

        Returns:
            (cp.ndarray, cupyx.scipy.sparse.csr_matrix):
            A tuple consisting of:
                * A 1D cp.ndarray with for each sample of the random walk the index of the vertex where the rw will jump next
                * A 2D cupyx.scipy.sparse.csr_matrix with rows indicates the next position of the random walk of a sample.

            Each row is of the form [0,0, ..., 0, 1, 0, ..., 0]. The entry 1 indicates the next position of the random walk.

        """

        # Compute the probabilities of moving to given neighbors by multiplying sparse matrices current_step and P
        proba_next_pos = array_current * self.P

        # Sample index of next step of the random walk.
        next_pos = self._compute_next_position_from_proba(proba_next_pos)

        # Create indices for sparse matrix
        ind_ptr = cp.arange(array_current.shape[0] + 1, dtype=cp.int64)

        # Create values for sparse matrix
        values = cp.ones(array_current.shape[0], dtype=cp.float64)

        # Construct a 2D sparse tensor with rows of the form [0,0, ..., 0, 1, 0, ..., 0], where the entry 1 indicates the position of the rw
        array_next = sparse.csr_matrix((values, next_pos, ind_ptr),
                                       shape=(array_current.shape[0], array_current.shape[1]))

        # One could initialise the sparse matrix as follows (more intuitive)
        # array_next = sparse.csr_matrix((values, (ind_row, next_pos)), shape=(array_current.shape[0],array_current.shape[1]))
        # However, this slows down the code remarkably

        return next_pos, array_next

    @staticmethod
    def _compute_next_position_from_proba(proba_next_pos: sparse.csr_matrix) -> cp.ndarray:
        """

        For each row i of proba_next_pos, corresponding to one random walk, it samples the column index j with the probability specified in the entry proba_next_pos[i][j].
        The index j corresponds to the next position of the walk

        Args:
            proba_next_pos (2D cupyx.scipy.sparse.csr_matrix): a matrix with entry i,j being the probability of moving from i to j for the random walk. \
            The matrix has axis 0 of length n_samples (i.e. number of walks) and axis 1 of length n_nodes (i.e. the probability \
            for the walk to move at each one of those nodes)

        Returns:
            (1D cp.ndarray): the next position of the walks, randomly sampled from proba_next_pos

        """

        # This method is actually 3xfaster if only we could write the cuda code for it!!
        # Indeed, instead of transforming proba_next_pos into an array with a bunch of zeros, we could only
        # consider the entries in which we are interested.

        # Changes a cupy array into a torch tensor
        proba_next_pos_torch = torch.as_tensor(proba_next_pos.toarray(), device='cuda')

        # Samples stuff using torch
        next_pos_torch = torch.multinomial(proba_next_pos_torch, 1).flatten()

        # Converts back to cupy format
        next_pos = cp.fromDlpack(to_dlpack(next_pos_torch))

        return next_pos

    def _start_position(self, n_samples: int, start_position: cp.ndarray) -> Tuple[cp.ndarray, sparse.csr_matrix]:
        """

        Randomly chooses starting vertices for each sample of the random walks

        Args:
            n_samples (int): the number of random walks you want to sample
            start_position (1D cp.ndarray): starting positions for each sample of the walk (if None random starting positions are selected)

        Returns:
            [(cp.ndarray, cupyx.scipy.sparse.csr_matrix)]:
            A tuple consisting of:
                * A 1D cp.ndarray with for each sample of the random walks the index of the starting vertex of the random walks
                * A 2D cupyx.scipy.sparse.csr_matrix with rows indicates the starting position of the random walk of a sample.

            Each row is of the form [0,0, ..., 0, 1, 0, ..., 0]. The entry 1 indicates the position of the random walk.

        """

        # Set the start position
        start_pos = start_position

        # Create indices for sparse matrix
        ind_ptr = cp.arange(n_samples + 1, dtype=cp.int64)

        # Create values for sparse matrix
        values = cp.ones(n_samples, dtype=cp.float64)

        # Construct a 2D sparse tensor with rows of the form [0,0, ..., 0, 1, 0, ..., 0], where the entry 1 indicates the position of the rw
        array_start = sparse.csr_matrix((values, start_pos, ind_ptr), shape=(n_samples, self.n_nodes))

        # One could initialise the sparse matrix as follows (more intuitive)
        # array_next = sparse.csr_matrix((values, (ind_row, next_pos)), shape=(array_current.shape[0],array_current.shape[1]))
        # However, this slows down the code remarkably

        return start_pos, array_start
