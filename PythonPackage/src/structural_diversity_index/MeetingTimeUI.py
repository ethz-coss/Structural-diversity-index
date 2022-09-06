from .MeetingTimeEstimator import MeetingTimeEstimator

import numpy as np
import networkx as nx
from scipy import sparse
from typing import Tuple


class MeetingTimesUI:
    """

    This class provides a user interface for the code computing the structural diversity index.

    """

    @staticmethod
    def _get_random_walk_simulator_networkx(g: nx.Graph, on_cuda: bool):
        """

        Given a graph it gets the random walk simulator for that graph.

        Args:
            g (nx.Graph): the graph on which you want to simulate a random walk
            on_cuda (bool): if True, uses cuda to make computations

        Returns:
            (RandomWalkSimulator or RandomWalkSimulatorCUDA): An instance of the class RandomWalkSimulator or RandomWalkSimulatorCUDA

        """
        adjacency = nx.to_scipy_sparse_array(G=g, format='csr', dtype=float)
        degrees = np.array([d[1] for d in g.degree])
        return MeetingTimesUI._get_random_walk_simulator(adjacency=adjacency, degrees=degrees, on_cuda=on_cuda)

    @staticmethod
    def _get_random_walk_simulator(adjacency: sparse.csr_matrix, degrees: np.ndarray, on_cuda: bool):
        """

        Given an adjacency matrix and a degree sequence of a graph it gets the random walk simulator for that graph.

        Args:
            adjacency (sparse.csr_matrix): the adjacency matrix of the graph
            degrees (np.ndarray): the degree sequence of the graph
            on_cuda (bool): if True, uses cuda to make computations

        Returns:
            (RandomWalkSimulator or RandomWalkSimulatorCUDA): An instance of the class RandomWalkSimulator or RandomWalkSimulatorCUDA

        """
        if on_cuda:
            from .RandomWalkSimulatorCUDA import RandomWalkSimulator as RandomWalkSimulatorCUDA
            random_walk_matrix = RandomWalkSimulatorCUDA.random_walk_matrix_from_adjacency(adjacency=adjacency,
                                                                                           degrees=degrees)
            rws = RandomWalkSimulatorCUDA(random_walk_matrix=random_walk_matrix)
        else:
            from .RandomWalkSimulator import RandomWalkSimulator
            random_walk_matrix = RandomWalkSimulator.random_walk_matrix_from_adjacency(adjacency=adjacency,
                                                                                       degrees=degrees)
            rws = RandomWalkSimulator(random_walk_matrix=random_walk_matrix)
        return rws

    @staticmethod
    def get_meeting_times(g: nx.Graph, max_time_steps: int = None, n_samples: int = 1000, on_cuda: bool = False) -> np.ndarray:
        """

        Returns the n_samples of the meeting time of two uniformly started random walks on the graph g


         Args:
             g (nx.Graph): the graph for which you want the meeting time samples
             max_time_steps (int): the maximum number of time steps to run the random walks \
              to compute the meeting time (default = 10*g.number_of_nodes())
             n_samples (int): the number of samples of the meeting time (default = 1000)
             on_cuda (bool): if True, uses cuda to make computations (default = False)

         Returns:
             (np.ndarray): An array in which each entry is one sample of the meeting time on the graph g. \
             Note: To estimate the meeting time of random walks that do not meet within max_times_steps steps \
             we use the method MeetingTimeEstimator.estimate_meeting_times_unmet_random_walks(). \
             See the class for a description.

         """

        if max_time_steps is None:
            max_time_steps = 10 * g.number_of_nodes()

        # With k random walks we can compute k**2 - k samples of the meeting time. Hence, if we want
        # n_samples of the meeting time we just need [1 + sqrt(1+4n_samples)]/2 random walks
        n_samples_random_walks = int(np.ceil((1 + np.sqrt(1+4*n_samples))/2))

        # Create a random walk simulator and compute the meeting times
        rws = MeetingTimesUI._get_random_walk_simulator_networkx(g=g, on_cuda=on_cuda)
        meeting_times = rws.get_meeting_times(max_time_steps=max_time_steps, n_samples=n_samples_random_walks)
        meeting_times = MeetingTimeEstimator.estimate_meeting_times_unmet_random_walks(meeting_times=meeting_times,
                                                                                       max_time_steps=max_time_steps)

        # We get some more samples of the meeting time because n_samples_random_walks is a bit larger than [1 + sqrt(1+4n_samples)]/2
        # Hence we cut out some samples
        meeting_times = meeting_times[:n_samples]

        return meeting_times

    @staticmethod
    def get_structural_diversity_index(g: nx.Graph, max_time_steps: int = None, n_samples: int = 100, on_cuda: bool = False) -> Tuple[float, float]:
        """

        Returns the structural diversity index of a graph g

         Args:
             g (nx.Graph): the graph for which you want the structural diversity index
             max_time_steps (int): the maximum number of time steps to run the random walks \
              to compute the meeting time (default = 10*g.number_of_nodes())
             n_samples (int): the number of samples of the meeting time (default = 1000)
             on_cuda (bool): if True, uses cuda to make computations (default = False)

         Returns:
             (float, float): The structural diversity index of the graph g and the standard deviation in the samples \
             used for the computation. This index is computed by obtaining n_samples of the meeting time of the graph g \
              and averaging over them. Note: To estimate the meeting time of random walks that do not meet \
              within max_times_steps steps we use the method MeetingTimeEstimator.estimate_meeting_times_unmet_random_walks(). \
             See the class for a description.

         """

        meeting_times = MeetingTimesUI.get_meeting_times(g=g, max_time_steps=max_time_steps, n_samples=n_samples, on_cuda=on_cuda)
        delta_g = float(np.mean(meeting_times) / g.number_of_nodes())
        std_delta_g = float(np.std(meeting_times/g.number_of_nodes()))
        return delta_g, std_delta_g
