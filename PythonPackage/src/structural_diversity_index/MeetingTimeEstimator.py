import numpy as np


class MeetingTimeEstimator:
    """

    The class MeetingTimeEstimator implements some methods that allow to estimate the value of a meeting time sample, given that the walks simulated to compute that sample did not meet.
    The main idea behind these methods is to use the fact that meeting times are approximately geometrically distributed

    """

    @staticmethod
    def estimate_meeting_times_unmet_random_walks(meeting_times: np.ndarray, max_time_steps: int) -> np.ndarray:
        """

        Given samples of the meeting times of two random walks containing some samples in which the walks did not meet, this method estimates the value of the samples in which the walks did not meet.

        Our estimate for the meeting time is based on the following reasoning. If the meeting time is approximately geometrically distributed it suffices to:
            * Estimate the parameter p of the geometric distribution which best approximate the distribution of the meeting time
            * Replace each -1 (i.e. each sample of the meeting time to be estimated) with: v + max_time_steps + 1, where v is a value sampled from the geometric distribution with parameter p. \
              Here we are using the memory-less property of the geometric distribution.


        Args:
            meeting_times (1D np.ndarray): a 1D np.ndarray containing samples of the meeting times of two random walks. Indicate by -1 the samples which you want to be estimated
            max_time_steps (int): The number of time steps that the random walks have been run in order to compute the samples of the meeting times passed in the array meeting_times. \
            For the samples with a -1 we assume that the walks took more than max_time_steps to meet.

        Returns:
             (1D np.ndarray): a 1D np.ndarray containing samples of the meeting times of two random walks. The -1 in the array passed as parameter are replaced by estimates for the meeting time.

        """

        # If there are no walks that did not meet there is nothing to estimate
        n_unmet_random_walks = len(meeting_times) - np.count_nonzero(meeting_times + 1)
        if n_unmet_random_walks == 0:
            return meeting_times

        # Otherwise, we estimate the meeting time as described above
        else:
            # Estimate the parameter p of the geometric distribution
            p = MeetingTimeEstimator.estimate_p_geometric_approximation(meeting_times=meeting_times,
                                                                        max_time_steps=max_time_steps)

            # Find the indices of the samples where the walks did not meet
            index_unmet_walks = np.argwhere(meeting_times == -1).reshape(-1)

            # Replace the value -1 at the indices in index_unmet_walks with the estimate constructed as described above
            meeting_times_with_estimate = np.zeros(len(meeting_times))
            meeting_times_with_estimate[index_unmet_walks] = np.random.geometric(p=p, size=len(
                index_unmet_walks)) + max_time_steps + 1
            meeting_times_with_estimate = meeting_times_with_estimate + meeting_times

            return meeting_times_with_estimate

    @staticmethod
    def estimate_p_geometric_approximation(meeting_times: np.ndarray, max_time_steps: int) -> float:
        """

        Computes the parameter p of the geometric distribution which best approximate the distribution of the meeting time.

        The computation is done by using the following equation:
        P[T leq t] = (1 - p)^t.
        Here T is the meeting time of the walks. If T is geometrically distributed the equation above should hold for some p.
        To compute this p, we compute all terms in the equation and then isolate p.
        Here t = max_time_steps and P[T leq t] is computed by counting the number of walks that did not meet.

        Args:
            meeting_times (1D np.ndarray): a 1D np.ndarray containing samples of the meeting times of two random walks. Indicate by -1 the samples which you want to be estimated
            max_time_steps (int): The number of time steps that the random walks have been run in order to compute the samples of the meeting times passed in the array meeting_times. \
            For the samples with a -1 we assume that the walks took more than max_time_steps to meet.

        Returns:
             (float): the parameter p of the geometric distribution which best approximate the distribution of the meeting time
        """

        # Compute P[T \leq t] where T is the meeting time of the random walks and t = max_time_steps
        n_unmet_random_walks = len(meeting_times) - np.count_nonzero(meeting_times + 1)
        probability_not_met = n_unmet_random_walks / len(meeting_times)

        # Compute p using the formula P[T \leq t] = (1 - p)^t
        p = 1 - np.exp(np.log(probability_not_met) / max_time_steps)

        return p
