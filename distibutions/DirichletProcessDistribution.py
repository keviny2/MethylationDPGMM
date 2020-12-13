import numpy as np
from utils import get_cluster_matrix, log_factorial, get_normalized_p


class DirichletProcessDistribution(object):

    @staticmethod
    def rvs(alpha, N):

        # create numpy arrays

        cluster_label = np.zeros(N)

        points_per_cluster = np.array([])

        # assign first data point

        cluster_label[0] = 0

        points_per_cluster = np.append(points_per_cluster, 1)

        num_tables = 1

        # assign other data points

        for i in range(N-1):

            # calculate existing probs

            existing_log_prob = np.log(points_per_cluster) - np.log(N - 1 + alpha)

            # calculate new probs

            new_log_prob = np.log(alpha) - np.log(N - 1 - alpha)

            # merge existing and new probs

            log_prob = np.append(existing_log_prob, new_log_prob)

            prob = get_normalized_p(log_prob)

            # sample new table

            new_label = np.random.choice(num_tables + 1, p=prob)

            cluster_label[i] = new_label

            # update fields

            if new_label == num_tables:

                points_per_cluster = np.append(points_per_cluster, 1)

                num_tables += 1

            else:

                points_per_cluster[new_label] += 1

        return get_cluster_matrix(cluster_label)

    @staticmethod
    def log_p(alpha, Z):

        N = Z.shape[0]

        K = Z.shape[1]

        m = np.sum(Z, axis=0)

        if K == 0:

            return -np.inf

        log_p = 0

        log_p -= np.sum(np.log(np.arange(N) + alpha))

        log_p += K * np.log(alpha)

        for i in range(K):

            log_p += log_factorial(m[i] - 1)

        return log_p




