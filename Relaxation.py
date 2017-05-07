import numpy as np
import copy


class Relaxation(object):
    """
    Particle tracking algorithm base on the article "Particle-tracking velocimetry with new algorithms"
    Kazuo Ohmi and Hang-Yu Li

    Parameters
    ----------
    R_n: float (default=30.0)
                Neighbours Similarly threshold, every ﬁrst-frame particle preselects its
                neighbours in the same ﬁrst frame, using also the distance threshold.
    R_p: float (default=30.0)
                Candidate partners threshold, ﬁrst-frame particle preselects its candidate
                partners from the second frame, using the following distance threshold.
    R_c: float (default=5.0)
                Radius of the relaxation area in which deviation from parallel motion is allowed.
    A:   float (default=0.3)
                Weighting constant for updating pair probability (the pair loss term).
    B:   float (default=3.0)
                Weighting constant for updating pair probability (impact of the neighbors term).
    epoch_n: float (default=3)
                Number epoch.
    verbose: int (default=0)
                Any positive number for verbosity.

    Attributes
    ----------
    frame_0:  array, shape = [n_particles, n_dimensions]
                Coordinates of particles form the first frame.
    frame_1:  array, shape = [n_particles, n_dimensions]
                Coordinates of particles form the second frame.

    Examples
    --------


    """

    def __init__(self, R_n=30.0, R_p=30.0, R_c=1, A=0.3, B=3.0, epoch_n=1, verbose=0):

        self.frame_0 = []
        self.frame_1 = []
        self.R_n = R_n
        self.neighbours = []
        self.R_p = R_p
        self.pairs = []
        self.R_c = R_c
        self.A = A
        self.B = B
        self.epoch_n = epoch_n
        self.vector_field = []

    def _find_neighbours(self):
        """
         Find neighbors for each particle in circle with radius R_n.
        """

        neighbours = []
        for i, p in enumerate(self.frame_0):
            nearests = np.where(np.linalg.norm(self.frame_0 - p, axis=1) <= self.R_n)[0]
            # delete self index
            index = np.argwhere(nearests==i)
            nearests = np.delete(nearests, index)
            neighbours.append(nearests)

        return neighbours

    def _find_pairs(self):
        """
         Find pair candidate for each particle in circle with radius R_p.
        """
        pairs = []
        for i, p in enumerate(self.frame_0):
            nearests = np.where(np.linalg.norm(self.frame_1 - p, axis=1) <= self.R_p)[0]
            # add probability missing pair.
            nearests = np.append(nearests, -1)
            prob = np.zeros_like(nearests) + 1.0 / nearests.shape[0]

            ind_prob = np.vstack([nearests, prob])

            pairs.append(ind_prob)

        return pairs

    def _calculate_Q(self, src_p, dst_p, pairs_last):
        """
        Calculate the contribution from the neighbour’s probability Pkl (for the neighbouring particle k with respect
        to its own candidate l)

        :param src_p: index of particle from the first frame, which is updating.
        :param dst_p:   index of candidate from the second frame.
        :param pairs_last: pairs and probability from the last epoch.
        :return: Probability sum of respect pairs.
        """

        src_shift = self.frame_1[dst_p] - self.frame_0[src_p]

        Q = 0.0
        for nei_i in self.neighbours[src_p]:
            nei_shift = self.frame_0[nei_i] + src_shift
            for seq_i, pair_i in enumerate(pairs_last[nei_i][0]):
                pair_i = int(pair_i)
                pair_coord = self.frame_1[pair_i]
                if np.linalg.norm(nei_shift - pair_coord) <= self.R_c:
                    Q += pairs_last[nei_i][1][seq_i]

        return Q

    def _update_probabilities(self):
        """
        Update pair probability for each particle from  the first frame.
        """
        pairs_last = copy.deepcopy(self.pairs)
        # pairs_last = [el for el in pairs]
        for src_p in range(self.frame_0.shape[0]):
            for seql_num, dst_p in enumerate(self.pairs[src_p][0]):
                Q = self._calculate_Q(src_p, dst_p, pairs_last)
                # update pair probability.
                self.pairs[src_p][1][seql_num] = pairs_last[src_p][1][seql_num] * (self.A + self.B * Q)

            # normalize probability
            self.pairs[src_p][1] = self.pairs[src_p][1] / self.pairs[src_p][1].sum()

    def fit(self, frame_0, frame_1):
        """
        Fit pair probability particles from the frame_0 to frame_1.
        :param frame_0: array, shape = [n_particles, n_dimensions]
                Coordinates of particles form the first frame.
        :param frame_1: array, shape = [n_particles, n_dimensions]
                Coordinates of particles form the second frame.
        """

        self.frame_0 = frame_0
        self.frame_1 = frame_1

        self.neighbours = self._find_neighbours()
        self.pairs = self._find_pairs()

        for _ in range(self.epoch_n):
            self._update_probabilities()

    def predict(self):
        """
        Predict pair for eadh particle from frame_0 to frame_1

        :return: vector field : array [[x_0, y_1 ... , x_1, y_1]]
        """
        for src_p, pair in enumerate(self.pairs):
            dst_p = pair[1].argmax()
            dst_ind = pair[0][dst_p]

            self.vector_field.append(np.hstack([self.frame_0[src_p], self.frame_1[dst_ind]]))

        self.vector_field = np.vstack(self.vector_field)

        return self.vector_field
