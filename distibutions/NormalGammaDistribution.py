import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal, gamma


class NormalGammaDistribution(object):

    def __init__(self, mu, kappa, alpha, beta):

        self.mu = mu

        self.kappa = kappa

        self.alpha = alpha

        self.beta = beta

    def rvs(self, size):

        tau = gamma(shape=self.alpha, scale=self.beta, size=size)

        mean = normal(loc=self.mu, scale=(1 / (self.kappa * tau)), size=size)

        return mean, tau

    def pdf(self, mean, tau):

        # val0 = (self.beta ** self.alpha) * np.sqrt(self.kappa) / (gamma(self.alpha) * np.sqrt(2*np.pi))

        val1 = tau ** (self.alpha - 1/2)

        val2 = np.exp(-self.beta * tau)

        val3 = np.exp((-self.kappa * tau * (mean - self.mu) ** 2) / 2)

        return val1 * val2 * val3

    def plot(self, mu_range0, mu_range1, tau_range0, tau_range1):

        mus = np.linspace(mu_range0, mu_range1, 1000)

        taus = np.linspace(tau_range0, tau_range1, 100)

        M, T = np.meshgrid(mus, taus, indexing="ij")

        Z = np.zeros_like(M)

        for i in range(Z.shape[0]):

            for j in range(Z.shape[1]):

                Z[i][j] = self.pdf(mus[i], taus[j])

        plt.contourf(M, T, Z)

        plt.title("base distribution")

        plt.xlabel("mean")

        plt.ylabel("tau")

        plt.colorbar()

        plt.show()





