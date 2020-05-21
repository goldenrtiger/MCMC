import numpy as np

class Gaussian:
    def __init__(self, mean, cov):
        self.mean = np.array(mean).reshape(-1, 1).astype(np.float64)
        self.cov = cov.astype(np.float64)
        self.dim = self.mean.shape[0]
        self.prec = np.linalg.inv(self.cov)
        self.cov_det = np.linalg.det(self.cov)

        self.Z = 1.0 / np.sqrt((2.0 * np.pi) ** self.dim * self.cov_det)
        
        if self.mean.shape[0] != self.cov.shape[0] :
            raise(ValueError('Inconsistent dimensions for mean and cov'))
        if(self.cov.shape[0] != self.cov.shape[1]):
            raise(ValueError('Covariance Matrix should be square'))

    def sample(self, n_samp = 1):
        return np.random.multivariate_normal(self.mean.reshape(self.dim),
                                                self.cov,
                                                size = n_samp).T

    def eval_pdf(self, x):
        return self.Z * np.exp(-0.5 * (
                (x - self.mean).T @ self.prec @ (x - self.mean)
        ))

    def eval_true_posterior_gaussian(prior, likelihood, x):
        """Will evaluate true posterior for Gaussian Prior/Posterior

        For Gaussian Prior N(q|mu_0, sigma_0)
            Gaussian Likelihood N(x|mu_l, sigma_l)
        From [1, eqn. 4.124 and 4.125]
        Posterior is,
        p(q|x) = N(q | mu, sigma)
        Sigma^{-1} = Sigma_0^{-1} + Sigma_l^{-1}
        mu = Sigma (Sigma_l^{-1}x + Sigma_0^{-1}mu_0)
        (To use result from [1], matrix A is set to identity and
        vector b is set to zero).

        Args:
        prior (Gaussian):
            Gaussian Object
        likelihood (Gaussian):
            Gaussian Object
        x (array):
            our data, with each column being an independent sample

        Returns:
        Gaussian Object with our true posterior

        References:
        [1] Murphy, K. (2012). "Machine Learning : A Probabilistic Perspective."
            Cambridge: MIT Press.
        """
        x_bar = np.mean(x, axis = 1).reshape(-1, 1)
        n_samp = x.shape[1]
        cov = np.linalg.inv(prior.prec + n_samp * likelihood.prec)
        mean = cov @ (n_samp * likelihood.prec @ x_bar + prior.prec @ prior.mean)
        print(f"true mean: {mean}")

        return Gaussian(mean, cov)

















