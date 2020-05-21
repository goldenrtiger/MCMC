import numpy as np
from libGaussian import Gaussian


class MCMC:
    '''
    '''
    def __init__(self, initial, prior_gaussian, likeli_gaussian, observations, fun_estimation, 
                    select, stepsize=0.3, steps=20, iters=1000, n_samp=10, potential=None, 
                    grad_potential=None):
        self.prior_gaussian = prior_gaussian
        self.likeli_gaussian = likeli_gaussian
        self.observations = observations
        self.stepsize = stepsize
        self.steps = steps
        self.iters = iters
        self.initial_q = None
        self.initial = None
        self.estimate = fun_estimation
        self.n_samp = n_samp

        if select == "HMC":
            self.sampling = self.HMC_sampling
            self.acceptance_rule = self.HMC_acceptance_rule
            self.initial_q = initial
            self.potential = potential
            self.grad_potential = grad_potential
        elif select == "metropolis_hastings":
            self.sampling = self.metropolis_hastings_sampling
            self.acceptance_rule = self.metropolis_hastings_acceptance_rule            
            self.initial = initial
        else:
            raise(ValueError(f"Invaild select:{select}. Please select HMC or metropolis_hastings." ))  

        self.points = []   


    def run(self):
        self.sampling()

    def get_samples(self):
        return self.points

    def metropolis_hastings_sampling(self):
        pass

    def metropolis_hastings_acceptance_rule(self):
        pass

    def HMC_sampling(self):
        current_q = self.initial_q
        size = len(self.initial_q)
        # p_dist = Gaussian([0.0] * size, np.eye(size))
        accept_rate = 0.

        for i in range(self.iters):
            q = current_q
            # p = p_dist.sample()
            p = np.random.randn(3, 1)

            current_p = p    
            # half update of momentum
            gradient = self._grad_potential_gaussian(self.observations, q, self.prior_gaussian, self.likeli_gaussian)
            # gradient = self.grad_potential(q)
            assert not np.isnan(gradient).any()

            p = p - self.stepsize * gradient / 2.0
            assert not np.isnan(p).any()

            for _ in range(self.steps - 1):
                # full update of the position
                q = q + self.stepsize * p
                assert not np.isnan(q).any()
                gradient = self._grad_potential_gaussian(self.observations, q, self.prior_gaussian, self.likeli_gaussian)
                # gradient = self.grad_potential(q)
                print(f"gradient:{gradient}")
                assert not np.isnan(gradient).any()
                # if np.abs(gradient) < 0.1 or np.abs(gradient) > 100.:
                #     break
                # make a full step in momentum unless we are on the last step
                p = p - self.stepsize * gradient
                
    
            # make a half step and then negate the momentum term
            p = p - self.stepsize * self._grad_potential_gaussian(self.observations, q, self.prior_gaussian, self.likeli_gaussian) / 2.0
            # p = p - self.stepsize * self.grad_potential(q) / 2.0
            p = -p

            # evaluate the potential and kinetic energies to see if we accept or reject
            current_U = self._potential(self.observations, current_q, self.prior_gaussian, self.likeli_gaussian, self.n_samp)
            # current_U = self.potential(current_q)
            current_K = current_p.T @ current_p / 2.0
            proposed_U = self._potential(self.observations, q, self.prior_gaussian, self.likeli_gaussian, self.n_samp)
            # proposed_U = self.potential(q)
            proposed_K = p.T @ p / 2.0
            # accept/reject this sample using Metropolis Hastings
            test = np.exp(current_U - proposed_U + current_K - proposed_K)
            
            if self.HMC_acceptance_rule(test):
                self.points.append(q)
                current_q = q
                accept_rate += 1

        accept_rate = accept_rate / self.iters        
        print(f"accept_rate:{accept_rate}")


    def HMC_acceptance_rule(self, test):        
        one_samp = np.random.uniform(low = 0.0, high = 1.0)
        if(one_samp < test):
            return True
        else:
            return False

    def _potential(self, observations, q, prior, likelihood, n_samp):
        "Calculate the potential energy at current position"
        estimation = self.estimate(q)
        # value = prior.eval_pdf(q)
        value = prior.eval_pdf(q)
        if (value == 0.).any():
            return 0.
        U_log_prior = np.log(value)
        obs_sum = np.sum(observations - estimation, axis = 0).reshape(-1, 1) 

        k = observations.shape[0]
        U_log_likelihood = - (
            0.5 * obs_sum.T @ likelihood.prec @ obs_sum - n_samp * np.log(likelihood.Z))
        return -(U_log_prior + U_log_likelihood)        

    def _grad_potential_gaussian(self, observations, q, prior, likelihood):
        """Computes gradient of potential energy for MVN

        Potential energy for HMC is typically defined as,
        U(q) = -log(p(q) p(x | q)) = -log(p(q)) - log(p(x | q))
        And we will assume that we have a Gaussian Prior and likelihood.
        From the Matrix Cookbook [1, eqn. 85], know that
        d(log(p(x)))/dq = Sigma^{-1}(q - mu)
        (85) applies to our prior, and equation (86) applies for the likelihood.
        They only differ by a negative sign.
        This is because we are using the log of the distribution, so we get rid
        of the exp() and seperate the exponent from the normalising constant,
        and when taking the derivative the constant terms will vanish, and
        because a valid covariance matrix is symmetric.

        Args:
        x (Np arr ay)
            data drawn from the likelihood. Each column is an independent
            draw from likelihood (column = single sample)

        References:
        [1] Petersen, Kaare Brandt, and Michael Syskind Pedersen.
        "The matrix cookbook." Technical University of Denmark 7.15 (2012): 510.
        """
        # now compute the gradient  
        estimation = self.estimate(q)
        dq_prior = - np.matmul(prior.prec, (q - prior.mean).reshape(-1, 1))
        # dq_prior = - np.matmul(prior.prec, (q - prior.mean).reshape(-1, 1))
        dq_likelihood =  np.matmul(
            likelihood.prec, np.sum(observations - estimation, axis = 0).reshape(-1, 1)) # todo
        # dq_likelihood = likelihood.prec * np.sum(observations - estimation)
        # return  -(np.sum(dq_prior) + np.sum(dq_likelihood)) 
        return -(dq_prior + np.sum(dq_likelihood))
























