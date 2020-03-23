import numpy as np
import pandas as pd
import warnings, os

isDebug = False

def isPrint(Debug, string):
    if Debug: print(string)

class specimenSize:
    '''
        s: nozzle size
        fm: extrusion speed
        Vr: printing speed
        Zn: nozzle offset

        observation: [s, fm, Vr, Zn]
        y: targeted width

        f(w|a, b, c, d) = s * fm ** a * Zn ** b * c / Vr ** d 
        p(w|a, b, c, d) = (s * fm ** a * Zn ** b * c / Vr ** d) 
        params: [a, b, c, d]


        
    '''
    def __init__(self, num_samples, debug = False):
        self.mu_abc = 1
        self.sigma_abc = 1
        self.mu_d = 1
        self.sigma_d = 1
        self.num_samples = num_samples

        [self.y, self.observation, w] = self.make_training_data(num_samples)
        
        self.params_Gaussian = Gaussian([self.mu_abc, self.mu_abc, self.mu_abc, self.mu_d]
                                , np.eye(4))

        self.debug = debug
        self.observation_prec = 1.0 / np.array(self.observation).reshape(-1,1)
        self.observation_Z = 1.0 / np.sqrt(2.0 * np.pi) * np.std(self.observation).reshape(-1, 1)
    
    def make_training_data(self, num_samples):
        # uniform
        x = np.abs(np.random.randn(4, num_samples)) # fm, Zn, Vr
        [s, fm, Zn, Vr] = x

        mean = np.array([1., 1., 1., 1.])
        cov = np.array([1., 1., 1., 1.])
        noise = np.random.normal(1, 0.5, num_samples)

        w = np.abs(np.random.normal(mean, cov, (num_samples, cov.shape[0]))).T # a, b, c, d
        [a, b, c, d] = w

        try: 
            y = s* np.power(fm, a) * np.power(Zn, b) * c / np.power(Vr, d) + noise
        except Exception as e:
            print(f"!! s:{s}, fm:{fm}, a:{a}, Zn:{Zn}, b:{b}, c:{c}, Vr:{Vr}, d:{d}")
            print(e)
            os._exit(0)
        else:
            return y, x, w

    def gradient_likelihood(self, observation, params):
        s, fm, Vr, Zn = self.observation[0], self.observation[1], self.observation[2], self.observation[3]
        [a, b, c, d] = params

        warnings.filterwarnings('error')
        gradient = np.array([0.0] * 4)
        gradient = 0
        with warnings.catch_warnings():
            try:
                obs = s * fm ** a * Zn ** b * c / Vr ** d                 
                obs_sum = np.sum(self.observation - obs).reshape(-1,1)
                np.matmul(
                    self.observation_prec, obs_sum
                )
            except Warning as e:
                isPrint(self.debug, f"!! {self.gradient_likelihood.__name__}: len(observation): {len(observation)}, abcd:{a}{b}{c}{d} {e}")
        return gradient

    def gradient_prior(self, observation, params):
        '''
        And we will assume that we have a Gaussian Prior.
        From the Matrix Cookbook [1, eqn. 85], know that
        d(log(p(x)))/dq = Sigma^{-1}(q - mu)
        (85) applies to our prior, and equation (86) applies for the likelihood.
        They only differ by a negative sign.
        This is because we are using the log of the distribution, so we get rid
        of the exp() and seperate the exponent from the normalising constant,
        and when taking the derivative the constant terms will vanish, and
        because a valid covariance matrix is symmetric.
        '''
        gradient = -np.matmul(self.params_Gaussian.prec, (params - self.params_Gaussian.mean).reshape(-1, 1))
        isPrint(self.debug, f"gradient_prior,{gradient}")
        return gradient

    def prior_log(self, observation, params):
        warnings.filterwarnings('error')
        with warnings.catch_warnings():
            try:
                value = np.log(self.params_Gaussian.eval_pdf(params))
            except Warning as warning:
                print(f"!! prior_log")
                print(warning)
                os._exit(0)
        return value
        

    def likelihood_log(self, observation, params):   
        s, fm, Vr, Zn = self.observation[0], self.observation[1], self.observation[2], self.observation[3]
        [a, b, c, d] = params

        value = 0
        with warnings.catch_warnings():
            try:
                obs = (s * np.power(fm, a) * np.power(Zn, b) * c / np.power(Vr, d)).T                 
                obs_sum = np.sum(self.y - obs).reshape(-1,1)
                U_log = - ( 0.5 * obs_sum ** 2 / obs_sum 
                            - self.num_samples * np.log(self.observation_Z))
            except Warning as warning:
                print(f"!! s:{s}, fm:{fm}, a:{a}, Zn:{Zn}, b:{b}, c:{c}, Vr:{Vr}, d:{d}")
                print(warning)
                os._exit(0)
        return value
    
    def acceptance(self, test):
        # accept/reject this sample using Metropolis Hasting
        one_samp = np.random.uniform(low = 0.0, high = 1.0)
        if one_samp < test:
            return 1
        else:
            return 0

    def transition_model(self):
        params = (self.params_Gaussian.sample_multivariate())
        return params

    def run_estimate_hmc_log(self, initial_q, observation,
                    epsilon = 0.01, L = 20, iters = 1000):
        hmc = MCMC().hmc(initial_q, self.prior_log, self.likelihood_log, observation,
                            self.gradient_likelihood, self.gradient_prior
                            , self.transition_model, debug = self.debug, epsilon = epsilon
                            , iters = iters)
        return hmc.run()

class MCMC:
    class hmc:
        '''
            q: position
            p: velocity
        '''
        def __init__(self, initial_q, prior, likeli, observation, 
                        gradient_likelihood, gradient_prior,
                        transition_model,  debug = False,
                        epsilon = 0.01, L = 20, iters = 1000):
            self.initial_q = initial_q
            self.prior = prior
            self.likeli = likeli
            self.observation = observation
            self.gradient_likelihood = gradient_likelihood
            self.gradient_prior = gradient_prior
            self.transition_model = transition_model
            self.debug = debug
            self.epsilon = epsilon
            self.L = L
            self.iters = iters

        def potential_gradient(self, q):
            return -(self.gradient_likelihood(self.observation, q) + self.gradient_prior(self.observation, q))

        def potential_log(self, q):
            return -(self.prior(self.observation, q) + self.likeli(self.observation, q))
        
        def run(self):
            current_q = self.initial_q
            dim = current_q.shape[0]
            q_pos = []
            q_acc = []
            accept_count = 0
            gradient = np.array(4)

            for i in range(self.iters):                
                # q = current_q 
                q = self.transition_model()
                p = self.transition_model()
                # save the current value of p
                current_p = p
                isPrint(self.debug, f">> ------------- current_q:{q}, p:{p} -----------")
                gradient = self.potential_gradient(q)
                p = p - self.epsilon * gradient / 2.0
                gradient_list = [gradient]
                q_p_list = [[q,p]]
                epsilon_new = self.epsilon
                for _ in range(self.L):                                        
                    q = np.abs(q + self.epsilon * p)
                    gradient_iter_new = self.potential_gradient(q)
                    p = p - epsilon_new * gradient_iter_new
                    isPrint(self.debug, f">> iteration:{q}{p}, gradient:{gradient}")

                isPrint(self.debug, f"q:{q}, p:{p}")
                gradient = self.potential_gradient(q) 
                p = p - self.epsilon * gradient / 2.0
                p = -p

                # avoid log(0)
                if (np.abs(p) < 0.001).any() or (q < 0.001).any()  \
                    or (np.abs(p) > 1000).any() or (q > 1000).any(): 
                    print("continue")
                    continue 
                
                current_q = np.abs(current_q)
                q = np.abs(q)
                current_U = self.potential_log(current_q)
                current_K = current_p.T @ current_p / 2.0
                proposed_U = self.potential_log(q)
                proposed_K = p.T @ p / 2.0
                isPrint(self.debug, f">> current_U - proposed_U:{current_U - proposed_U},current_K - proposed_K:{current_K - proposed_K}")

                test = np.exp(current_U - proposed_U + current_K - proposed_K)
                one_samp = np.random.uniform(low = 0.0, high = 1.0)
                isPrint(self.debug, f">> test:{test}")
                if one_samp < test:
                    accept_count += 1
                    q_pos.append(q)
                    current_q = q
                    
                else:
                    q_pos.append(current_q)
                    # pass          

            print(f">> --- accept ratio:{accept_count/self.iters}")
            return q_pos
            

class Gaussian:
    def __init__(self, mean, cov):
        print(f"mean:{mean}, cov:{cov}")
        self.mean = np.array(mean).reshape(-1, 1).astype(np.float64)
        self.cov =  np.array(cov).astype(np.float64)
        self.dim = self.mean.shape[0]
        self.prec = np.linalg.inv(self.cov)
        self.cov_det = np.linalg.det(self.cov)

        self.Z = 1.0 / np.sqrt((2.0 * np.pi) ** self.dim * self.cov_det)
        
        if self.mean.shape[0] != self.cov.shape[0] :
            raise(ValueError('Inconsistent dimensions for mean and cov'))
        if(self.cov.shape[0] != self.cov.shape[1]):
            raise(ValueError('Covariance Matrix should be square'))

    def sample_multivariate(self, n_samp = 1):
        sample = np.random.multivariate_normal(self.mean.reshape(self.dim),
                                                self.cov,
                                                size = n_samp).T
        return sample

    def sample(self, n_samp = 1):
        return np.random.normal(self.mean.reshape(self.dim),
                                self.cov, 
                                size = n_samp)

    def eval_pdf(self, x):
        return self.Z * np.exp(-0.5 * (
                (x - self.mean).T @ self.prec @ (x - self.mean)
        ))
        
if __name__ == '__main__':    
    # np.random.seed(1)
    warnings.filterwarnings('error')

    SS = specimenSize(num_samples = int(10), debug = isDebug)

    # initial data
    params = SS.params_Gaussian.sample_multivariate()
    q_pos = SS.run_estimate_hmc_log(params, SS.observation, epsilon=0.005, iters = 5000, L = 40)

    a, b, c, d = [], [], [], []
    for _ in range(len(q_pos)):
        a.append(q_pos[_][0])
        b.append(q_pos[_][1])
        c.append(q_pos[_][2])
        d.append(q_pos[_][3])
    
    if len(q_pos):
        print(f"a: {pd.DataFrame(a).describe()}")
        print(f"b: {pd.DataFrame(b).describe()}")
        print(f"c: {pd.DataFrame(c).describe()}")
        print(f"d: {pd.DataFrame(d).describe()}")
    else:
        print("None q_pos")

