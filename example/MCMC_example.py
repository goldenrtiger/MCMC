import numpy as np
import pandas as pd
from libMCMC import MCMC
from libGaussian import Gaussian 

'''

prob(param|x) = prob(x|param)prob(param)/prob(x)
prior: prob(param)
likelihood: prob(x|param)

f(x) = ((x^a + y^b)/(z^c)) - obs
df(x)|da = ((x^a)*log(x)/(z^c)) - obs
df(x)|db = ((y^b)*log(y)/(z^c)) - obs
df(z)|dc = (-(log(z)*(x^a + y^b))/z^c) - obs

'''


size = 10

class example:
    initial = None
    prior_gaussian = None 
    likeli_gaussian = None
    observations = None
    observations_x_mean = None
    observations_x_cov = None
    observations_y_mean = None
    observations_y_cov = None
    observations_z_mean = None
    observations_z_cov = None

    def __init__(self, observations):
        self.a_mean, self.a_cov = 1., 0.1
        self.b_mean, self.b_cov = 2., 0.1
        self.c_mean, self.c_cov = 3., 0.1

        example.observations = observations     
        self._observations_plus_noise()

        example.prior_gaussian = Gaussian(np.array([1., 2., 3.]).reshape(3, -1), np.eye(3))
        example.initial = example.prior_gaussian.sample()
        example.likeli_gaussian = Gaussian(example.observations[3,:].reshape(-1, size)
                                            , np.eye(size) * (np.var(example.observations[3,:])))


        # example.observations_x_mean, example.observations_x_cov = observations[:,0].mean(), np.cov(observations[:,0])

    @staticmethod
    def calc_estimations(param):
        a, b, c = param[0], param[1], param[2]
        x, y, z = example.observations[0,:], example.observations[1,:], example.observations[2,:]
        valuex, valuey, valuez = np.power(x, a), np.power(x, b), np.power(z, c)

        assert not np.isnan(valuex).any() and not np.isnan(valuey).any() and not (valuez == 0.).any()

        return ((valuex + valuey) / valuez).reshape(-1, size)

    @staticmethod
    def make_weights_prior():
        pass

    @staticmethod
    def posterior_log_prob(params):
        pass

    @staticmethod
    def potential(params):
        observation = example.calc_estimations(params)[0]

        return observation / example.observations[3, :][0]

    @staticmethod
    def potential_grad(params):
        a, b, c = params[0], params[1], params[2]

        x, y, z = example.observations[0, :], example.observations[1, :], example.observations[2, :]
        obs = example.observations[3, :][0]

        df_da = np.power(x, a) * np.log(x) / np.power(z, c) / obs
        df_db = np.power(y, b) * np.log(y) / np.power(z, c) / obs
        df_dc = (-(np.log(z) * (np.power(x, a) + np.power(y, b)) / np.power(z, c))) / obs            
        
        return np.array([df_da[0], df_db[0], df_dc[0]]).reshape(-1, 3)  

    def _observations_plus_noise(self):
        self.observations = self._true_polynomial() + np.random.randn()

    def _true_polynomial(self):
        x, y, z = example.observations[0,:], example.observations[1,:], example.observations[2,:]
        observation = ((np.power(x, self.a_mean) + np.power(y, self.b_mean)) / np.power(z, self.c_mean)).reshape(-1, size)
        example.observations = np.concatenate((example.observations, observation), axis=0)

        return example.observations


def summary(samples):    
    data = pd.DataFrame(samples)    
    describe = data.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
    return describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]

if __name__ == "__main__":
    x, y, z = np.abs(np.random.randint(1, 10, size=(1, size)))   \
            , np.abs(np.random.randint(1, 10, size=(1, size)))   \
            , np.abs(np.random.randint(1, 10, size=(1, size)))

    # x, y, z = np.array([2.] * size).reshape(size, -1), np.array([4.] * size).reshape(size, -1), np.array([3.] * size).reshape(size, -1)
    observations = np.concatenate((x, y, z), axis=0)
    test = example(observations)

    mcmc = MCMC(initial = example.initial, prior_gaussian = example.prior_gaussian
                , likeli_gaussian = example.likeli_gaussian
                , observations = example.observations[3,:].reshape(-1, size)
                , fun_estimation = example.calc_estimations
                , select = "HMC"
                , stepsize = 0.003
                , iters = 1000
                , steps = 80
                , potential = example.potential
                , grad_potential = example.potential_grad)
    
    mcmc.run()

    samples = mcmc.get_samples()
    total_samples = len(samples)
    a_samples, b_samples, c_samples = np.array([sample[0] for sample in samples]).reshape(-1,1) \
                                    , np.array([sample[1] for sample in samples]).reshape(-1,1) \
                                    , np.array([sample[2] for sample in samples]).reshape(-1,1)

    print(f"a:\n {summary(a_samples)}")
    print(f"b:\n {summary(b_samples)}")
    print(f"c:\n {summary(c_samples)}")
















