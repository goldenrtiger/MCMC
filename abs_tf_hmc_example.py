import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd

print(f">> ---- start --------- {tf.__version__}")

# tf.enable_eager_execution()

tfd = tfp.distributions

def make_training_data(num_samples, dims, params):
    '''
    s * fm ** a * Zn ** b * c / Vr ** d
    '''
    dt = np.asarray(params).dtype
    x = np.abs(np.random.randn(dims, num_samples).astype(dt)) # measurement data
    w = np.abs(params)   # params
    noise = np.random.randn(num_samples).astype(dt)

    [ s, fm, Zn, Vr ] = x
    [ a, b, c, d ] = w

    y = s* tf.math.pow(fm, a) * tf.math.pow(Zn, b) * c / tf.math.pow(Vr, d) + noise[0]
    return y, x, w

def make_weights_prior(dims, mean, log_sigma):
  return tfd.MultivariateNormalDiag(
      loc = mean,
      scale_identity_multiplier=tf.math.exp(log_sigma))

def make_response_likelihood(w, x):
    s, fm, Zn, Vr = x[0], x[1], x[2], x[3]
    a, b, c, d = w[0], w[1], w[2], w[3]
    
    y_bar = s* tf.math.pow(fm, a) * tf.math.pow(Zn, b) * c / tf.math.pow(Vr, d)
    return tfd.Normal(loc=y_bar, scale=tf.ones_like(y_bar))  # [n]

# Setup assumptions.
dtype = np.float32
num_samples = 500 # 100 is awful, 10 is fine
dims = 4
tf.compat.v1.random.set_random_seed(10014)
np.random.seed(10014)

params = np.array([1] * 4, dtype) 
y, x, _ = make_training_data(
    num_samples, dims, params)

log_sigma = tf.Variable(0., dtype=dtype, name='log_sigma')
mean = tf.Variable([1., 1., 1., 1.])

optimizer = tf.optimizers.SGD(learning_rate=0.0001)

prior_ = make_weights_prior(dims, mean, log_sigma)
def unnormalized_posterior_log_prob_test(w):
    likelihood = make_response_likelihood(w, x)
    return (
        prior_.log_prob(w) +
        tf.reduce_sum(likelihood.log_prob(y), axis=-1))  # [m]

@tf.function
def mcem_iter(weights_chain_start, step_size):
  with tf.GradientTape() as tape:
    tape.watch(log_sigma)
    prior = make_weights_prior(dims, mean, log_sigma)

    def unnormalized_posterior_log_prob(w):
      likelihood = make_response_likelihood(w, x)
      return (
          prior.log_prob(w) +
          tf.reduce_sum(likelihood.log_prob(y), axis=-1))  # [m]

    def trace_fn(_, pkr):
      return (
          pkr.inner_results.log_accept_ratio,
          pkr.inner_results.accepted_results.target_log_prob,
          pkr.inner_results.accepted_results.step_size)

    num_results = 2
    weights, (
        log_accept_ratio, target_log_prob, step_size) = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=0,
        current_state=weights_chain_start,
        kernel=tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_posterior_log_prob,
                num_leapfrog_steps=2,
                step_size=step_size,
                state_gradients_are_stopped=True,
            ),
            # Adapt for the entirety of the trajectory.
            num_adaptation_steps=2),
        trace_fn=trace_fn,
        parallel_iterations=1)

    # We do an optimization step to propagate `log_sigma` after two HMC
    # steps to propagate `weights`.
    loss = -tf.reduce_mean(target_log_prob)

  avg_acceptance_ratio = tf.math.exp(
      tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.)))

  optimizer.apply_gradients(
      [[tape.gradient(loss, log_sigma), log_sigma]])

  weights_prior_estimated_scale = tf.math.exp(log_sigma)
  return (weights_prior_estimated_scale, weights[-1], loss,
          step_size[-1], avg_acceptance_ratio)

num_iters = int(2000)

weights_prior_estimated_scale_ = np.zeros((num_iters), dtype)
weights_ = np.zeros([num_iters + 1, dims], dtype) # initial 
loss_ = np.zeros([num_iters], dtype)
weights_[0] = np.random.randn(dims).astype(dtype)
step_size_ = 0.0006 # 0.0001

unnormalized_posterior_log_prob_test(weights_[0])

for iter_ in range(num_iters):
    [
        weights_prior_estimated_scale_[iter_],
        weights_[iter_ + 1],
        loss_[iter_],
        step_size_,
        avg_acceptance_ratio_,
    ] = mcem_iter(weights_[iter_], step_size_)
    tf.compat.v1.logging.vlog(
        1, ('iter:{:>2}  loss:{: 9.3f}  scale:{:.3f}  '
            'step_size:{:.4f}  avg_acceptance_ratio:{:.4f}').format(
                iter_, loss_[iter_], weights_prior_estimated_scale_[iter_],
                step_size_, avg_acceptance_ratio_))

# Should converge to ~0.22.
import matplotlib.pyplot as plt

print(pd.DataFrame(weights_).describe(), f"avg_acceptance_ratio_:{avg_acceptance_ratio_}")

plt.plot(weights_prior_estimated_scale_)
plt.ylabel('weights_prior_estimated_scale')
plt.xlabel('iteration')

plt.show()