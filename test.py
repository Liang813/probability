import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions

d = tfd.RelaxedBernoulli(0.01, probs=tf.ones(1)*.5)
iid = tfd.Independent(d, reinterpreted_batch_ndims=1)

iid.log_prob(iid.sample()) # <-- usually given nan
print(iid.log_prob(iid.sample()))
iid.log_prob(d.sample()) # always gives the right result
print(iid.log_prob(d.sample()))
