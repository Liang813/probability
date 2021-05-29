d = tfd.RelaxedBernoulli(0.01, probs=tf.ones(1)*.5)
iid = tfd.Independent(d, reinterpreted_batch_ndims=1)

iid.log_prob(iid.sample()) # <-- usually given nan

iid.log_prob(d.sample()) # always gives the right result
