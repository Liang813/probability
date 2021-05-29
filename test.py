import tensorflow_probability as tfp
tfd = tfp.distributions

rv = tfd.Multinomial(1000, probs=[0.7, 0.0, 0.3])
x = [489, 0, 511]
logp = rv.logp(x)
