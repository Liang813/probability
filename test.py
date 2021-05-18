import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tf.reset_default_graph()

for target_shape in [1, 10, 100, 1000]:
    mean_vec = np.tile([0.0], target_shape)
    std_vec = np.tile([0.1], target_shape)
    dist_test = tfp.distributions.MultivariateNormalDiag(loc = mean_vec,
                                                         scale_diag = std_vec,
                                                         allow_nan_stats=False)

    with tf.Session() as sess:
        print(dist_test.prob(np.tile([1.0], target_shape)).eval())
        
