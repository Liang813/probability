import tensorflow_probability as tfp
import tensorflow as tf
import traceback
try:
    tfd = tfp.distributions
    tfl = tf.linalg


    num_dims = 3
    num_timesteps = 1
    step_std = 1.0
    noise_std = 5.0

    ssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=tfl.LinearOperatorIdentity(num_dims),
        transition_noise=tfd.MultivariateNormalDiag(
            scale_diag=tf.fill([num_dims], tf.square(step_std))
        ),
        observation_matrix=tfl.LinearOperatorIdentity(num_dims),
        observation_noise=tfd.MultivariateNormalDiag(
            scale_diag=tf.fill([num_dims], tf.square(noise_std))
        ),
        initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=tf.ones([num_dims])),
    )

    sample = ssm.sample()
except Exception as e:
    traceback.print_exc(file=open('/script/probability417-buggy.txt','w+'))
    
