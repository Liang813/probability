from tensorflow_probability import distributions as tfd
import numpy as np

def get_random_mixture_pdf(float_type):
    N_predictions = int(1e2)
    N_components = 5
    N_grid_points = int(1e2)
    pi_sim = np.random.uniform(size=(N_predictions, N_components)).astype(float_type)
    pi_sim = pi_sim/np.transpose(np.tile(pi_sim.sum(axis=1), (N_components, 1)))
    mean_sim = np.random.uniform(low=5, high=10, size=(N_predictions, N_components)).astype(float_type)
    sd_sim = np.random.uniform(low=0.1, high=1, size=(N_predictions, N_components)).astype(float_type)
    mixture_distribution = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=pi_sim),
        components_distribution=tfd.Normal(
            loc=mean_sim,
            scale=sd_sim,
            validate_args=True,
            allow_nan_stats=False,
        )
    )

    grid = np.linspace(-10, 10, num=N_grid_points).astype(float_type)
    pdf = mixture_distribution.prob(grid)
    return pdf
  
pdf = get_random_mixture_pdf(float_type='float32')  # works
pdf = get_random_mixture_pdf(float_type='float64')  #InvalidArgumentError
