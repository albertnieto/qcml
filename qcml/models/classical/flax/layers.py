import flax.linen as nn


class KernelLayer(nn.Module):
    n_centers: int
    kernel_func: callable
    kernel_params: dict
    input_dim: int

    def setup(self):
        self.centers = self.param(
            "centers", nn.initializers.uniform(), (self.n_centers, self.input_dim)
        )

    def __call__(self, x):
        x_expanded = jnp.expand_dims(x, 1)  # (batch_size, 1, input_dim)
        centers_expanded = jnp.expand_dims(self.centers, 0)  # (1, n_centers, input_dim)
        return self.kernel_func(x_expanded, centers_expanded, **self.kernel_params)
