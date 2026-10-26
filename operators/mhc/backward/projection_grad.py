"""Projection and squared-sum gradients; accepts torch or NumPy arrays."""


def projection_gradients(x, phi, grad_projection, grad_sqsum):
    return (grad_projection @ phi + 2 * grad_sqsum * x,
            grad_projection.T @ x)
