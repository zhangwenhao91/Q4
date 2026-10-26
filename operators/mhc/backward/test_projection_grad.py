"""Finite difference check of the actual projection-gradient helper (no NPU/torch)."""
import unittest
import numpy as np
from projection_grad import projection_gradients


class ProjectionGradientTest(unittest.TestCase):
    def test_two_branches(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(2,3));phi=rng.normal(size=(4,3))
        gm=rng.normal(size=(2,4));gs=rng.normal(size=(2,1))
        gx,gp=projection_gradients(x,phi,gm,gs)
        def loss(): return ((x@phi.T)*gm).sum()+(x*x*gs).sum()
        for arr,grad in [(x,gx),(phi,gp)]:
            numeric=np.empty_like(arr)
            for index in np.ndindex(arr.shape):
                old=arr[index];h=1e-6
                arr[index]=old+h;plus=loss()
                arr[index]=old-h;minus=loss()
                arr[index]=old;numeric[index]=(plus-minus)/(2*h)
            np.testing.assert_allclose(grad,numeric,rtol=1e-7,atol=1e-8)


if __name__=='__main__': unittest.main()
