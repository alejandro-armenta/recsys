import jax.numpy as jnp
import numpy as np

"""
x=np.array([1,2,3], dtype=np.float32)

x[0] = 4
print(x)

x_jax=jnp.array(x)

print(x_jax)
"""

x=jnp.array([[1,2,3],[4,5,6],[7,8,9]],dtype=jnp.int32)

print(x)

#print(x[-1])

#print(x[:,1])

print(x[::2,::2])