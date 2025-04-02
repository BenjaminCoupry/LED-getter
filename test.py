import jax
import jax.numpy as jnp
from jax import lax

# Define a function that takes two arguments
def my_function(x, y):
    return x + y, x * y  # Returns a tuple of results

# Batched inputs
x_batch = jnp.array([1, 2, 3])  # Shape (3,)
y_batch = jnp.array([4, 5, 6])  # Shape (3,)

# Wrap the function to take a single argument (tuple of inputs)
def wrapped_func(args):
    x, y = args  # Unpack
    return my_function(x, y)  # Call the original function

# Use lax.map, passing a tuple of arrays
results = lax.map(wrapped_func, (x_batch, y_batch))

# Unpack results
sum_result, product_result = results

print("Sum result:", sum_result)          # [5, 7, 9]
print("Product result:", product_result)  # [4, 10, 18]
