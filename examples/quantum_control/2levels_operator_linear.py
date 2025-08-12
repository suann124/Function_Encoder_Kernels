import jax
jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp
from datasets import load_dataset
import equinox as eqx
import optax

# Load dataset
ds = load_dataset("QR12/2-level_system_control") 
ds = ds.with_format("jax")

# Define a simple neural network operator
class QuantumOperator(eqx.Module):
    layers: list
    
    def __init__(self, key):
        keys = random.split(key, 3)
        # Simple 3-layer network
        self.layers = [
            eqx.nn.Linear(1, 32, key=keys[0]),
            eqx.nn.Linear(32, 32, key=keys[1]), 
            eqx.nn.Linear(32, 1, key=keys[2])
        ]
    
    def __call__(self, x):
        # Forward pass through network
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply tanh activation to hidden layers
            if i < len(self.layers) - 1:
                x = jax.nn.tanh(x)
        return x

# Initialize model
rng = random.PRNGKey(0)
model = QuantumOperator(key=rng)

# Training function
@eqx.filter_value_and_grad
def compute_loss(model, point):
    # Get prediction
    Tf_pred = eqx.filter_vmap(model)(point["f"][:, None])
    # Mean squared error loss
    return optax.squared_error(point["Tf"][:, None], Tf_pred).mean()

# Optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# Training loop
def train_step(model, opt_state, point):
    loss, grads = compute_loss(model, point)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

# Train for n_epochs
n_epochs = 100
for epoch in range(n_epochs):
    for point in ds["train"].take(1000):
        model, opt_state, loss = train_step(model, opt_state, point)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Plot results
point = ds["train"].take(1)[0]

X = point["X"][:, None]
f = point["f"][:, None] 
Y = point["Y"][:, None]
Tf = point["Tf"][:, None]

idx = jnp.argsort(X, axis=0).flatten()
X = X[idx]
f = f[idx]

idx = jnp.argsort(Y, axis=0).flatten()
Y = Y[idx]
Tf = Tf[idx]

Tf_pred = eqx.filter_vmap(model)(f)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X, f, label="Input State")
ax.scatter(X, f, label="Data", color="red")
ax.plot(Y, Tf, label="True Output State")
ax.plot(Y, Tf_pred, label="Predicted Output State")

ax.legend()
plt.savefig('quantum_operator_output.png')
print("Saved plot to 'quantum_operator_output.png'")
plt.close()