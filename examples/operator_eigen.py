import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

import equinox as eqx
import optax

from datasets import load_dataset


from function_encoder.losses import basis_orthogonality_loss
from function_encoder.operator_encoder import EigenOperatorEncoder
from function_encoder.utils.training import fit

import matplotlib.pyplot as plt

# Load dataset

ds = load_dataset("ajthor/derivative_polynomial")
ds = ds.with_format("jax")


# Create model

rng = random.PRNGKey(0)
rng, key = random.split(rng)

model = EigenOperatorEncoder(
    basis_size=8,
    layer_sizes=(1, 64, 64, 1),
    activation_function=jax.nn.tanh,
    key=key,
)

# Train


def loss_function(model, point):
    coefficients = model.compute_coefficients(point["X"][:, None], point["f"][:, None])
    Tf_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(
        point["Y"][:, None], coefficients
    )
    pred_loss = optax.squared_error(Tf_pred, point["Tf"][:, None]).mean()
    orth_loss = basis_orthogonality_loss(
        model.function_encoder.basis_functions, point["X"][:, None]
    )
    return pred_loss + orth_loss


model = fit(model, ds["train"], loss_function)


# Plot

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

coefficients = model.compute_coefficients(X, f)
Tf_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(Y, coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X, f, label="Original")
ax.scatter(X, f, label="Data", color="red")

ax.plot(Y, Tf, label="True")
ax.plot(Y, Tf_pred, label="Predicted")

plt.legend()
plt.show()
