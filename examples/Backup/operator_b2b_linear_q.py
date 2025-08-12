import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

from datasets import load_dataset

import equinox as eqx
import optax

from function_encoder.losses import basis_normalization_loss
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.utils.training import fit

import matplotlib.pyplot as plt

# Load dataset

# ds = load_dataset("ajthor/derivative_polynomial")
ds = load_dataset("QR12/2-level_system_no_control")
ds = ds.with_format("jax")


# Create model

# rng = random.PRNGKey(0)
# source_key, target_key, operator_key = random.split(rng, 3)

# source_encoder = FunctionEncoder(
#     basis_size=8,
#     layer_sizes=(1, 32, 1),
#     activation_function=jax.nn.tanh,
#     key=source_key,
# )

# target_encoder = FunctionEncoder(
#     basis_size=8,
#     layer_sizes=(1, 32, 1),
#     activation_function=jax.nn.tanh,
#     key=target_key,
# )


# # Train


# # Train the source encoder.
# def source_loss_function(model, point):
#     coefficients = model.compute_coefficients(point["X"][:, None], point["f"][:, None])
#     f_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(
#         point["X"][:, None], coefficients
#     )
#     pred_loss = optax.squared_error(point["f"][:, None], f_pred).mean()
#     norm_loss = basis_normalization_loss(model.basis_functions, point["X"][:, None])
#     return pred_loss + norm_loss


# source_encoder = fit(source_encoder, ds["train"], source_loss_function)


# # Train the target encoder.
# def target_loss_function(model, point):
#     coefficients = model.compute_coefficients(point["Y"][:, None], point["Tf"][:, None])
#     Tf_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(
#         point["Y"][:, None], coefficients
#     )
#     pred_loss = optax.squared_error(point["Tf"][:, None], Tf_pred).mean()
#     norm_loss = basis_normalization_loss(model.basis_functions, point["X"][:, None])
#     return pred_loss + norm_loss


# target_encoder = fit(target_encoder, ds["train"], target_loss_function)

# Train the operator.
ds_subset = ds["train"].take(1000)

source_coefficients = ds_subset["X"]
# source_coefficients = eqx.filter_vmap(source_encoder.compute_coefficients)(
#     ds_subset["X"][:, :, None], ds_subset["f"][:, :, None]
# )

target_coefficients = ds_subset["Y"]
# target_coefficients = eqx.filter_vmap(target_encoder.compute_coefficients)(
#     ds_subset["Y"][:, :, None], ds_subset["Tf"][:, :, None]
# )

operator = jnp.linalg.lstsq(source_coefficients, target_coefficients)[0]


# Plot

point = ds["train"].take(1)[0]

X = point["X"][:, None]
# f = point["f"][:, None]
Y = point["Y"][:, None]
# Tf = point["Tf"][:, None]

idx = jnp.argsort(X, axis=0).flatten()
X = X[idx]
# f = f[idx]

idx = jnp.argsort(Y, axis=0).flatten()
Y = Y[idx]
# Tf = Tf[idx]

source_coefficients = X.T
# source_coefficients = source_encoder.compute_coefficients(X, f)
target_coefficients = jnp.dot(source_coefficients, operator)
print(source_coefficients.shape)
# Tf_pred = eqx.filter_vmap(target_encoder, in_axes=(eqx.if_array(0), None))(
#     Y, target_coefficients
# )

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.plot(X, f, label="Original")
# ax.scatter(X, f, label="Data", color="red")

# ax.plot(Y, Tf, label="True")
# ax.plot(Y, Tf_pred, label="Predicted")

# ax.legend()
# # plt.show()

# plt.savefig('operator_b2b_linear_output.png')
# print("Saved plot to 'operator_b2b_linear_output.png'")
# plt.close()  # clean up the figure

print(operator)