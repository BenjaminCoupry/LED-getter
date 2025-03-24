import optax
import jax
import functools


def get_gradient_descent(optimizer, loss, iterations, projections = None, output=None, unroll=10, extra = False):
    def gradient_descent(parameters, optimizer, loss, iterations, projections, output, unroll, extra, **kwargs):
        if projections is not None:
            projector = lambda parameters : jax.tree.map(lambda parameter, projection : projection(parameter), parameters, projections)
        else:
            projector = lambda parameters : parameters
        conditionned_loss = lambda parameters : loss(projector(parameters), **kwargs)
        value_and_grad = jax.value_and_grad(conditionned_loss)
        def body_fun(i, val):
            opt_state, parameters, losses = val
            value, grad = value_and_grad(parameters)
            losses = losses.at[i].set(value)
            if extra:
                updates, opt_state = optimizer.update(grad, opt_state, parameters, value=value, grad=grad, value_fn=conditionned_loss)
            else:
                updates, opt_state = optimizer.update(grad, opt_state, parameters)
            parameters = optax.apply_updates(parameters, updates)
            parameters = projector(parameters)
            if output is not None:
                output(value, i, iterations)
            return (opt_state, parameters, losses)
        opt_state = optimizer.init(parameters)
        losses = jax.numpy.zeros((iterations,))
        _, parameters, losses = jax.lax.fori_loop(
            0,
            iterations,
            body_fun,
            (opt_state, parameters, losses),
            unroll=unroll,
        )
        return parameters, losses
    partial_gradient_descent = functools.partial(gradient_descent, optimizer=optimizer, loss=loss, iterations=iterations, projections=projections, output=output, unroll=unroll, extra=extra)
    return partial_gradient_descent

