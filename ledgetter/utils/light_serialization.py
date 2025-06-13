import jax.export
import ledgetter.utils.functions as functions

def serialize_light(light, values):
    lambda_expression = lambda points, pixels : light(**(values | {'points':points, 'pixels':pixels}))
    my_scope = jax.export.SymbolicScope()
    s1 = jax.export.symbolic_shape("batch,3", scope=my_scope)
    s2 = jax.export.symbolic_shape("batch,2", scope=my_scope)
    args_specs = (jax.ShapeDtypeStruct(s1, dtype=jax.numpy.float32), jax.ShapeDtypeStruct(s2, dtype=jax.numpy.int32))
    exported = jax.export.export(jax.jit(lambda_expression), platforms=['cuda','cpu'])(* args_specs)
    serialized = exported.serialize()
    return serialized

def deserialize_light(serialized):
    light = functions.filter_args(lambda points, pixels : jax.export.deserialize(serialized).call(points, pixels))
    return light