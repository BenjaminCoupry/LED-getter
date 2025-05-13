import jax.export
import numpy


a = numpy.load('/media/bcoupry/T7 Shield/Chauvet_1203_matin/lights/PS_02/LED_SH/values.npz')

def f1(x, y): # x: f32[a, 1], y : f32[a, 4]
    return x + y




my_scope = jax.export.SymbolicScope()

c = jax.export.symbolic_shape("None,1", scope=my_scope)

d = jax.export.symbolic_shape("None,a", scope=my_scope)

args_specs = (jax.ShapeDtypeStruct(c, dtype=jax.numpy.float32), jax.ShapeDtypeStruct(d, dtype=jax.numpy.float32))

exported = jax.export.export(jax.jit(f1))(* args_specs)

serialized: bytearray = exported.serialize()

rehydrated_exp = jax.export.deserialize(serialized).call

print(rehydrated_exp(jax.numpy.ones((1,)),jax.numpy.ones((2,))))


import re
import numpy as np
import jax
from jax import export

def f(x): return 2 * x * x


#TODO : tester si exporter une fonction elimine bien les arguments qui lui sont donnés dont elle n'a pas besoin
#Verifier aussi que les abstract shapes se correspondent entre elles, notemment les ...


exported: export.Exported = export.export(jax.jit(f))(
   jax.ShapeDtypeStruct((), np.float32))

# You can inspect the Exported object
exported.fun_name

exported.in_avals

print(re.search(r".*@main.*", exported.mlir_module()).group(0))

# And you can serialize the Exported to a bytearray.
serialized: bytearray = exported.serialize()


with open("exported_func.jax", "wb") as f:
    f.write(serialized)

with open("exported_func.jax", "rb") as f:
    serialized = f.read()

# The serialized function can later be rehydrated and called from
# another JAX computation, possibly in another process.
rehydrated_exp: export.Exported = export.deserialize(serialized)
rehydrated_exp.in_avals

def callee(y):
 return 3. * rehydrated_exp.call(y * 4.)

callee(1.)
