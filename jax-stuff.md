---
layout: distill
title: "Programming TPUs in JAX"
# permalink: /main/
description: "How to use JAX to program TPUs efficiently! Much of this section is taken from <a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>here</a>."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 10

previous_section_url: "../profiling"
previous_section_name: "Part 9: Profiling"

next_section_url: ../conclusion
next_section_name: "Part 11: Conclusions"

giscus_comments: true

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Sholto Douglas
    url: "https://x.com/_sholtodouglas"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
    url: "https://x.com/pchoy95"
  - name: Vinay Ramasesh
    url: "https://x.com/vinayramasesh"
  - name: Albert Webson
    url: "https://representation.ai/"
  - name: Yash Katariya
    url: https://x.com/yashk2810
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "How Does Parallelism Work in JAX?"
  - subsections:
    - name: "Auto sharding mode"
    - name: “Explicit sharding mode”
    - name: "Manual sharding mode via shard_map"
  - name: "Worked Problems"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## How Does Parallelism Work in JAX?

JAX supports three levels of parallelism for multi-device programming:

1. **Compiler, take the wheel!** You write normal, single device JAX code, and the compiler automatically shards data and partitions computation for you, no matter how many devices you're running on.
2. **JAX, take the wheel!** Automatic parallelism is great, but sometimes the compiler does something crazy. Explicit sharding lets you write single-device code like usual, but have JAX handle sharding propagation (not the compiler). This means JAX can ask you for clarification when it's unclear what you want.
3. **Just let me write what I mean, damnit!** While compilers are nice, they sometimes do the wrong thing and add communication you don't intend. Sometimes we want to be explicit about what communication collectives we want to write.

| Mode | View? | Explicit sharding? | Explicit Collectives? |
|---|---|---|---|
| Auto | Global | ❌ | ❌ |
| Explicit | Global | ✅ | ❌ |
| Manual | Per-device | ✅ | ✅ |

Correspondingly, JAX provides APIs for each of these modes:

1. `jax.jit` + `Auto` axes lets you specify the sharding of the inputs and outputs to a program (via `in_shardings` and `out_shardings`) and infers the rest using the [Shardy](https://openxla.org/shardy) compiler. While it isn't perfect, it usually does a decent job at automatically scaling your program to any number of chips.
2. `jax.jit` + `Explicit` axes will let JAX's sharding propagation decide how the intermediates and outputs should be sharded. JAX will error out when it detects ambiguous communication and lets the user resolve it.
3. `jax.shard_map` is the more manual counterpart. You get a device-local view of the program and have to write any communication you want explicitly. Have a sharded array and want the whole thing on each device? Add a `jax.lax.all_gather`. Want to sum an array across your devices? Add a `jax.lax.psum` (an AllReduce). Programming is harder but far less likely to do something you don't want.

<h3 id="auto-sharding-mode">Auto sharding mode</h3>

jax.jit plays two roles inside JAX. As the name suggests, it "just-in-time" compiles a function from Python into bytecode (via XLA/HLO/LLO) so it runs faster. But if the input is sharded or the user specifies an `in_sharding` or `out_sharding`, it also lets XLA distribute the computation across multiple devices and add communication as needed. For example, here's how you could write a sharded matmul using jax.jit:

```py
import jax
import jax.numpy as jnp

# Running on an TPU v5e 2x2. This assigns names to the two physical axes of the hardware.
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'))

# This tells JAX to use this mesh for all operations, so you can just specify the PartitionSpec P.
jax.set_mesh(mesh)

# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# We can explicitly compile the sharded matmul function here. This adds all the
# necessary comms (e.g. an AllReduce after the matmul).
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

This will run automatically with any sharding and partition the computation across our devices. **But what's actually happening at the hardware level?**

1. First we create In and W sharded across our devices<d-footnote>Notice how we did this.  This is one way to create an array with a particular sharding (i.e. by adding the device argument to the creation function). Another one is to create an array normally with `jnp.array(....)` and then do e.g. `jax.device_put(..., P('x', 'y'))`.  Yet another is to write a function which creates the array you want, and jit-compile it with `out_shardings` being what you want.</d-footnote>. W is sharded 2 way along the contracting dimension, while In is sharded 4-ways (along both the contracting and output dimensions). This corresponds to a sharding W[D<sub>X</sub>, F] and In[B<sub>X</sub>, D<sub>Y</sub>], aka a kind of model and data parallelism. 
2. If we were running this locally (i.e. on one device), `matmul_square` would simply square the input and perform a simple matmul. But because we specify the `out_shardings` as `P('X', None)`, the output will be sharded along the batch but replicated across the model dimension and will require an AllReduce to compute.

Using our notation from previous sections, this will likely do something like

1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] 
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F] { U<sub>Y</sub> })

`jax.jit` will add this for us automatically! We can actually print the HLO with `jit_matmul.as_text()` and see the following HLO (abbreviated dramatically):

```py
# This fusion is the actual matmul of the sharded inputs and matrix
%fusion = bf16[4,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[4,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# We reduce the partially summed results across devices
ROOT %AllReduce = bf16[4,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[4,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

We can see the matmul (the fusion) and the AllReduce above. Pay particular attention to the shapes. `bf16[4, 1024]` is a local view of the activations, since our `batch_size=8` is split across 2 devices and our `d_model=2048` is likewise split 2 ways.

**This is pretty magical!** No matter how complicated our program is, Shardy and jit will attempt to find shardings for all the intermediate activations and add communication as needed. With that said, Shardy has its flaws. It can make mistakes. Sometimes you'll look at a profile and notice something has gone wrong. A giant AllGather takes up 80% of the profile, where it doesn't need to. When this happens, we can try to correct the compiler by explicitly annotating intermediate tensors with `jax.lax.with_sharding_constraint`. For instance, with two matmuls I can force the intermediate activations to be sharded along the `y` dimension (not that this is a good idea) with the following:

```py
import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X', 'Y'))

def matmul(x, Win, Wout):
  hidden = jnp.einsum('bd,df->bf', x, Win)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('x', 'y'))
  return jnp.einsum('bf,df->bd', hidden, Wout)
```

This makes up like 60% of JAX parallel programming in the automatic partitioning world where you control the intermediate shardings via `jax.lax.with_sharding_constraint`. But "compiler tickling" is famously not a fun programming model. You could annotate every intermediate variable and still not know if you'll get the right outcome. Instead, what if JAX itself could handle and control sharding propagation?

<h3 id="explicit-sharding-mode">Explicit sharding mode</h3>

Explicit sharding (or “sharding in types”) is the mode where sharding propagation happens at the JAX level at trace time. Each JAX operation has a sharding rule that takes the shardings of the op's arguments and produces a sharding for the op's result. For most operations these rules are simple and obvious because there's only one reasonable choice (e.g. elementwise ops retain the same sharding). But for some operations it's ambiguous how to shard the result in which case JAX throws a trace-time error and we ask the programmer to provide an `out_sharding` argument explicitly (e.g. jnp.einsum, jnp.reshape, etc).

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

# Running on an TPU v5e 2x2. This assigns names to the two physical axes of the hardware.
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'),
                                       axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))

# This tells JAX to use this mesh for all operations, so you can just specify the PartitionSpec P.
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16).reshape(8, 2), P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # bfloat16[8@X,2@Y]
  out = x * 2
  print(jax.typeof(out))  # bfloat16[8@X,2@Y]
  return out

f(x)
```

As you can see, JAX propagated the sharding from input (`x`) to output (`x`) which are inspectable at trace-time via `jax.typeof`. Let's see another example where you have conflicts:

```py
# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # This will error
```

This code errors with `Contracting dimensions are sharded and it is ambiguous how the output should be sharded. Please specify the output sharding via the `out_sharding` parameter. Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)`

This is awesome because how the output of einsum should be sharded is ambiguous. The output sharding can be:
* P('X', 'Y') which will induce a reduce-scatter or
* P('X', None) which will induce an all-reduce

Unlike Auto mode, explicit mode errors out when it detects ambiguous communication and requires the users to resolve it. So here you can do:

```py
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

Auto mode and Explicit mode can be composed via `jax.sharding.auto_axes` and `jax.sharding.explicit_axes` APIs. This is a [great doc to read](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html) for more information.

<h3 id="manual-sharding-mode-via-shard_map">shard_map: explicit parallelism control over a program</h3>

While Shardy is the "compiler take the wheel" mode, jax [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) puts everything in your hands. You specify the sharding of the inputs, like in jax.jit, but then you write all communication explicitly. Whereas `jax.jit` leaves you with a global cross-device view of the program, `shard_map` gives you a local per-device view.

Here's an example. Try to reason about what this function does:<d-footnote>If you want to play with this yourself in a colab by emulating a mesh, you can do so using the following cell `import jax; jax.config.update('jax_num_cpu_devices', 8)`</d-footnote>

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

mesh = jax.make_mesh((2, 4), ('x', 'y'), (shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=P(('x', 'y')))

# This function will operate on 1/8th of the array.
@jax.shard_map(in_specs=P(('x', 'y')), out_specs=P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
assert out.shape == (4,)
```

**What does this do?** `slice_and_average` is run on each TPU with 1/8th of the array, from which we slice the first 4 elements and average them across the full mesh. This means we're effectively doing `mean(x[:4], x[64:68], x[128:132], …)`. This is pretty cool, because that's not an easy operation to express in JAX otherwise.

**Why do this instead of jax.jit?** If we'd used `jax.jit`, `slice_and_average` would have seen a global view of the array (the full `[512,]` array). We'd have had to slice out this non-uniform slice and then perform an average which XLA would have had to interpret correctly. XLA might have added the wrong communication or gotten confused. Here we see the local view and write only the communication we need.

**Example [Collective Matmul]:** To take a more realistic example, say we to implement model parallelism where the activations are initially model sharded, i.e. A[B<sub>X</sub>, D<sub>Y</sub>] \* W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]. Naively, we would do this by AllGathering A first followed by a local matrix multiplication:

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>]) 
2. Out[B<sub>X</sub>, F<sub>Y</sub>] = A[B<sub>X</sub>, D] *<sub>D</sub> W[D, F<sub>Y</sub>]

Sadly, this is bad because it doesn't allow us to overlap the communication with the computation. Overlapping them can be done with a "collective matmul", as described in [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959). The algorithm is basically as follows:

* For each Y shard, perform a matmul of the local chunk of A with the local chunk of W, producing a result of shape `[B / X, F / Y]`. Simultaneously, permute A so you get the next chunk locally, perform the matmul, and sum the result.

We can implement that quite easily with shard\_map:

```py
import functools

import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np

mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=('X', 'Y'),
                                       axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, jax.P('X', 'Y'))
W = jax.device_put(W, jax.P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=jax.P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs is the looped operand; rhs is the local operand
  axis_size = jax.lax.axis_size('Y')  # axis_size = 4 for this example
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # Matmul for a chunk
    update = lhs @ rhs_chunk
    # Circular shift to the left
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pvary(accum, ('X', 'Y'))
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # Compute the last chunk after the final permute to leave lhs in the state we found it
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(jax.shard_map(
  collective_matmul_allgather_lhs_contracting,
  in_specs=(jax.P('X', 'Y'), jax.P(None, 'Y')), out_specs=jax.P('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

This is pretty neat! We can benchmark this and see that it's also a lot faster! [Here's](https://imgur.com/a/e9I6SrM) the profile with the default jit matmul which takes 311us with a big blocking AllGather at the beginning:

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}

And [here's](https://imgur.com/a/21iy0Sv) the version above that takes 244 us. You can see the profile doesn't have the AllGather. It's all useful work! Our FLOPs utilization is also a lot higher. 

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}

It's also worth noting that the matmul time with no sharding on the contracting dimension is [224us](https://imgur.com/a/i3gNKfq), so we're remarkably close to the unsharded baseline here. This is a good example of the kind of performance engineering you might end up doing to improve TPU utilization. For more `shard_map` examples, [this note is great](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side).

Now here are a couple of useful worked problems to try and implement using `jax.jit` or `shard_map`!

## Worked Problems

Here are some random JAX-related problems. I'll add some more later. For all of these, you'll need some number of TPUs in a Colab. You can use a public Colab with TPUv2-8. From now on, we'll assume you have N devices available.

**Problem 1:** Let **A** be an array of activations of shape float32[S<sub>X</sub>, D<sub>Y</sub>] with `X * Y = N`. Do the following:

1. Write a function in JAX that computes the average within each `(X, Y)` shard, i.e. it returns an array of size [X, Y] where `arr[i, j]` is the average over shard `(i, j)`. Do this with both `jax.jit` and `shard_map`. Profile each and see how long they took. Was there any communication added? *Hint: there shouldn't be, but sometimes XLA adds it anyway.*

2. Write a function in JAX that returns roll(x, shift, axis=0) - x for some shift **within each shard X**. I'm not enough of a masochist to make you do this in jax.jit, so just do this with `shard_map`.

{% details Click here for the answer. %}

Part 1: Here is a solution to part 1. Note the fairly complex reshapes we have to do for the `jax.jit` solution.
 
```py
import numpy as np

import jax
import jax.numpy as jnp

P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((4, 2), ('X','Y'))

average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True), 
    mesh=mesh, 
    in_specs=P('X','Y'), out_specs=P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

Part 2: Here is a similar solution to Part 2.

```py
import numpy as np

import jax
import jax.numpy as jnp

import functools

P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((4, 2), ('X','Y'))

def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0), 
      mesh=mesh, 
      in_specs=P('X','Y'), out_specs=P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return jnp.roll(reshaped, shift, axis=1).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

{% enddetails %}

**Problem 2:** Here we'll make a basic "mixture of experts" model together. Let **W**: float32[E<sub>X</sub>, D, F<sub>Y</sub>] be a set of E "expert" matrices. Let **A**: float32[S<sub>X</sub>, D<sub>Y</sub>] (our activations) and let **B** be a set of "routing assignments" where B[i] is an integer in the range `[0, E)` telling us which matrix we want to process that activation. We want to write a function in JAX that returns `Out[i] = W[B[i]] @ A[i]`.

1. Let's start by ignoring sharding altogether. Make all of these tensors small enough so they fit in one device. Write a local implementation of this function. *Make sure you don't materialize an array of shape `[S, D, F]`! Hint: try sorting the tokens into a new buffer of shape `[E, S, D]` with some attention to masking (why do we need the second dimension to have size S?).*

2. If you just `jax.jit` the above method, something will happen. Profile this and see what communication it decided to do. How long does it take?

3. One problem you'll notice with the above is that it likely gathers the full set of activations **A** locally, i.e. AllGather<sub>X</sub>([S<sub>X</sub>, D<sub>Y</sub>]), Not only is this expensive communication-wise, it's also incredibly expensive memory-wise if we can't fit the full set of activations locally. Implement the above using `shard_map` and explicit communication.

      1. For a first pass, it might be easiest to use a `jax.lax.all_gather` and reorder as in (a).  

      2. For a second pass, try to avoid materializing any array of size `[E, S, D]`, i.e. try to perform the computation in a ragged fashion using a `jax.lax.all_to_all` inside a `jax.lax.while_loop`. This way, you can avoid materializing the full activations and wasting compute on padding. How much faster is this than your original implementation?  

4. Most MoEs route to multiple (k) experts and then average the result. Refactor the above to implement this. Let **B**: int32[S, k] in this case for the k experts to route to.

**Problem 3:** The collective matmul example above is actually super relevant for real LLMs. Let's tweak the example to do the full Transformer stack.

1. As an exercise, let's start by implementing an AllReduce collective matmul, i.e. A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]. Note that the output isn't replicated. The naive algorithm is discussed above, basically just a local matmul followed by an AllReduce. Try to make a comms overlapped "collective" version of this operation. *Hint: tile over the output dimension and feel free to use `jax.lax.psum` (aka AllReduce).* *Note: due to the way XLA handles this, it may not actually be faster than the baseline.*

2. The complement to the AllReduce collective matmul above is a ReduceScatter collective matmul, as in Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]. This occurs in the down-projection matrix in a Transformer. Implement a collective, overlapped version of this in JAX. Be careful about passing only the minimal amount of data you need. *Hint: try permuting the result as you accumulate it.*

3. Put these two together into an end-to-end Transformer block that performs In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>] with overlapped communication.<d-footnote>As before, we can't do $W_{in} \cdot W_{out}$ first because of a non-linearity we've omitted here.</d-footnote> How much faster is this than a `jax.jit` implementation?

**Problem 4:** All of the collective matmuls implemented above are unidirectional: they only permute in one direction. Rewrite the collective AllReduce matmul and the collective ReduceScatter matmuls to use bidirectional communication. How much faster are these?

### That's all for Part 10. That's basically it! For final conclusions and further reading, click [here](../conclusion).
