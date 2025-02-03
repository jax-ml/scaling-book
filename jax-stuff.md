---
layout: distill
title: "Programming TPUs in JAX"
# permalink: /main/
description: "How to use JAX to program TPUs efficiently! Much of this section is taken from <a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>here</a>."
date: 2025-02-03
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
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
  - name: Vinay Ramasesh
  - name: Albert Webson
    url: "https://representation.ai/"

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "How Does Parallelism Work in JAX?"
  - subsections:
    - name: "jax.jit: the automatic parallelism solution"
    - name: "shard_map: explicit parallelism control over a program"
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

JAX supports two schools of thought for multi-device programming:

1. **Compiler, take the wheel!** Let the compiler automatically partition arrays and decide what communication to add to facilitate a given program. This lets you write a program on a single device and automatically run it on hundreds without changing anything. 
2. **Just let me write what I mean, damnit!** While compilers are nice, they sometimes do the wrong thing and add communication you don't intend. Sometimes we want to be extremely explicit about what we're doing.

Correspondingly, JAX provides two APIs for each of these schools: **jit** (`jax.jit`) and **shard\_map** (`jax.experimental.shard_map.shard_map`).

1. `jax.jit` lets you specify the sharding of the inputs and outputs to a program (via `in_shardings` and `out_shardings`) and infers the rest using the [GSPMD](https://arxiv.org/abs/2105.04663) compiler. While it isn't perfect, it usually does a decent job at automatically scaling your program to any number of chips.
2. `jax.experimental.shard_map.shard_map` is the more explicit counterpart. You get a device-local view of the program and have to write any communication you want explicitly. Have a sharded array and want the whole thing on each device? Add a `jax.lax.all_gather`. Want to sum an array across your devices? Add a `jax.lax.psum` (an AllReduce). Programming is harder but far less likely to do something you don't want.

<h3 id="jax-jit-the-automatic-parallelism-solution">jax.jit: the automatic parallelism solution</h3>

jax.jit plays two roles inside JAX. As the name suggests, it "just-in-time" compiles a function from Python into bytecode (via XLA/HLO/LLO) so it runs faster. But if the input is sharded or the user specifies an `in_sharding` or `out_sharding`, it also lets XLA distribute the computation across multiple devices and add communication as needed. For example, here's how you could write a sharded matmul using jax.jit:

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

# Running on a VLP=2x2. This assigns names to the two physical axes of the hardware.
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'))
def P(*args):
  return shd.NamedSharding(mesh, shd.PartitionSpec(*args))

# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P('Y', None))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# We can explicitly compile the sharded matmul function here. This adds all the
# necessary comms (e.g. an AllReduce after the matmul).
jit_matmul = jax.jit(matmul_square, out_shardings=P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

This will run automatically with any sharding and partition the computation across our devices. **But what's actually happening at the hardware level?**

1. First we create In and W sharded across our devices<d-footnote>Notice how we did this.  This is one way to create an array with a particular sharding (i.e. by adding the device argument to the creation function). Another one is to create an array normally with `jnp.array(....)` and then do e.g. `jax.device_put(..., P(‘x', ‘y'))`.  Yet another is to write a function which creates the array you want, and jit-compile it with `out_shardings` being what you want.</d-footnote>. W is sharded 2 way along the contracting dimension, while In is sharded 4-ways (along both the contracting and output dimensions). This corresponds to a sharding W[D<sub>X</sub>, F] and In[B<sub>X</sub>, D<sub>Y</sub>], aka a kind of model and data parallelism. 
2. If we were running this locally (i.e. on one device), `matmul_square` would simply square the input and perform a simple matmul. But because we specify the `out_shardings` as `P(‘X', None)`, our output will be sharded along the batch but replicated across the model dimension and will require an AllReduce to compute.

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

**This is pretty magical!** No matter how complicated our program is, GSPMD and jit will attempt to find shardings for all the intermediate activations and add communication as needed. With that said, GSPMD has its flaws. It can make mistakes. Sometimes you'll look at a profile and notice something has gone wrong. A giant AllGather takes up 80% of the profile, where it doesn't need to. When this happens, we can try to correct the compiler by explicitly annotating intermediate tensors with `jax.lax.with_sharding_constraint`. For instance, with two matmuls I can force the intermediate activations to be sharded along the `y` dimension (not that this is a good idea) with the following:

```py
import jax
import jax.numpy as jnp

def matmul(Wout, Win, x):
  hidden = jnp.einsum('fd,bd->bf', Win, x)
  hidden = jax.lax.with_sharding_constraint(hidden, P('x', 'y'))
  return jnp.einsum('df,bf->bd', Wout, hidden)
```

This makes up like 60% of JAX parallel programming in the jit world, since it's our only way of intervening with the compiler. It's worth playing around with `with_sharding_constraint` in a Colab and getting a sense for how it works. When we write LLMs using `jax.jit`, 90% of what we do to control shardings is changing the input and output shardings (via `in_shardings` and `out_shardings`) and annotating intermediate tensors with `with_sharding_constraint` to ensure the correct comms are happening. For more jax.jit examples, [this is a great doc to read](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).

<h3 id="shard-map-explicit-parallelism-control-over-a-program">shard_map: explicit parallelism control over a program</h3>

While GSPMD is the "compiler take the wheel" mode, jax [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) puts everything in your hands. You specify the sharding of the inputs, like in jax.jit, but then you write all communication explicitly. Whereas `jax.jit` leaves you with a global cross-device view of the program, `shard_map` gives you a local per-device view.

Here's an example. Try to reason about what this function does:<d-footnote>If you want to play with this yourself in a colab by emulating a mesh, you can do so using the following cell `import os; os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'`</d-footnote>

```py
import jax
import jax.numpy as jnp
import jax.lax
import jax.sharding as shd

from jax.experimental.shard_map import shard_map as shmap

P = shd.PartitionSpec
mesh = jax.make_mesh(axis_shapes=(2,4), axis_names=('x','y'))

x = jnp.arange(0, 512, dtype=jnp.int32, device=jax.NamedSharding(mesh, P(('x', 'y'))))

# This function will operate on 1/8th of the array.
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = shmap(slice_and_average, mesh, in_specs=P(('x', 'y')), out_specs=P(None,))(x)
assert out.shape == (4,)
```

**What does this do?** `slice_and_average` is run on each TPU with 1/8th of the array, from which we slice the first 4 elements and average them across the full mesh. This means we're effectively doing `mean(x[:4], x[64:68], x[128:132], …)`. This is pretty cool, because that's not an easy operation to express in JAX otherwise.

**Why do this instead of jax.jit?** If we'd used `jax.jit`, `slice_and_average` would have seen a global view of the array (the full `[512,]` array). We'd have had to slice out this non-uniform slice and then perform an average which XLA would have had to interpret correctly. XLA might have added the wrong communication or gotten confused. Here we see the local view and write only the communication we need.

**Example \[Collective Matmul\]:** To take a more realistic example, say we to implement model parallelism where the activations are initially model sharded, i.e. A[B<sub>X</sub>, D<sub>Y</sub>] \* W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]. Naively, we would do this by AllGathering A first followed by a local matrix multiplication:

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

from jax.experimental.shard_map import shard_map

mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=('X', 'Y'))
def P(*args):
  return shd.NamedSharding(mesh, shd.PartitionSpec(*args))

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, P('X', 'Y'))
W = jax.device_put(W, P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs is the looped operand; rhs is the local operand
  axis_size = jax.lax.psum(1, axis_name='Y')  # axis_size = 4 for this example
 
  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, i * chunk_size, chunk_size)
    # matmul for a chunk
    update = lhs @ rhs_chunk
    # circular shift to the left
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # compute the last chunk after the final permute,
  # to leave lhs in the state we found it
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, i * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(shard_map(
  collective_matmul_allgather_lhs_contracting, mesh,
  in_specs=(shd.PartitionSpec('X', 'Y'), shd.PartitionSpec(None, 'Y')), out_specs=shd.PartitionSpec('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

This is pretty neat! We can benchmark this and see that it's also a lot faster\! [Here's](https://imgur.com/a/e9I6SrM) the profile with the default jit matmul which takes 311us with a big blocking AllGather at the beginning:

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}

And [here's](https://imgur.com/a/21iy0Sv) the version above that takes 244 us. You can see the profile doesn't have the AllGather. It's all useful work! Our FLOPs utilization is also a lot higher. 

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}

It's also worth noting that the matmul time with no sharding on the contracting dimension is [224us](https://imgur.com/a/i3gNKfq), so we're remarkably close to the unsharded baseline here. This is a good example of the kind of performance engineering you might end up doing to improve TPU utilization. For more `shard_map` examples, [this note is great](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side).

Now here are a couple of useful worked problems to try and implement using `jax.jit` or `shard_map`!

## Worked Problems

Here are some random JAX-related problems. I'll add some more later. For all of these, you'll need some number of TPUs in a Colab. You can use a public Colab with TPUv3 or launch your own internal Colab. From now on, we'll assume you have N devices available.

**Problem 1:** For the next several parts, we'll let **A** be an array of activations of shape float32[S<sub>X</sub>, D<sub>Y</sub>] with `X * Y = N`. Do the following:

1. Write a function in JAX that computes the average over each `X` shard, i.e. it returns an array of size [X, D<sub>Y</sub>] where `arr[i]` is the average over shard `i`. Do this with both `jax.jit` and `shard_map`. Profile each and see how long they took. Was there any communication added? *Hint: there shouldn't be, but sometimes XLA adds it anyway.* [*Here's the answer.*](https://pastecode.io/s/0v603d9o)

2. Write a function in JAX that returns roll(x, shift) - x for some shift **within each shard X**. I'm not enough of a masochist to make you do this in jax.jit, so just do this with `shard_map`.

**Problem 2:** Here we'll make a basic "mixture of experts" model together. Let **W**: float32[E<sub>X</sub>, D, F<sub>Y</sub>] be a set of E "expert" matrices. Let **A** be as above (our activations) and let **B** be a set of "routing assignments" where B[i] is an integer in the range `[0, E)` telling us which matrix we want to process that activation. We want to write a function in JAX that returns `Out[i] = W[B[i]] @ A[i]`.

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