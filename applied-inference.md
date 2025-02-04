---
layout: distill
title: "Serving LLaMA 3-70B on TPUs"
# permalink: /main/
description: "Let's take a close look at how we'd serve LLaMA 3-70B models on TPU v5e. How expensive are different models to serve at roofline? How large are their KV caches? What batch sizes should we use? How are the parameters and activations sharded during inference? Let's work through some back-of-the-envelope estimates for latency and throughput in production."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 8

previous_section_url: "../inference"
previous_section_name: "Part 7: Inference"

next_section_url: ../profiling
next_section_name: "Part 9: Profiling"

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
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "What's the LLaMA Serving Story?"
  - subsections:
    - name: "What about prefill?"
  - name: "Visualizing the Latency Throughput Tradeoff"
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

## What's the LLaMA Serving Story?

Let's remind ourselves what LLaMA 3-70B looks like (see [Section 6](../applied-training) for reference):

| **hyperparam**                            | **value** |
| ----------------------------------------- | :-------: |
| $$n_\text{layers}$$ (L)                   |    80     |
| $$d_\text{model}$$ (D)                    |   8,192   |
| $$\text{ffw}_\text{multiplier}$$ (F // D) |    3.5    |
| $$n_\text{heads}$$ (N)                    |    64     |
| $$n_\text{kv_heads}$$ (K)                 |     8     |
| $$d_\text{qkv}$$ (H)                      |    128    |
| $$n_\text{embeddings}$$ (V)               |  128,256  |

For serving, we mostly want to serve on TPU v5e since it is our current dedicated inference chip with the best performance / dollar (cost comes from [Google Cloud pricing](https://cloud.google.com/tpu/pricing)):

| **TPU type** | **bfloat16 FLOPs/s** | **Google Cloud USD / hour** | **FLOPs / $** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| v5p          |       4.59e14        |            $4.2             |    1.09e14    |
| v5e          |       1.97e14        |            $1.2             |  **1.64e14**  |

Let's start by thinking about a bulk inference regime, where our goal is to maximize throughput (samples / $). When we optimize for throughput, we want to be compute bound, meaning we come close to utilizing all the TPU MXU capacity. **Typically that means we want the batch size to be as large as possible, so we are doing as much work as possible.**

**Question:** How large do our batch sizes need to be for us to be compute-bound?

{% details Click here once you've thought it through! %}

As discussed in Section 1, for our big matmuls we have the simple rule:

$$\begin{equation}
\text{FFW blocks are compute bound when } B > 240
\end{equation}$$

assuming we use bfloat16 (or in general, if we store our parameters in the same dtype we use for computation). If we quantize our weights to int8 or int4 but do bfloat16 matmuls, this drops to 120 or 60, respectively. This holds for pretty much all models of interest.

**How realistic is this?** Larger batch sizes than this will still see some improvement in throughput because of imperfect overlapping, but this is a good heuristic. Note also that our assumption that communication will be perfectly overlapped with compute is not realistic, since XLA is quite fallible. In particular, XLA sometimes fails to overlap the ICI communication of our model-sharded matmuls with the FLOPs themselves, so we actually start taking a latency hit over $$BS=32$$.

**Caveat for attention:** During generation, we have to load our KV caches to perform attention with previous tokens. Especially with large sequence lengths and batch sizes, this KV loading can be more significant than weight loading. While this is more or less unavoidable since attention performs comparatively few FLOPs, the fraction of time spent on KV loading typically decreases with batch size so long as

$$\frac{\text{model FLOPs / token}}{\text{KV cache bytes per token}} > 240$$

which is typically true if $$DF / KH > 240$$ which is usually true. What this means is that we will see continued improvement in throughput with larger batch sizes well beyond 240.

{% enddetails %}

**Question:** How large are our KV caches? This determines how large our batch size can be on a given topology. You can assume we use int8 to store them.

{% details Click here once you've thought it through! %}

We have 8 KV heads for LLaMA 3-70B, so

$$\text{KV cache size per token in int8} = 2 * K * H * L = 2 * 8 * 128 * 80 = 160\text{kiB}$$

**Note just how big this is!** If we have a sequence length of 32k tokens (as is common), this uses `162e3 * 32,768 = 5.3GB / sequence`. For BS=240, this is 1.3TB! Since TPU v5e only have 16GB a piece, we would need about `(70e9 + 1.3e12) / 16e9 = 86` TPU v5e chips to even fit this much memory. Also note how large this is compared to the 70GB of model parameters.

{% enddetails %}

**Question:** What is the smallest slice we could serve on in bfloat16, int8, and int4 (both KVs and parameters)? You can assume batch size 1 and 8k context.

{% details Click here once you've thought it through! %}

This is easy! If we're OK with a tiny batch size then the only limit is fitting parameter memory in HBM, i.e. it is just `ceil(num_params * sizeof(dtype) / HBM per TPU`, or `ceil(70e9 * sizeof(dtype) / 16e9)` rounded to the nearest reasonable topology (some multiple of 2):

| dtype | param size | KV size / token (bytes) | min TPU v5es | actual min slice | remaining HBM for KV caches | num KV caches @ 8k |
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: |
| bf16  |   140GB    |          324kB          |     8.75     |  4x4 = 16 chips  |             116             |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8 chips   |             68              |         52         |
| int4  |    45GB    |          81kB           |     2.81     |  2x2 = 4 chips   |             19              |         67         |

That's pretty cool! It tells us we could fit LLaMA 70B on a TPU v5e 2x2 if we wanted to. Except you'll notice the number of KV caches is very small. That's our batch size! That means we'll be getting terrible FLOPs utilization. We'd be very happy to use a larger topology in order to push our batch size up to 240.

{% enddetails %}

**Question:** Assume we use the largest batch size that fits on these topologies, what latency we could expect for each generate step?

{% details Click here once you've thought it through! %}

This is also easy, since we're picking our batch size to fill up all our HBM! This is just a question of how long it takes to load a full TPU v5e's worth of bytes into the MXU. This is just `v5e HBM / v5e HBM memory bandwidth = 16GB / 8.2e11 = 19ms`, so this is **19ms / step**. Assuming our generations have a median length of 512 tokens, that is about 9s for each decode. Note that we could get marginally better latency with a smaller batch size, for instance if we only looked at model parameters in int4 our minimum latency is about 10ms / step, since HBM is no longer full.

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaway**: we can always lower bound decode latency by asking how long it takes to load all the model's parameters from HBM into the MXU. When our KV caches are small, you can think about each layer as just loading the weights chunk-by-chunk and then discarding them. Unless we're using large batch sizes or lots of inter-device comms, this is often a reasonable bound (within 1.5x). When our batch size is bigger, we need to model the KV cache loading as well, since that dominates the parameters.</p>

Likewise, in the FLOPs-bound regime (e.g. training or big-batch inference), we can use the $$\text{Total FLOPs} / (N \cdot C) = 2 \cdot \text{param count} \cdot B / (N \cdot C)$$ lower bound, which assumes no communication.

**Question:** For each of these, what throughput per chip does this give us (in terms of queries / chip)? This is often the quantity we care about because it's exactly correlated with cost / token. You can assume our median decode length is 512 tokens.

{% details Click here once you've thought it through! %}

With our assumption about median decode length, our throughput is just $$B / (\text{per-step latency} \cdot \text{median steps} \cdot N) \approxeq 43 / (0.019 * 512 * N)$$. This gives us roughly $$(4.42 / N)$$ QPS, so plugging in $$N$$ we get:

|  dtype   | QPS / chip |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.66    |
|   int4   |    1.72    |

Note that this is rather optimistic since it totally ignores the working memory of the forward pass (memory allocated to activations and attention). This is not ridiculous with Flash Attention, but it is also not realistic. The real numbers are likely maybe 1/2 of this. For absolutely maximum throughput we would probably want to more than double the number of chips and increase the batch size significantly as well.

{% enddetails %}

**Question:** What would happen if we doubled our topology for each of the above examples?

{% details Click here once you've thought it through! %}

If we used a 4x8 slice in bfloat16, we would have 186GB remaining for KV caches, which would let us up our batch size to 161. Then since our step time would remaining the same, we would have a throughput of `16.54 / num_chips`, or

|       dtype       | QPS / chip |
| :---------------: | :--------: |
| bfloat16 (on 4x8) |    0.51    |
|   int8 (on 4x4)   |    1.03    |
|   int4 (on 2x4)   |    2.06    |

A further increase would give an even bigger win! The big takeaway is that **the smallest topology is not the most performance topology** in all cases, if we're limited by KV cache size.

{% enddetails %}

**Question:** Can we actually serve on a 4x8? What sharding would we use for our model on a TPU v5e 4x8?

{% details Click here once you've thought it through! %}

As discussed in the previous section, we only really have two options for serving: (a) (pure) data parallelism and (b) model parallelism. FSDP is totally infeasible because of its high cost at small batch sizes (basically, moving weights or KV caches is very expensive compared to moving activations). The above calculations assume our model and caches are fully sharded, so we're implicitly assuming model parallelism. But then we have to ask: how much can we do before we become communication bound. As we've discussed in the previous section, our models become communication bound roughly when

$$Y > \frac{F \cdot n_\text{axes}}{2550}$$

although this will depend slightly on the dtype of the computation. For instance, above we have `F = 28,672`, so if we do 2 axes of model sharding this gives us roughly $$Y = 28672 \cdot 2 / 2550 = 22$$, so in general we could scale up to 16 chips, which lets us use a 4x4 but not a 4x8. Generally, since we do not perfectly overlap computation, even this estimate is overly optimistic. **Takeaway: we can't actually serve on a 4x8 with pure model parallelism.** The best we can do here is a 4x2 or _maybe_ a 4x4.

However, if we look at the int8 and int4 configs, we _can_ do those with pure model parallelism. So we've hit a point at which quantization actually gives us a meaningful advantage beyond faster FLOPs: it lets us use a larger batch size before we become comms-bound. **So the end of this story is that we can't actually serve on a 4x8 efficiently, but for the int8 and int4 configs we could do pure model parallelism*. Data parallelism is basically never useful because it replicates the parameters and we'd be better off with a smaller overall topology.

{% enddetails %}

<p markdown=1 class="takeaway">**Tip**: the maximum amount of useful model parallelism depends on $$d_{ff}$$ and the number of axes over which you're sharding your model. The maximum value usually ranges between 8 and 32 depending on the model size.</p>

### What about prefill?

Compared to generate, **it's much easier to be compute-bound during prefill**. That's because our "batch size" is just the input sequence length, which is often $$T>8000$$ tokens. You can think of generate as prefill but with a batch dimension instead of a sequence dimension. Assuming we do prefill and generation on separate servers, we don't have KV caches to worry about. So we can basically think of prefill as a totally separately, conceptually simple process that's typically compute bound at reasonable sequence lengths.

**Question:** How does prefill affect actual QPS numbers? What prefill:decode ratio should we use?

{% details Click here once you've thought it through! %}

We need to actually figure out how long prefill will take. At 8k tokens, we are solidly compute bound, so we just need to reason about FLOPs. Very roughly, we can just think about the MLP layers, which use $$\text{batch} \cdot 2 \cdot 3 \cdot d_\text{model}^2 \cdot \text{ffw}_\text{multiplier} \cdot n_\text{layers}$$ int8 OPs. Using BS=240, that is 26e12 FLOPs.

On a TPU v5e 4x2, this is 8ms minimum at 100% MFU. At best, we are probably getting half that, so 16ms is roughly reasonable. Including attention, I would guess this puts us somewhere around 20-30ms. Now this is roughly on the order of 1-2x the step latency. With a generate batch size of 240 and a median decode length of 512, that means roughly every 2 decode steps, one prefill finishes, so we could do 1:1 prefill:generate since prefill takes roughly `2 * generate`, but we only need to prefill once per generate step.

**Note**: with very large batch sizes during generation, we often end up needing many prefill steps per generate step. We can either achieve this with an interleaved server or by launching many prefill servers per-generate server.

{% enddetails %}

## Visualizing the Latency Throughput Tradeoff

Sticking with LLaMA 70B for a second, let's actually look at the latency and throughput for different batch sizes during generation. As we showed in the previous section for PaLM models, this gives us a Pareto frontier for throughput/latency. Let's assume 16-way tensor parallelism since that's a reasonable bound on what we can use while staying compute-bound in the MLP blocks. We'll use a TPU v5e 4x4 topology here. **The slider controls the sequence length so you can see the effect of larger KV caches.**

<div class="l-page">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

* **See how dramatic the trandeoff is between cost and latency.** At the cost of doubling per-token latency, we can achieve a roughly 100x reduction in per-token cost. Also, our latency can range anywhere from 5.5ms with low batch size to 20 ms with very large batches.
* Note how at 2k context the throughput effectively plateaus at around 1 token / ms / chip when it hits the BS 120 roofline (120 here because we do int8 weights but bf16 FLOPs). As the sequence length increases, however, we can no longer fit this batch size in memory, so we never hit the point of full saturation.
* Note how much higher the latency is at large batch sizes for the same throughput, since KV loading becomes dominant (instead of parameter loading).

We can understand this better by breaking down the sources of cost and latency into param loading time, KV loading time, and FLOPs time. The red sector is the region in which we expect to be compute-bound in our MLP blocks.

<div class="l-page">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

This tells quite a story. You can see that initially, parameter loading represents the vast majority of the latency, until the batch size becomes large enough that FLOPs and KV loading become more significant. Notably, at all batch sizes greater than 2k, FLOPs is always smaller than our KV loading time in this regime. **So we slowly transition from being bound by parameter loading to KV loading, but are always memory bound.**

<p markdown=1 class="takeaway">**Takeaway:** for LLaMA 3-70B, we are strongly KV cache memory bandwidth-bound (and HBM-bound) in almost all of these configurations, highlighting just how important reducing KV cache size is for generation throughput. Also note just how dramatic the latency/throughput tradeoff remains here.</p>

{% details The code for this is quite simple. %}

Here's the code for computing these rooflines:

```py
import numpy as np

num_chips = 16  # we fix 16 as the amount of total model parallelism we do
param_size = 70e9  # int8 means 1 byte per param
sequence_length = 8192  # can vary this

hbm_bandwidth = 8.20E+11  # v5e
flops = 1.97E+14  # v5e

param_size = bytes_per_param * param_count

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80
    
def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(max_num_chips: int = 16):
  # for num_chips in topo_sizes:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  num_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(num_chips <= max_num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(num_chips, sequence_length, param_size)  # get the largest batch size that can fit
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_count * batch_sizes / (num_chips * flops)  # roughly true in a 2ND sense

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # always bandwidth-bound for generate

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

Note how we very explicitly break out latency into two sources: KV loading and param loading, and how the latency is either bound by FLOPs or comms, whichever is bigger.

{% enddetails %}

## Worked Problems

Here are a few worked problems. Some of these repeat things that are worked above, but might be pedagogically useful.

**Question 1:** How many FLOPs does each forward pass for LLaMA 3-405B use per-token? Assuming we're FLOPs bound, what is a lower bound on a single forward pass on N chips on TPU v5e? What if we're comms bound? *Ignore the fact that the model does not fit on a single chip.*

**Question 2:** Assume we want to serve LLaMA 3-8B with BS240 using int8 weights and int8 KV caches. How many bytes are used by (a) model parameters (b) KV caches and (c) peak working activations (roughly)? What's the smallest topology we can run this on?

**Question 3:** How would you serve LLaMA 3-405B on TPU v5e? Assume int8 weights and bfloat16 FLOPs. Let's say we have a firm limit of 15ms / token, what's the highest throughput configuration we could achieve? What is the theoretical minimum step time?

<h3 markdown=1 class="next-section">That's all for Part 8! For Part 9, with a deep dive into XLA and TPU profiling, click [here](../profiling).</h3>