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
    - name: "Thinking about throughput"
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

*This section will look at what it takes to serve LLaMA-3 and how efficiently it can be done. As in the previous "applied" section, try to work out the answers on your own with a pen and paper before looking them up!*

## What's the LLaMA Serving Story?

Let's remind ourselves what LLaMA 3-70B looks like (see [Section 6](../applied-training) for reference):

| **hyperparam**              | **value** |
| --------------------------- | :-------: |
| $$n_\text{layers}$$ (L)     |    80     |
| $$d_\text{model}$$ (D)      |   8,192   |
| $$d_{ff}$$ (F)              |  28,672   |
| $$n_\text{heads}$$ (N)      |    64     |
| $$n_\text{kv heads}$$ (K)   |     8     |
| $$d_\text{qkv}$$ (H)        |    128    |
| $$n_\text{embeddings}$$ (V) |  128,256  |

Let's start with a simple question: **what hardware should we serve on?** The answer is basically, whichever is cheapest in FLOPs / dollar.<d-footnote>This isn't always true, sometimes more HBM or ICI bandwidth is critical rather than FLOPs, but this is a good heuristic.</d-footnote> For this reason, we typically want to serve on TPU v5e, our current dedicated inference chip (cost comes from [Google Cloud pricing](https://cloud.google.com/tpu/pricing) as of February 2025):

| **TPU type** | **bfloat16 FLOPs/s** | **Google Cloud USD / hour** | **FLOPs / $** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| H100         |        9.9e14        |            $10.8            |    3.3e17     |
| v5p          |       4.59e14        |            $4.2             |    3.9e17    |
| v5e          |       1.97e14        |            $1.2             |  **5.8e17**  |

Each TPU v5e has 16GB of HBM which will require us to shard our model fairly aggressively. Let's start by thinking about some basic quantities that might matter for us:

**Question:** How large are LLaMA 3-70B's KV caches per token? *You can assume we store them in int8. This determines how large our batch size can be on a given topology.*

{% details Click here once you've thought it through! %}

LLaMA 3-70B has 8 KV heads, so the size per token is `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`.

**Note just how big this is!** If we have a sequence length of 32k tokens (as is common), this uses `162e3 * 32,768 = 5.3GB / sequence`. For BS=240, this is 1.3TB! Since TPU v5e only have 16GB a piece, we would need about `(70e9 + 1.3e12) / 16e9 = 86` TPU v5e chips to even fit this much memory. Also note how large this is compared to the 70GB of model parameters.

{% enddetails %}

**Question:** Let's say we want to serve L3 70B at batch size 32 and 8192 sequence length with everything (params and KVs) in int8. How much total memory will this use? What's the smallest slice we could serve this on?

{% details Answer %}

Since our KVs are `160e3` bytes in int8, our total KV memory is `160e3 * 8192 * 32 = 41.9e9` bytes. Our parameters are `70e9` bytes, since we have 1 byte per parameter. Thus, our total memory usage is `41.9e9 + 70e9 = 112GB`.

The smallest slice we could use would have `112e9 / 16e9 = 7` TPUs, or (rounding to an even size), TPU v5e `4x2`. This will be a tight fit and we might not be able to quite fit this accounting for other overhead, so we might need a `4x4` at minimum (or to drop the batch size).

{% enddetails %}

**Question:** At this batch size and quantization on a TPU v5e `4x2`, roughly what latency would we expect per decode step? What throughput (tokens / sec / chip). What about a `4x4`? *Assume we perform our FLOPs in bfloat16 and everything is fully sharded.*

{% details Answer %}

We can invoke the formula from the previous section that

$$\begin{align*}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align*}$$

Here our critical batch size will be about 120 since our parameters are in int8 but our FLOPs are in bfloat16. We could also manually calculate the RHS maximum, but that's basically a calculation we've already done several times. **So we're well into the memory-bound regime for both our matmul and our FLOPs.**

Strictly looking at memory bandwidth then, our step time is basically `(KV size + param size) / (8 * HBM bandwidth) = 112e9 / (8 * 8.1e11) = 17ms`. **So theoretically our step time is about 17ms.** Our throughput would be `32 / .017 = 1882 tokens / sec`, or `1882 / 8 = 235 tokens / sec / chip`.

There's one caveat here which is to check if we might be ICI bound on our matmuls. We could dedicate 2 axes to it here, so we're ICI bound in theory when $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$, so we're golden!

If we were to run on a `4x4`, we'd still be fine ICI-wise, so our latency would drop to `17 / 2 = 8.5ms`, but our throughput per-chip would remain the same.

{% enddetails %}

### Thinking about throughput

Let's spend a little time thinking purely about throughput. When we optimize for throughput, we want to be compute bound, meaning we come close to utilizing all the TPU MXU capacity. Typically that means we want the batch size to be as large as possible, so we are doing as much work as possible.

**Question:** On TPU v5e, using bfloat16 weights and activations, how large do our batch sizes need to be for us to be compute-bound in our matmuls? What if we do int8 weights but perform our FLOPs in bfloat16? What about int8 weights with int8 FLOPs?

{% details Answer %}

As discussed in Section 7, for any bfloat16 matmul for which $B \ll D, F$ we have

$$\begin{equation*}
T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM bandwidth}} = 240
\end{equation*}$$

When our weights are in int8, we lose a factor of 2 in the denominator, so we have $2BDF / DF = 2B > 240$, or equally $B > 120$, half the critical batch size from before. That's really helpful for us! When we do int8 weights and int8 FLOPs, we have to use the int8 value for TPU FLOPs/s, which goes from 1.97e14	for bfloat16 to 3.94e14, nearly double. That means we're back where we started at about $B > 240$.

The case of int8 weights and bfloat16 FLOPs is quite common, since quantizing parameters losslessly is often easier than doing low-precision arithmetic.

{% enddetails %}

**Question:** What is the smallest TPU v5e topology we could serve LLaMA 3-70B on using bfloat16, int8, and int4 (both KVs and parameters) with 8k context? *You can think of KV caches as negligibly small for this one.*

{% details Answer %}

This is easy! If we're OK with a tiny batch size then the only limit is fitting parameter memory in HBM, i.e. it is just `ceil(num_params * sizeof(dtype) / HBM per TPU`, or `ceil(70e9 * sizeof(dtype) / 16e9)` rounded to the nearest reasonable topology (some multiple of 2):

| dtype | param size | KV size / token (bytes) | min TPU v5es | actual min slice | remaining HBM for KV caches | num KV caches @ 8k |
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: |
| bf16  |   140GB    |          324kB          |     8.75     |  4x4 = 16 chips  |             116             |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8 chips   |             68              |         52         |
| int4  |    45GB    |          81kB           |     2.81     |  2x2 = 4 chips   |             19              |         67         |

That's pretty cool! It tells us we could fit LLaMA 70B on a TPU v5e 2x2 if we wanted to. Except you'll notice the number of KV caches is very small. That's our batch size! That means we'll be getting terrible FLOPs utilization. We'd be very happy to use a larger topology in order to push our batch size up to 240.

{% enddetails %}

**Question:** Assume we use the largest batch size that fits on these topologies, what latency we could expect for each generate step?

{% details Answer %}

This is also easy, since we're picking our batch size to fill up all our HBM! This is just a question of how long it takes to load a full TPU v5e's worth of bytes into the MXU. This is just `v5e HBM / v5e HBM memory bandwidth = 16GB / 8.2e11 = 19ms`, so this is **19ms / step**. Assuming our generations have a median length of 512 tokens, that is about 9s for each decode. Note that we could get marginally better latency with a smaller batch size, for instance if we only looked at model parameters in int4 our minimum latency is about 10ms / step, since HBM is no longer full.

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaway**: we can always lower bound decode latency by asking how long it takes to load all the model's parameters from HBM into the MXU. When our KV caches are small, you can think about each layer as just loading the weights chunk-by-chunk and then discarding them. Unless we're using large batch sizes or lots of inter-device comms, this is often a reasonable bound (within 1.5x). When our batch size is bigger, we need to model the KV cache loading as well, since that dominates the parameters.</p>

Likewise, in the FLOPs-bound regime (e.g. training or big-batch inference), we can use the $$\text{Total FLOPs} / (N \cdot C) = 2 \cdot \text{param count} \cdot B / (N \cdot C)$$ lower bound, which assumes no communication.

**Question:** For each of these, what throughput per chip does this give us (in terms of queries / chip)? *You can assume our median decode length is 512 tokens.*

{% details Answer %}

This is an important question because it's exactly correlated with cost / token.

With our assumption about median decode length, our throughput is just $$B / (\text{per-step latency} \cdot \text{median steps} \cdot N) \approxeq 43 / (0.019 * 512 * N)$$. This gives us roughly $$(4.42 / N)$$ QPS, so plugging in $$N$$ we get:

|  dtype   | QPS / chip |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.66    |
|   int4   |    1.72    |

Note that this is rather optimistic since it totally ignores the working memory of the forward pass (memory allocated to activations and attention). This is not ridiculous with Flash Attention, but it is also not realistic. The real numbers are likely maybe 1/2 of this. For absolutely maximum throughput we would probably want to more than double the number of chips and increase the batch size significantly as well.

{% enddetails %}

**Question:** How would our peak throughput change if we doubled our topology for each of the above examples?

{% details Answer %}

If we used a 4x8 slice in bfloat16, we would have 186GB remaining for KV caches, which would let us up our batch size to 161. Then since our step time would remaining the same, we would have a throughput of `16.54 / num_chips`, or

|       dtype       | QPS / chip |
| :---------------: | :--------: |
| bfloat16 (on 4x8) |    0.51    |
|   int8 (on 4x4)   |    1.03    |
|   int4 (on 2x4)   |    2.06    |

A further increase would give an even bigger win! The big takeaway is that **the smallest topology is not the most performance topology** in all cases, if we're limited by KV cache size.

{% enddetails %}

**Question:** Now let's dig into the question of sharding. Let's say we wanted to serve in bfloat16 on a TPU v5e 4x8. What sharding would we use for our model on a TPU v5e 4x8 during generation? Can we avoid being communication bound?

{% details Answer %}

As discussed in the previous section, we only really have one option for sharding during generation: model parallelism. How much can we do before we become communication bound? As we've discussed in the previous section, our models become communication bound roughly when

$$Y > \frac{F \cdot M_Y}{2200}$$

For LLaMA 3-70B we have `F = 28,672`, so if we do 2 axes of model sharding this gives us roughly $$Y = 28672 \cdot 2 / 2200 = 26$$, so in general we could scale up to about 16 chips without being communication bound, which lets us use a `4x4` but not a `4x8`. Generally, since we do not perfectly overlap computation, even this estimate is overly optimistic.

**Takeaway: we cannot actually serve on a 4x8 with pure model parallelism.** The best we can do here is a 4x2 or _maybe_ a 4x4.

However, as we've discussed, when our batch size is small we can often do more model parallelism without significantly hurting throughput, since our model is memory-bandwidth-bound and not FLOPs bound. We said before that this value is roughly $Y=F / (8\cdot B)$, so if we did batch size 64, we could in theory go up to `Y = 28,672 / (8 * 64) = 56` way model parallelism before we become ICI-bound. To sanity check this, we can look at $T_\text{ici comms}$, $T_\text{hbm comms}$, and $T_\text{math}$ for a single matmul. We clearly have:

$$\begin{align*}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}$$

For a `4x8`, this would give us $T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`, $T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.1e11) = 18us`, and $T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`, so in theory we're still HBM bandwidth bound, which is great! *Note that scaling up from a `4x4` to a `4x8` probably isn't helpful from a throughput standpoint, but it'll reduce our latency!

If we look at the int8 and int4 configs, we _can_ do those with pure model parallelism. So we've hit a point at which quantization actually gives us a meaningful advantage beyond faster FLOPs: it lets us use a larger batch size before we become comms-bound. **So the end of this story is that we can't achieve peak throughput on a 4x8, but for the int8 and int4 configs we could do pure model parallelism*.

{% enddetails %}

<p markdown=1 class="takeaway">**Tip**: the maximum amount of useful model parallelism depends on $$d_{ff}$$ and the number of axes over which you're sharding your model. The maximum value usually ranges between 8 and 32 depending on the model size. You can scale beyond this limit to improve latency at some throughput cost.</p>

### What about prefill?

We've mostly ignored prefill here because it's much simpler. Let's put a couple of concepts together and think about the end-to-end picture.

**Question:** Assume we achieve a 40% FLOPs utilization during prefill. How long will a prefill of length 8192 take on 16 TPU v5e chips?

{% details Answer %}

At 8k tokens, we are solidly compute bound, so we just need to reason about FLOPs. We know our model has `70e9` parameters so each forward pass uses `2 * 70e9 * B` FLOPs. Assuming 40% MFU (FLOPs utilization), this gives us a runtime of about `2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s`. Compared to the numbers we've been looking at before, that's actually quite a lot!

{% enddetails %}

**Question:** Assume we have a median prefill length of 8192 tokens and a median decode length of 4096 tokens. Say we have a generate batch size of 32. On average how many sequences finish decoding per step? On average how many tokens are evicted from our KV cache each step?

{% details Answer %}

This is kind of straightforward. Since we have a median decode length of 4096 tokens, a sequence will finish roughly every 1 / 4096 tokens. Given a batch size of 32, this means we have `32 / 4096` sequences evicted per step. Since our KV cache length is roughly `8192 + 4096`, this is `32 * (8192 + 4096) / 4096 = 96` tokens evicted per step. The general formula is $B * (P + G) / G$ where $P$ and $G$ are the prefill and generate lengths.

{% enddetails %}

**Question:** Assume we do disaggregated serving with a median prefill length of 8192 and a median decode length of 512. Assume the prefill and generate latencies calculated above in bfloat16. What ratio of prefill:generate servers will you need to keep both fully saturated.

{% details Answer %}

This is kind of a fun question. Let $P$ be the number of prefill servers and $G$ be the number of generate servers. So generally speaking, this is a pipeline problem where we feed sequences in at a rate of `P / prefill_latency` and consume them at a rate of `B * G / (generate_latency * median_decode_length)`. We had calculated `910ms` per prefill step and `19ms` per decode step at batch size 43 (let's call that 32). Therefore we need `P / 0.91 = 32 * G / (0.019 * 512)` or `P = 3G`, i.e. we need about 3 times more prefill servers than generation servers!

{% enddetails %}

## Visualizing the Latency Throughput Tradeoff

Sticking with LLaMA 70B for a second, let's actually look at the latency and throughput for different batch sizes during generation. As we showed in the previous section for PaLM models, this gives us a Pareto frontier for throughput/latency. Let's assume 16-way tensor parallelism since that's a reasonable bound on what we can use while staying compute-bound in the MLP blocks. We'll use a TPU v5e 4x4 topology here. **The slider controls the sequence length so you can see the effect of larger KV caches.**

<div class="l-page">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

* **See how dramatic the tradeoff is between cost and latency.** At the cost of doubling per-token latency, we can achieve a roughly 100x reduction in per-token cost. Also, our latency can range anywhere from 5.5ms with low batch size to 20 ms with very large batches.
* Note how at 2k context the throughput effectively plateaus at around 1 token / ms / chip when it hits the BS 120 roofline (120 here because we do int8 weights but bf16 FLOPs). As the sequence length increases, however, we can no longer fit this batch size in memory, so we never hit the point of full saturation.
* Note how much higher the latency is at large batch sizes for the same throughput, since KV loading becomes dominant (instead of parameter loading).

We can understand this better by breaking down the sources of cost and latency into param loading time, KV loading time, and FLOPs time. The red sector is the region in which we expect to be compute-bound in our MLP blocks.

<div class="l-page">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

This tells quite a story. You can see that initially, parameter loading represents the vast majority of the latency, until the batch size becomes large enough that FLOPs and KV loading become more significant. Notably, at all sequence lengths greater than 2048, we spend more time on KV cache loading than we do on FLOPs! **So while we can improve our hardware utilization by increasing batch size, at long context lengths KV loading always dominates the total step time.**

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
