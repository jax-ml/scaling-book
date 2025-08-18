---
layout: distill
title: "All About Transformer Inference"
# permalink: /main/
description: "Performing inference on a Transformer can be very different from training. Partly this is because inference adds a new factor to consider: latency. In this section, we will go all the way from sampling a single new token from a model to efficiently scaling a large Transformer across many slices of accelerators as part of an inference engine."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 7

previous_section_url: "../applied-training"
previous_section_name: "Part 6: Training LLaMA"

next_section_url: ../applied-inference
next_section_name: "Part 8: Serving LLaMA"

bibliography: main.bib

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
  - name: "The Basics of Transformer Inference"
  - subsections:
    - name: "What do we actually want to optimize?"
    - name: "Linear operations: what bottlenecks us?"
    - name: "What about attention?"
    - name: "Theoretical estimates for LLM latency and throughput"
    - name: "What about memory?"
    - name: "Modeling throughput and latency for LLaMA 2-13B"
  - name: "Tricks for Improving Generation Throughput and Latency"
  - name: "Distributing Inference Over Multiple Accelerators"
  - subsections:
    - name: "Prefill"
    - name: "Generation"
    - name: "Sharding the KV cache"
  - name: "Designing an Effective Inference Engine"
  - subsections:
    - name: "Continuous Batching"
    - name: "Prefix Caching"
    - name: "Let's look at an implementation: JetStream"
  - name: "Worked Problems"
  - name: "Appendix"

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

## The Basics of Transformer Inference

So you've trained a Transformer, and you want to use it to generate some new sequences. _At the end of the day, benchmark scores going up and loss curves going down are only proxies for whether something interesting is going to happen once the rubber hits the road!_<d-footnote>Historically, you can do a surprising amount of research on Transformers without ever touching inference — LLM loss, multiple choice benchmarks can be run efficiently without a proper KV cache or generation loop implementation. This meant, especially in research codebases, there's often a lot of low hanging fruits in the inference codepath.</d-footnote>

Sampling is conceptually simple. We put a sequence in and our favorite Transformer will spit out $$\log p(\text{next token}_i \vert \text{previous tokens})$$, i.e. log-probabilities for all possible next tokens. We can sample from this distribution and obtain a new token. Append this token and repeat this process and we obtain a sequence of tokens which is a continuation of the prompt.

{% include figure.liquid path="assets/img/naive-inference.png" class="img-fluid" caption="<b>Figure:</b> naive sampling from a Transformer. The blue logits give us a distribution over the next token that we can sample from. Note that each step re-processes the entire prefix, leading to a $\Theta(n^2)$ runtime for the algorithm." %}

We have just described the naive implementation of Transformer sampling, and while it works, **we never do it in practice** because we are re-processing the entire sequence every time we generate a token. This algorithm is $$O(n^2)$$ on the FFW and $$O(n^3)$$ on the attention mechanism to generate $$n$$ tokens!

**How do we avoid this?** Instead of doing the full forward pass every time, it turns out we can save some intermediate activations from each forward pass that let us avoid re-processing previous tokens. Specifically, since a given token only attends to previous tokens during dot-product attention, we can simply write each token's key and value projections into a new data structure called a **KV cache**. Once we've saved these key/value projections for past tokens, future tokens can simply compute their $$q_i \cdot k_j$$ products without performing any new FLOPs on the earlier tokens. Amazing!

With this in mind, inference has two key parts:

* <b style="color: red;">Prefill</b>: Given a long prompt, we process all the tokens in the prompt at the same time and save the resulting activations (specifically, the key-value projections) in a **"KV cache"**. We also save the logits for the last token.
* <b style="color: blue;">Generation</b>: Given a KV cache and the previous logits, we incrementally sample one token from the logits, feed that token back into the Transformer, and produce a new set of logits for the next step. We also append the KV activations for that new token to the KV cache. We repeat this until we hit a special `<EOS>` token or reach some maximum length limit.

Here's a diagram of sampling with a KV cache:

{% include figure.liquid path="assets/img/cached-inference.png" class="img-fluid" caption="<b>Figure:</b> diagram of efficient Transformer sampling with a KV cache. <b style=\"color: red;\">Prefill</b> processes our prompt and saves all the per-token key-value activations in a cache. <b style=\"color: blue;\">Generation</b> takes this cache (and the last-token logits), samples a new token, and passes that new token through the model, attending to the KV cache and saving the new token's key-value projections back to the cache. This is an $O(n)$ algorithm in the MLP block." %}

By sampling with a KV cache, we've reduced our time complexity to generate $n$ tokens to $$O(n)$$ on the FFW and $$O(n^2)$$ on the attention, since we never reprocess a previous token. However, many forward passes are still needed to generate a sequence — that's what's happening when you query Gemini or ChatGPT and the result streams back to you. Every token is (usually) a separate (but partially cached) Transformer call to a massive model.

We will soon see that <b style="color: red;">prefill</b> and <b style="color: blue;">generation</b> are very different beasts —— Transformer inference is two tasks in disguise! Compared to training, the KV cache is also a novel and significant source of complexity.

### What do we actually want to optimize?

Before we proceed further, it's worth highlighting one aspect of inference that's totally new: latency. While during training we only care about throughput (total tokens processed per second), during inference we have to worry about how fast we're producing tokens (both the **Time To First Token (TTFT)** and the **per-token latency**). For example:

* **Offline batch inference** for evals and data generation only cares about bulk cost of inference and is blind to the latency of individual samples.
* **Chat interfaces/streaming tasks** need to run cheaply at scale while having low TTFT and generating tokens fast enough to exceed human reading speed.
* **Edge inference** (e.g. `llama.cpp` on your laptop) only needs to service one user at a time at the lowest possible latency, potentially with heavy hardware constraints.

Maximizing hardware utilization is still critical and helps with cost and TTFT, but unlike training, it does not *necessarily* translate to better experience for individual users in all contexts. Many optimizations at the accelerator, systems and model architectural level make tradeoffs between latency, throughput, context length and even model quality.

### A more granular view of the Transformer

So far we've mostly treated a Transformer as a stack of feedforward blocks. While this is often reasonable from a FLOPs and memory standpoint, it's not sufficient to properly model inference.<d-footnote>One thing you'll notice throughout this section is that inference is much less forgiving than training. We typically have far fewer FLOPs, less opportunity for batching, and a much greater sensitivity to latency. KV caches dramatically complicate inference as well.</d-footnote> As we saw in [Part 4](../transformers), the major components of a Transformer forward pass are:

1. **A bunch of linear operations**, including the MLP ($W_{in}$, $W_{out}$) and the attention QKV projections and output projections ($W_Q$, $W_K$, $W_V$, and $W_O$). These all involve reading parameters and a batch of activations from HBM, doing some FLOPs, and writing the result back to HBM.
2. **Dot-product attention**. We need to read a batch of key-value projections and a batch of query activations from HBM, do a few inner products and some softmax operations, and write the attention result back to HBM.
3. **Everything else**, including applying layer norms, activation functions, tokens sampling, updating KV caches, and positional embeddings. These do take some FLOPs, but are dominated by, or fused into, the above.

For the next couple of sections, we're going to look at each of these in the context of prefill and generation and ask what is likely to bottleneck our performance. Within a single accelerator, are we compute-bound or memory-bound? We want to emphasize how different the answers will be for prefill versus generation.

### Linear operations: what bottlenecks us?

All our linear operations are conceptually the same, whether they live in the MLP block or attention. Their arithmetic intensity depends on the batch size. We did this math in [Section 1](../roofline) but it's worth repeating. Let's look at a single matrix multiply of a $\text{bf16[B, D]}$ batch by a $\text{bf16[D, F]}$ matrix. This could be the big MLP block ($W_\text{in}$ or $W_\text{out}$) or one of the smaller attention projections ($W_Q$, $W_K$, $W_V$, $W_O$). To do this matmul, we need to load both of these arrays from HBM into the MXU, do the multiplicaton, then write the result back to HBM. As before, we have:

$$T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}} = \frac{2BDF}{\text{Accelerator FLOPs/s}}$$

$$T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}} = \frac{2BD + 2FD + 2BF}{\text{Bandwidth Bytes/s}}$$

A TPU or GPU can overlap these by loading as it does the compute, so to be compute-bound, we need $$T_\text{math} \geq T_\text{comms}$$, or:

$$\frac{2BDF}{2BD + 2DF + 2BF} \geq \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} \underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} = 240$$

where the RHS is the arithmetic intensity of our hardware. Now let's assume $D$ and $F$ are very large compared to $B$ (usually our batches are at most 500 and $D$ and $F > 10k$), we can simplify the denominator by using the fact that $\small{2BD + 2DF + 2BF \approxeq 2DF}$ which gives us

$$\begin{align*}
\frac{2BDF}{2BD + 2DF + 2BF} \approxeq \frac{2BDF}{2DF} \geq \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} \\
\underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} \implies B \geq 240 = B_{\text{crit}}
\end{align*}$$

If we quantize our weights or use lower precision FLOPs for the matrix multiplication, this critical batch size can change. For instance, if we quantize our weights to int8 or fp8, $B_\text{crit}$ decreases by 2x. If we do our FLOPs in int8 or fp8, $B_\text{crit}$ increases by 2x. Thus if we let $\beta = \text{bits per param} / \text{bits per activation}$ and $\alpha_\text{hbm} = C / W_\text{hbm}$, our critical batch size is actually $B_\text{crit} = \beta \alpha_\text{hbm}$.

<p markdown=1 class="takeaway">**Takeaway:** Transformer matmuls are compute-bound *iff* the per-replica **token** batch size is greater than $B_\text{crit} = C / W_\text{hbm} \cdot (\text{bits per param} / \text{bits per activation}) = \beta \cdot \alpha_\text{hbm}$. For bf16 activations on TPU v5e, this is 240 tokens. For an H100, it is about 280 tokens.</p>

During training, we'll have a high intensity during all our matrix multiplications because we reuse the same weights over a very large batch. **That high arithmetic intensity carries over to prefill, since user prompts are typically hundreds if not thousands of tokens long.** As we saw before, the hardware arithmetic intensity of a TPUv5e is 240, so if a sequence longer than 240 tokens is fed into a dense model running on this hardware at bf16, we would expect to be compute-bound and all is well. Prompts shorter than this can technically be batched together to achieve higher utilization, but this is typically not necessary.

<p markdown=1 class="takeaway">**Takeaway:** During prefill, all matrix multiplications are basically always compute-bound. Therefore, simply maximizing hardware utilization or MFU (Model FLOPs Utilization) is enough to maximize throughput-per-chip (cost) and latency (in the form of TTFT). Unless prompts are extremely short, batching at a per-prompt level only adds latency for a small improvements in prefill throughput.</p>

However, during generation, for each request, we can only do our forward passes one token at a time since there's a sequential dependency between steps! Thus we can only (easily) achieve good utilization by batching multiple requests together, parallelizing over the batch dimension. We'll talk about this more later, but actually batching many concurrent requests together without affecting latency is hard. For that reason, **it is much harder to saturate the hardware FLOPs with generation.**

<p markdown=1 class="takeaway">**Takeaway:** During generation, the total token batch size must be greater than $B_{\text{crit}}$ to be compute-bound on the linear/feed-forward operations (240 for bf16 params on TPU v5e). Because generation happens serially, token-by-token, this requires us to batch multiple requests together, which is hard!</p>

*It's worth noting just how large this is!* Generate batch size of 240 means 240 concurrent requests generating at once, and 240 separate KV caches for dense models. That means this is difficult to achieve in practice, except in some bulk inference settings. In contrast, pushing more than 240 tokens through during a prefill is pretty routine, though some care is necessary as sparsity increases.

**Note that this exact number will differ on the kind of quantization and hardware.** Accelerators often can supply more arithmetic in lower precision. For example, if we have int8 parameters but do our computation in bf16, the critical batch size drops to 120. With int8 activations and int8 params, it jumps back up to 240 since the TPUv5e can supply 400 TOPs/s of int8 x int8.

### What about attention?

Things get more complicated when we look at the dot-product attention operation, especially since we have to account for KV caches. Let's look at just one attention head with pure multi-headed attention. In a single Flash Attention fusion, we<d-footnote>We're simplifying a fair bit here by ignoring the non-matmul FLOPs in applying the softmax, masks etc. They should be overlapped with computation or HBM reads, but it can be non-trivial to do on certain TPU generations. Whese details don't change the main message, which is that KV caches are usually memory bound.</d-footnote>:

1. Read the $Q$ activations of shape $\text{bf16[B, T, D]}$ from HBM.
2. Read the $KV$ cache, which is a pair of $\text{bf16[B, S, D]}$ tensors from HBM.
3. Perform $2BSTD$ FLOPs in the $$QK$$ matmul. With Flash Attention, we don't need to write the $\text{bf16[B, S, T]}$ attention matrix back into HBM.
4. Perform $2BSTD$ in the attention $$AV$$ matmul.
5. Write the resulting $\text{bf16[B, T, D]}$ tensor back into HBM.

Putting it all together, we get:

$$\text{Multiheaded Attention Arithmetic Intensity} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}$$

For prefill, $S=T$ since we're doing self-attention, so this simplifies to $T^2 / 2T = T / 2$. This is great because it means **the arithmetic intensity of attention during prefill is $\Theta(T)$**. That means it's quite easy to be compute-bound for attention. As long as our sequence length is fairly large, we'll be fine!

But since generation has a trivial sequence dim, and the $B$ and $D$ dims cancel, we can make the approximation:

$$S \gg T = 1 \implies \frac{ST}{S+T} \approx 1$$

This is bad, since it means we cannot do anything to improve the arithmetic intensity of attention during generation. We're doing a tiny amount of FLOPs while loading a massive KV cache. **So we're basically always memory bandwidth-bound during attention!**

<p markdown=1 class="takeaway">**Takeaway:** during prefill, attention is usually compute bound for any reasonable sequence length (roughly $\gt 480$ tokens) while during generation our arithmetic intensity is low and constant, so we are always memory bandwidth-bound.</p>

*Why is this, conceptually?* Mainly, we're compute-bound in linear portions of the model because the parameters (the memory bandwidth-heavy components) are reused for many batch items. However, every batch item has its own KV cache, so a bigger batch size means more KV caches. We will almost *always* be memory bound here unless the architecture is adjusted aggressively.

This also means you will get diminishing returns on throughput from increasing batch size once params memory becomes comparable to KV cache memory. The degree to which the diminishing returns hurt you depends on the ratio of parameter to KV cache bytes for a single sequence, i.e. roughly the ratio $2DF / SHK$. Since $HK\approx D$, this roughly depends on the ratio of $F$ to $S$, the sequence length. This also depends on architectural modifications that make the KV cache smaller (we'll say more in a moment).

### Theoretical estimates for LLM latency and throughput

From this math, we can get pretty good bounds on the step time we should aim for when optimizing. **(Note: if there is one thing we want to the reader to take away from this entire chapter, it's the following).** For small batch sizes during generation (which is common), we can lower-bound our per-step latency by assuming we're memory bandwidth bound in both the attention and MLP blocks:

$$\begin{equation*}
\text{Theoretical Min Step Time} = \frac{\text{Batch Size} \times \text{KV Cache Size} + \text{Parameter Size}}{\text{Total Memory Bandwidth}}
\end{equation*}$$

Similarly, for throughput:

$$\begin{equation*}
\text{Theoretical Max Tokens/s} = \frac{\text{Batch Size} \times \text{Total Memory Bandwidth}}{\text{Batch Size} \times \text{KV Cache Size} + \text{Parameter Size}}
\end{equation*}$$

Eventually, as our batch size grows, FLOPs begin to dominate parameter loading, so in practice we have the more general equation:

$$\begin{align}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align}$$

where the attention component (left) is never compute-bound, and thus doesn't need a FLOPs roofline. These are fairly useful for back-of-the-envelope calculations, e.g.

<b markdown=1 style="color: #57cf57;">Pop Quiz:</b> Assume we want to take a generate step with a batch size of 4 tokens from a 30B parameter dense model on TPU v5e 4x4 slice in int8 with bf16 FLOPs, 8192 context and 100 kB / token KV caches. What is a reasonable lower bound on the latency of this operation? What if we wanted to sample a batch of 256 tokens?

{% details Click here for the answer. %}

**Answer:** in int8, our parameters will use 30e9 bytes and with the given specs our KV caches will use `100e3 * 8192 = 819MB` each. We have 16 chips, each with `8.1e11` bytes/s of bandwidth and `1.97e14` bf16 FLOPs/s. From the above equations, since we have a small batch size, we expect our step time to be at least `(4 * 819e6 + 30e9) / (16 * 8.1e11) = 2.5 ms`. At 256 tokens, we'll be well into the compute-bound regime for our MLP blocks, so we have a step time of roughly `(256 * 819e6) / (16 * 8.1e11) + (2 * 256 * 30e9) / (16 * 1.97e14) = 21ms`.

{% enddetails %}

As you can see, there's a clear tradeoff between throughput and latency here. Small batches are fast but don't utilize the hardware well. Big batches are slow but efficient. Here's the latency-throughput Pareto frontier calculated for some older PaLM models (from the [ESTI paper](https://arxiv.org/pdf/2211.05102)<d-cite key="esti"></d-cite>):

{% include figure.liquid path="assets/img/latency-cost.png" class="img-fluid" caption="<b>Figure:</b> Pareto frontier of cost (read: throughput) versus latency for several PaLM models. Note how chip count (C) and batch size (B) moves you along the Pareto frontier, with the exception of the green dot (C:32 B:16 for PaLM 540B) where the available memory prevented the setup from supporting a good batch size and caused throughput to suffer. Note how throughput generally tends to flatten around after the batch size 240. int8 weights offers a better latency-throughput pareto optimal, but not a better max throughput." %}

Not only do we trade off latency and throughput with batch size as knob, we may also prefer a larger topology to a smaller one so we can fit larger batches if we find ourselves limited by HBM. The [next section](../applied-inference) explores this in more detail.

<p markdown=1 class="takeaway">**Takeaway:** if you care about generation throughput, use the largest per-chip batch size possible. Any per-chip batch size above the TPU arithmetic intensity ($B_\text{crit}$, usually 120 or 240) will maximize throughput. You may need to increase your topology to achieve this. Smaller batch sizes will allow you to improve latency at the cost of throughput.</p>

{% details There are some caveats to this from a hardware standpoint. Click here for some nits. %}

This is all quite theoretical. In practice we often don't quite see a sharp roofline for a few reasons:

* Our assumption that HBM reads will be perfectly overlapped with FLOPs is not realistic, since our compiler (XLA) is fallible.
* For sharded models, XLA also often fails to efficiently overlap the ICI communication of our model-sharded matrix multiples with the FLOPs themselves, so we often start taking a latency hit on linears over $$\text{BS}=32$$.
* Batch sizes larger than the theoretical roofline will still see some improvement in throughput because of imperfect overlapping, but this is a good heuristic.

{% enddetails %}

### What about memory?

We've spent some time looking at bandwidth and FLOPs, but not at memory. The memory picture looks a lot different at inference time, thanks to our new data structure, the KV cache. For this section, let's pick a real model (LLaMA 2-13B) to demonstrate how different things look:


| hyperparam               | value  |
| ------------------------ | ------ |
| n\_layers (L)            | 40     |
| d\_model (D)             | 5,120  |
| ffw\_multiplier (F // D) | 2.7    |
| n\_heads (N)             | 40     |
| n\_kv\_heads (K)         | 40     |
| d\_qkv (H)               | 128    |
| n\_embeddings (V)        | 32,000 |

What's using memory during inference? Well, obviously, our parameters. Counting those, we have:

| param            | formula                                                                                                                   | size (in bytes)                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| FFW params       | d_model<sup>2</sup> x ffw\_multiplier x 3 (for gelu \+ out-projection) x n\_layers                                        | 5,120 x 5,120 x 2.7 x 3 x 40 \= **8.5e9**                         |
| Vocab params     | 2 (input and output embeddings) x n\_embeddings x d\_model                                                                | 2 x 32,000 x 5,120 \= **0.3e9**                                   |
| Attention params | \[2 (*q and output*) x d\_model x n\_heads x d\_qkv \+ 2 (*for k and v*) x d\_model x n\_kv\_heads x d\_qkv\] x n\_layers | (2 x 5,120 x 40 x 128 \+ 2 x 5,120 x 40 x 128\) x 40 \= **4.2e9** |

Adding these parameters up, we get 8.5e9 + 4.2e9 + 0.3e9 = **13e9 total parameters**, just as expected. As we saw in the previous sections, during training we might store our parameters in bfloat16 with an optimizer state in float32. That may use around 100GB of memory. That pales in comparison to our gradient checkpoints, which can use several TBs.

**How is inference different?** During inference, we store one copy of our parameters, let's say in bfloat16. That uses 26GB — and in practice we can often do much better than this with quantization. There's no optimizer state or gradients to keep track of. Because we don't checkpoint (keep activations around for the backwards pass), our activation footprint is negligible for both prefill<d-footnote>Particularly thanks to Flash Attention, which avoids materializing our attention matrix</d-footnote> and generate. If we prefill 8k tokens, a single activation only uses around `8,192 x 5,120 x 2 bytes = 80MB` of memory. Longer prefills can be broken down into many smaller forward passes, so it's not a problem for longer contexts either. Generation use even fewer tokens than that, so activations are negligible.

**The main difference is the KV cache**. These are the keys and value projections for all past tokens, bounded in size only by the maximum allowed sequence length. The total size for $$T$$ tokens is

$$\text{KV cache size} = 2 \cdot \text{bytes per float} \cdot H \cdot K \cdot L \cdot T$$

where $$H$$ is the dimension of each head, $$K$$ is the number of KV heads, $$L$$ is the number of layers, and the 2 comes from storing both the keys and values.

**This can get big very quickly**, even with modest batch size and context lengths. For LLaMA-13B, a KV cache for a single 8192 sequence at bf16 is

$$8192\ (T) \times 40\ (K) \times 128\ (H) \times 40\ (L) \times 2\ (\text{bytes}) \times 2 = 6.7 \text{GB}$$

**Just 4 of these exceed the memory usage of our parameters!** To be clear, LLaMA 2 was not optimized for KV cache size at longer contexts (it isn't always this bad, since usually $K$ is much smaller, as in LLaMA-3), but this is still illustrative. We cannot neglect these in memory or latency estimates.

### Modeling throughput and latency for LLaMA 2-13B

Let's see what happens if we try to perform generation perfectly efficiently at different batch sizes on 8xTPU v5es, up to the critical batch size (240) derived earlier for maximum theoretical throughput.

| Batch Size                        |      1 |      8 |     16 |     32 |     64 |    240 |
| :-------------------------------- | -----: | -----: | -----: | -----: | -----: | -----: |
| KV Cache Memory (GiB)             |    6.7 |   53.6 |  107.2 |  214.4 |  428.8 |   1608 |
| Total Memory (GiB)                |   32.7 |   79.6 |  133.2 |  240.4 |  454.8 |   1634 |
| Theoretical Step Time (ms)        |   4.98 |  12.13 |  20.30 |  36.65 |  69.33 | 249.09 |
| Theoretical Throughput (tokens/s) | 200.61 | 659.30 | 787.99 | 873.21 | 923.13 | 963.53 |

8x TPU v5es gives us 128GiB of HBM, 6.5TiB/s of HBM bandwidth (0.82TiB/s each) and 1600TF/s of compute.

For this model, increasing the batch size does give us better throughput, but we suffer rapidly diminishing returns. We OOM beyond batch size 16, and need an order of magnitude more memory to go near 240. A bigger topology can improve the latency, but we've hit a wall on the per chip throughput.

Let's say we keep the total number of params the same, but magically make the KV cache 5x smaller (say, with 1:5 [GMQA](#tricks-for-improving-generation-throughput-and-latency), which means we have 8 KV heads shared over the 40 Q heads — see next section for more details).

| Batch Size                        |      1 |        8 |       16 |       32 |       64 |      240 |
| :-------------------------------- | -----: | -------: | -------: | -------: | -------: | -------: |
| KV Cache Memory (GiB)             |   1.34 |    10.72 |    21.44 |    42.88 |    85.76 |    321.6 |
| Total Memory (GiB)                |  27.34 |    36.72 |    47.44 |    68.88 |   111.76 |    347.6 |
| Theoretical Step Time (ms)        |   4.17 |     5.60 |     7.23 |    10.50 |    17.04 |    52.99 |
| Theoretical Throughput (tokens/s) | 239.94 | 1,429.19 | 2,212.48 | 3,047.62 | 3,756.62 | 4,529.34 |

With a smaller KV cache, we still have diminishing returns, but the theoretical throughput per chip continues to scale up to batch size 240. We can fit a much bigger batch of 64, and latency is also consistently better at all batch sizes. The latency, maximum throughput, and maximum batch size all improve dramatically! In fact, later LLaMA generations used this exact optimization — LLaMA-3 8B has 32 query heads and 8 KV heads ([source](https://huggingface.co/MaziyarPanahi/Llama-3-13B-Instruct-v0.1/blob/dfdeb40bdb2c149dfa399ea2be0d56eb120f0831/config.json)).

<p markdown=1 class="takeaway">**Takeaway:** In addition to params, the size of KV cache has a lot of bearing over the ultimate inference performance of the model. We want to keep it under control with a combination of architectural decisions and runtime optimizations.</p>

## Tricks for Improving Generation Throughput and Latency

Since the original [Attention is All You Need paper](https://arxiv.org/abs/1706.03762), many techniques have been developed to make the model more efficient, often targeting the KV cache specifically. Generally speaking, a smaller KV cache makes it easier to increase batch size and context length of the generation step without hurting latency, and makes life easier for the systems surrounding the Transformer (like request caching). Ignoring effects on quality, we may see:

**Grouped multi-query attention (aka GMQA, GQA):** We can reduce the number of KV heads, and share them with many Q heads in the attention mechanism. In the extreme case, it is possible to share a single KV head across all Q heads.  This reduces the KV cache by a factor of the Q:KV ratio over pure MHA, and it has been observed that the performance of models is relatively insensitive to this change.

{% include figure.liquid path="assets/img/gmqa.png" class="img-fluid" %}

This also effectively increases the arithmetic intensity of the attention computation (see Question 4 in [Section 4](../transformers)).

**Mixing in some local attention layers:** Local attention caps the context to a small to moderately sized max length. At training time and prefill time, this involves masking the attention matrix to a diagonal strip instead of a triangle. This effectively caps the size of the max length of the KV cache for the local layers. By mixing in some local layers into the model with some global layers, the KV cache is greatly reduced in size at contexts longer than the local window.

**Sharing KVs across layers:** The model can learn to share the same KV caches across layers in some pattern. Whilst this does reduce the KV cache size, and provide benefits in increasing batch size, caching, offline storage etc. shared KV caches may need to be read from HBM multiple times, *so it does not necessarily improve the step time.*

{% include figure.liquid path="assets/img/kv-sharing.png" class="img-fluid" caption="
 <b>Left:</b> Multiple layers of pure global attention. <b>Right:</b> An example of some global/local interleaving pattern with sharing with adjacent layers. Source: <a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai blog</a>."%}

**Quantization:** Inference is usually less sensitive to the precision of parameters and KVs. By quantizing the parameters and KV cache (e.g. to int8, int4, `fp8` etc.), we can save on memory bandwidth on both, decrease the batch size required to reach the compute roofline and save memory to run at bigger batch sizes. Quantization has the added advantage that even if the model was not trained with quantization it can often be applied post training.

**Using ragged HBM reads and Paged Attention:** We allocated 8k of context for each KV cache in the calculations above but it is often not necessary to read the entire KV cache from memory — requests have a wide range of length distributions and don't use the max context of the model, so we can often implement kernels (e.g. Flash Attention variants) that only read the non-padding part of the KV cache.

Paged Attention<d-cite key="paged"></d-cite> is a refinement upon this that stores KV caches in OS-style page tables and mostly avoids padding the KV caches altogether. This adds a lot of complexity but means every batch only uses as much memory as it needs. This is a runtime optimization, so again it is indifferent to architecture.

{% include figure.liquid path="assets/img/paged-attention.png" class="img-fluid img-small" caption="<b>Figure:</b> during generation, a single token (forth) attends to multiple KV cache blocks/pages. By paging the KV cache, we avoid loading or storing more memory than we need to. Taken from the <a href=\"https://arxiv.org/pdf/2309.06180\">PagedAttention paper</a>." %}

<p markdown=1 class="takeaway">**Big Picture:** All told, these KV cache optimizations can reduce KV cache sizes by over an order of magnitude compared to a standard MHA Transformer. This can lead to an order-of-magnitude improvement in the overall cost of the Transformer.</p>

## Distributing Inference Over Multiple Accelerators

So far we've handwaved how we're scaling beyond a single chip. Following [Section 5](../training), let's explore the different strategies available to us and their tradeoffs. As always, we will look at prefill and generation separately.

### Prefill

From a roofline standpoint, **prefill is almost identical to training** and almost all the same techniques and tradeoffs apply — model (Megatron) parallelism, sequence sharding (for sufficiently long context), pipelining, even FSDP are all viable! You just have to keep the KVs kicking around so you can do generation later. As in training, increasing the number of chips gives us access to more FLOPs/s (for potentially lower TTFT), but adds communication overhead (potentially reducing throughput per chip).

**The general rule for sharding prefill:** here's a general set of rules for prefill. We'll assume we're doing prefill on a single sequence only (no batch dimension):

1. *Model sharding:* We typically do some amount of model parallelism first, up to the point we become ICI-bound. As we saw in [Section 5](../training), this is around $F / 2550$ for 1 axis (usually around 4-8 way sharding).
2. *Sequence parallelism:* Beyond this, we do sequence parallelism (like data parallelism but sharding across the sequence dimension). While sequence parallelism introduces some extra communication in attention, it is typically fairly small at longer contexts. As with training, we can overlap the communication and computation (using collective matmuls for Megatron and ring attention respectively).

<p markdown=1 class="takeaway">**Takeaway:** during prefill, almost any sharding that can work during training can work fine. Do model parallelism up to the ICI bound, then do sequence parallelism.</p>

### Generation

Generation is a more complicated beast than prefill. For one thing, it is harder to get a large batch size because we need to batch many requests together. Latency targets are lower. Together, these mean we are typically more memory-bound and more sensitive to communication overhead, which restrict our sharding strategies:

1. **FSDP is impossible:** since we are memory-bound in loading our parameters and KV caches from HBM to the MXU, we do not want to move them via ICI which is orders of magnitudes slower than HBM. *We want to move activations rather than weights.* This means methods similar to FSDP are usually completely unviable for generation.<d-footnote>Accidentally leaving it on after training is an easy and common way to have order of magnitude regressions</d-footnote>

2. **There is no reason to do data parallelism:** pure data parallelism is unhelpful because it replicates our parameters and doesn't help us load parameters faster. You're better off spinning up multiple copies of the model instead.<d-footnote>By this we mean, spin up multiple servers with copies of the model at a smaller batch size. Data parallelism at the model level is strictly worse.</d-footnote>

3. **No sequence = no sequence sharding.** Good luck sequence sharding.

_This mostly leaves us with variants of model sharding for dense model generation_. As with prefill, the simplest thing we we can do is simple model parallelism (with activations fully replicated, weights fully sharded over hidden dimension for the MLP) up to 4-8 ways when we become ICI bound. However, since we are often memory bandwidth bound, we can actually go beyond this limit to improve latency!

**Note on ICI bounds for generation:** during training we want to be compute-bound, so our rooflines look at when our ICI comms take longer than our FLOPs. However, during generation, if we're memory bandwidth bound by parameter loading, we can increase model sharding beyond this point and improve latency at a minimal throughput cost. More model sharding gives us more HBM to load our weights over, and our FLOPs don't matter.<d-footnote>In the sense that FLOPs time isn't bottlenecking us, so the thing we need to worry about is ICI time exceeding parameter loading time.</d-footnote> Let's look at how much model parallelism we can do before it becomes the bottleneck.

$$\begin{align*}T_\text{HBM comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{ICI comms} = \frac{2BD}{W_\text{ici}}\end{align*}$$

$$T_\text{ICI comms} > T_\text{HBM comms} \rightarrow \frac{W_\text{hbm}}{W_\text{ici}} > \frac{F}{Y \cdot B} \rightarrow Y > F / (B \cdot \beta)$$

where $\beta = W_\text{hbm} / W_\text{ici}$. This number is usually around 8 for TPU v5e and TPU v6e. That means e.g. if $F$ is 16,384 and $B$ is 32, we can in theory do model parallelism up to `16384 / (32 * 8) = 64` ways without a meaningful hit in throughput. This assume we can fully shard our KV caches 64-ways which is difficult: we discuss this below.

For the attention layer, we also model shard attention $$W_Q$$ and $$W_O$$ over heads Megatron style. The KV weights are quite small, and replicating them is often cheaper than sharding beyond $K$-way sharding.

<p markdown=1 class="takeaway">**Takeaway:** our only options during generation are variants of model parallelism. We aim to move activations instead of KV caches or parameters, which are larger. When our batch size is large, we do model parallelism up to the FLOPs-ICI bound ($F / \alpha$). When our batch size is smaller, we can improve latency by model sharding more (at a modest throughput cost). When we want to model shard more ways than we have KV heads, we can shard our KVs along the batch dimension as well.</p>

### Sharding the KV cache

**We also have an additional data structure that needs to be sharded — the KV cache.** Again, we almost always prefer to avoid replicating the cache, since it is the primary source of attention latency. To do this, we first Megatron-shard the KVs along the head dimension. This is limited to $K$-way sharding, so for models with a small number of heads, we shard the head dimension as much as possible and then shard along the batch dimension, i.e. $\text{KV}[2, B_Z, S, K_Y, H]$. This means the KV cache is completely distributed.

{% include figure.liquid path="assets/img/esta-figure.png" class="img-fluid" caption="<b>Figure:</b> comparison of the attention mechanism with (a) Multi head attention with pure model sharding and (b) Multiquery attention with batch sharding of the KV cache. Notice how we need two extra AllToAlls to shift the activations from model sharding to batch sharding, so they can act on the KV caches." %}

The cost of this is two AllToAlls every attention layer — one to shift the Q activations to the batch sharding so we can compute attention with batch sharding, and one to shift the batch sharded attention output back to pure model sharded.

{% details Here's the full algorithm! %}

Here we'll write out the full attention algorithm with model parallelism over both $Y$ and $Z$. I apologize for using $K$ for both the key tensor and the KV head dimension. Let $M=N/K$.

<div markdown=1 class="algorithm">

1. X[B, D] = ... (existing activations, unsharded from previous layer)
2. K[B<sub>Z</sub>, S, K<sub>Y</sub>, H], V[B<sub>Z</sub>, S, K, H] = ... (existing KV cache, batch sharded)
3. Q[B, N<sub>YZ</sub>, H] = X[B, D] \* W<sub>Q</sub>[D, N<sub>YZ</sub>, H]
4. Q[B<sub>Z</sub>, N<sub>Y</sub>, H] = **AllToAll**<sub>Z->B</sub>(Q[B, N<sub>YZ</sub>, H])
5. Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = **Reshape**(Q[B<sub>Z</sub>, N<sub>Y</sub>, H])
6. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] \*<sub>H</sub> K[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
7. O[B<sub>Z</sub>, S, K, M] = **Softmax**<sub>S</sub>(O[B<sub>Z</sub>, S, K<sub>Y</sub>])
8. O[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = O[B<sub>Z</sub>, S, K, M] \*<sub>S</sub> V[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
9. O[B, K<sub>Y</sub>, M<sub>Z</sub>, H] = **AllToAll**<sub>Z->M</sub>(O[B<sub>Z</sub>, K<sub>Y</sub>, M, H])
10. O[B, N<sub>YZ</sub>, H] = **Reshape**(O[B, K<sub>Y</sub>, M<sub>Z</sub>, H])
11. X[B, D] {U<sub>YZ</sub>} = W<sub>O</sub>[N<sub>YZ</sub>, H, D] \*<sub>N,H</sub> O[B, N<sub>YZ</sub>, H]
12. X[B, D] = **AllReduce**(X[B, D] { U<sub>YZ</sub>})

This is pretty complicated but you can see generally how it works. The new comms are modestly expensive since they operate on our small activations, while in return we save a huge amount of memory bandwidth loading the KVs (which are stationary).

</div>

{% enddetails %}

* **Sequence sharding:** If the batch size is too small, or the context is long, we can sequence shard the KV cache. Again, we pay a collective cost in accumulating the attention across shards here. First we need to AllGather the Q activations, and then accumulate the KVs in a similar fashion to Flash Attention.

## Designing an Effective Inference Engine

So far we've looked at how to optimize and shard the individual prefill and generate operations efficiently in isolation. To actually use them effectively, we need to design an inference engine which can feed these two operations at a point of our choosing on the latency/throughput Pareto frontier.

The simplest method is simply to run a batch of prefill, then a batch of generations:

{% include figure.liquid path="assets/img/batched-prefill.png" class="img-fluid" caption="<b>Figure:</b> in the simplest setup, requests are aggregated, and the server alternates between running a batch of prefills and calling the generate function until completion for all sequences." %}

This is easy to implement and is the first inference setup in most codebases, but it has multiple drawbacks:

1. **Latency is terrible.** We couple the prefill and generate batch size. Time to first token (TTFT) is terrible at big prefill batch sizes — you need to finish all prefills before any users can see any tokens. Generate throughput is terrible at small batch sizes.
2. **We block shorter generations on longer ones.** Many sequences will finish before others, leaving empty batch slots during generation, hurting generate throughput further. The problem exacerbates as batch size and generation length increases.
3. **Prefills are padded.** Prefills are padded to the longest sequence and we waste a lot of compute. There are solutions for this, but historically XLA made it quite difficult to skip these FLOPs. Again this becomes worse the bigger the batch size and prefill sequence length.
4. **We're forced to share a sharding between prefill and generation.** Both prefill and generate live on the same slice, which means we use the same topology and shardings (unless you keep two copies of the weights) for both and is generally unhelpful for performance e.g. generate wants a lot more model sharding.

Therefore this method is only recommended for edge applications (which usually only cares about serving a single user and using hardware with less FLOPs/byte) and rapid iteration early in the lifecycle of a Transformer codebase (due to its simplicity).

A slightly better approach involves performing prefill at batch size 1 (where it is compute-bound but has reasonable latency) but batch multiple requests together during generation:

{% include figure.liquid path="assets/img/interleaving.png" class="img-fluid" %}

This will avoid wasted TTFT from batched prefill while keeping generation throughput high. We call this an **interleaved** configuration, since we "interleave" prefill and generation steps. This is very powerful for bulk generation applications like evaluations where throughput is the main goal. The orchestrator can be configured to prioritise prefill the moment any generation slots open up, ensuring high utilisation even for very large generation batch sizes. We can also avoid padding our prefill to the maximum length, since it isn't batched with another request.

The main disadvantage is that when the server is performing a prefill, the generation of all other requests pauses since all the compute resources will be consumed by the prefill. User A whose response is busy decoding will be blocked by user B whose prefill is occurring. This means even though TTFT has improved, the token generation will be jittery and slow on average, which is not a good user experience for many applications — other user's prefills are on the critical path of the overall latency of a request.

To get around this, we separate decode and prefill. While Transformer inference can be done on one server, it is often better from a latency standpoint to execute the two different tasks on two sets of TPUs/GPUs. Prefill servers generate KV caches that get sent across the network to the generate servers, which batch multiple caches together and generate tokens for each of them. We call this **"disaggregated"** serving.

{% include figure.liquid path="assets/img/disaggregation.png" class="img-fluid" %}

This provides a few advantages:

1. **Low latency at scale**: A user's request never blocks on another user's, except if there is insufficient prefill capacity. The request should be immediately prefilled, then sent to the generation server, then immediately slotted into the generation buffer. If we expect many concurrent requests to come in, we can scale the number of prefill servers independently from the number of generate servers so users are not left in the prefill queue for an extended period of time.

2. **Specialization:** Quite often, the latency-optimal parameter sharding strategy/hardware topology for prefill and generate is quite different (for instance, more model parallelism is useful for generate but not prefill). Constraining the two operations to use the same sharding hurts the performance of both, and having two sets of weights uses memory. Also, by moving prefill onto its own server, it doesn't need to hold any KV caches except the one it's currently processing. That means we have a lot more memory free for history caching (see the next section) or optimizing prefill latency.

One downside is that the KV cache now needs to be shifted across the network. This is typically acceptable but again provides a motivation for reducing KV cache size.

<p markdown=1 class="takeaway">**Takeaway:** for latency-sensitive, high-throughput serving, we typically have to separate prefill and generation into separate servers, with prefill operating at batch 1 and generation batching many concurrent requests together.</p>

### Continuous Batching

Problem (2) above motivates the concept of **continuous batching**. We optimize and compile:

* A number of prefill functions with variable context lengths and inserts it into some KV buffer, some maximum batch size and context length/number of pages.
* A generate function which takes in the KV cache, and performs the generation step for all currently active requests.

We then combine these functions with an orchestrator which queues the incoming requests, calls prefill and generate depending on the available generate slots, handles history caching (see next section) and streams the tokens out.

{% include figure.liquid path="assets/img/continuous-batching.gif" class="img-fluid" %}

### Prefix Caching

Since prefill is expensive and compute-bound (giving us less headroom), one of the best ways to reduce its cost is to do less of it. Because LLMs are autoregressive, the queries ["I”, "like”, "dogs”] and ["I”, "like”, "cats”] produce KV caches that are identical in the first two tokens. What this means is that, in principle, if we compute the "I like dogs” cache first and then the "I like cats” cache, we only need to do 1 / 3 of the compute. We can save most of the work by reusing the cache. This is particularly powerful in a few specific cases:

1. **Chatbots**: most chatbot conversations involve a back-and-forth dialog that strictly appends to itself. This means if we can save the KV caches from each dialog turn, we can skip computation for all but the newest tokens.
2. **Few-shot prompting**: if we have any kind of few-shot prompt, this can be saved and reused for free. System instructions often have this form as well.

The only reason this is hard to do is memory constraints. As we've seen, KV caches are big (often many GB), and for caching to be useful we need to keep them around until a follow-up query arrives. Typically, any unused HBM on the prefill servers can be used for a local caching system. Furthermore, accelerators usually have a lot of memory on their CPU hosts (e.g. a 8xTPUv5e server has 128GiB of HBM, but around 450GiB of Host DRAM). This memory is much slower than HBM — too slow to do generation steps usually — but is fast enough for a cache read. In practice:

* Because the KV cache is local to the set of TPUs that handled the initial request, we need some form of affinity routing to ensure follow-up queries arrive at the same replica. This can cause issues with load balancing.
* A smaller KV cache is helpful (again) — it enables us to save more KV caches in the same amount of space, and reduce read times.
* The KV cache and their lookups can be stored quite naturally in a tree or trie. Evictions can happen on an LRU basis.

{% include figure.liquid path="assets/img/prefix-caching-trie.png" class="img-fluid" caption="<b>Figure:</b> KV prefix cache implemented as an LRU trie. We can avoid duplicating KV memory by sharing prefixes. Source: <a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai blog</a>." %}

### Let's look at an implementation: JetStream

Google has open-sourced a library that implements this logic called [JetStream](https://github.com/google/JetStream). The server has a set of "prefill engines” and "generate engines”, usually on different TPU slices, which are orchestrated by a single controller. Prefill happens in the "[prefill thread](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L499)”, while generation happens in the "[generate thread](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L629)”. We also have a "[transfer thread](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L592)” that orchestrates copying the KV caches from the prefill to generate slices.

The Engine interface (implemented [here](https://github.com/google/JetStream/blob/445f1aa8e857d0a09d72618e365daf80723bdf4c/jetstream/engine/engine_api.py#L138)) is a generic interface that any LLM must provide. The key methods are:

* **prefill:** takes a set of input tokens and generates a KV cache.
* **insert:** takes a KV cache and inserts it into the batch of KV caches that generate is generating from.
* **generate:** takes a set of batched KV caches and generates one token per batch entry, appending a single token's KV cache to the decode state for each token.

We also have a PyTorch version of JetStream available [here](https://github.com/google/jetstream-pytorch).

## Worked Problems

I'm going to invent a new model based on LLaMA-2 13B for this section. Here are the details:

| hyperparam        | value  |
| ----------------- | ------ |
| n\_layers (L)     | 64     |
| d\_model (D)      | 4,096  |
| d\_ff (F)         | 16,384 |
| n\_heads (N)      | 32     |
| n\_kv\_heads (K)  | 8      |
| d\_qkv (H)        | 256    |
| n\_embeddings (V) | 32,128 |

**Question 1:** How many parameters does the above model have? How large are its KV caches per token? *You can assume we share the input and output projection matrices.*

{% details Click here for the answer. %}

**Parameter count:**

* MLP parameter count: $L * D * F * 3$
* Attention parameter count: $L * 2 * D * H * (N + K)$
* Vocabulary parameter: $D * V$ (since we share these matrices)

Our total parameter count is thus $L * D * (3F + 2H * (N + K)) + D * V$. Plugging in the numbers above, we have `64 * 4096 * (3*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 18.4e9`. Thus, this model has about 18.4 billion parameters.

{% enddetails %}

**Question 2:** Let's say we want to serve this model on a TPUv5e 4x4 slice and can fully shard our KV cache over this topology. What's the largest batch size we can fit, assuming we use int8 for everything. What if we dropped the number of KV heads to 1?

**Question 3:** Let's pretend we're totally HBM bandwidth bound. How long does it take to load all the parameters into the MXU from HBM? *This is a good lower bound on the per-step latency.*

**Question 4:** Let's say we want to serve this model on a TPUv5e 4x4 slice. How would we shard it? *Hint: maybe answer these questions first:*

1. What's the upper bound on tensor parallelism for this model over ICI?
2. How can we shard the KV caches?

For this sharding, what is the rough per-step latency for generation?

**Question 5:** Let's pretend the above model is actually an MoE. An MoE model is effectively a dense model with E copies of the FFW block. Each token passes through k of the FFW blocks and these `k` are averaged to produce the output. Let's use `E=16` and `k=2` with the above settings.

1. How many parameters does it have?
2. What batch size is needed to become FLOPs bound?
3. How large are its KV caches per token (assume no local attention)?
4. How many FLOPs are involved in a forward pass with T tokens?

**Question 6:** With MoEs, we can do "expert sharding”, where we split our experts across one axis of our mesh. In our standard notation, our first FFW weight has shape `[E, D, F]` and we shard it as [E<sub>Z</sub>, D<sub>X</sub>, F<sub>Y</sub>] where `X` is only used during training as our FSDP dimension. Let's say we want to do inference on a TPU v5e:

1. What's the HBM weight loading time for the above model on a TPU v5e 8x16 slice with Y=8, Z=16? How much free HBM is available per TPU?
2. What is the smallest slice we could fit our model on?

**Question 7 [2D model sharding]:** Here we'll work through the math of what the [ESTI paper](https://arxiv.org/pdf/2211.05102) calls 2D weight-stationary sharding. We describe this briefly in Appendix B, but try doing this problem first to see if you can work out the math. The basic idea of 2D weight stationary sharding is to shard our weights along both the $D$ and $F$ axes so that each chunk is roughly square. This reduces the comms load and allows us to scale slightly farther.

Here's the algorithm for 2D weight stationary:

<div markdown=1 class="algorithm">

1.  In[B, D<sub>X</sub>] = **AllGather**<sub>YZ</sub>(In[B, D<sub>XYZ</sub>])
2.  Tmp[B, F<sub>YZ</sub>] {U.X} = In[B, D<sub>X</sub>] \*<sub>D</sub> W<sub>in</sub>[D<sub>X</sub>, F<sub>YZ</sub>]
3.  Tmp[B, F<sub>YZ</sub>] = **AllReduce**<sub>X</sub>(Tmp[B, F<sub>YZ</sub>] {U.X})
4.  Out[B, D<sub>X</sub>] {U.YZ} = Tmp[B, F<sub>YZ</sub>] \*<sub>F</sub> W2[F<sub>YZ</sub>, D<sub>X</sub>]
5.  Out[B, D<sub>XYZ</sub>] = **ReduceScatter**<sub>YZ</sub>(Out[B, D<sub>X</sub>] {U.YZ})
</div>

Your goal is to work out $T_\text{math}$ and $T_\text{comms}$ for this algorithm and find when it will outperform traditional 3D model sharding?

{% details Click here for the answer! %}

Let's work out $T_\text{math}$ and $T_\text{comms}$. All our FLOPs are fully sharded so as before we have $T_\text{math} = 4BDF / (N \cdot C)$ but our comms are now

$$\begin{align*}
T_\text{2D comms} = \frac{2BD}{2X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} + \frac{2BD}{2X \cdot W_\text{ici}} = \frac{2BD}{X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}}
\end{align*}$$

where we note that the AllReduce is twice as expensive and we scale our comms by the number of axes over which each operation is performed. Assuming we have freedom to choose our topology and assuming $F=4D$ (as in LLaMA-2), we claim (by some basic calculus) that the optimal values for $X$, $Y$, and $Z$ are $X = \sqrt{N / 8}$, $YZ = \sqrt{8N}$ so the total communication is

$$T_\text{2D comms} = \frac{2B}{W_\text{ici}} \left(\frac{D}{X} + \frac{8D}{YZ}\right) = \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \approx \frac{11.3 BD}{\sqrt{N} \cdot W_\text{ici}}$$

Firstly, copying from above, normal 1D model parallelism would have $T_\text{model parallel comms} = 4BD / (3 \cdot W_\text{ici})$, so when are the new comms smaller? We have

$$\begin{align*}
T_\text{model parallel comms} > T_\text{2D comms} \iff \frac{4BD}{3 \cdot W_\text{ici}} > \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \\
\iff N > 128 \cdot \left(\frac{3}{4}\right)^2 = 81
\end{align*}$$

For a general $F$, we claim this condition is

$$N > 32 \cdot \left(\frac{F}{D}\right) \cdot \left(\frac{3}{4}\right)^2$$

So that tells us if we have more than 81 chips, we're better off using this new scheme. Now this is a slightly weird result because we've historically found ourselves ICI bound at around ~20 way tensor parallelism. But here, even if we're communication-bound, our total communication continues to decrease with the number of total chips! What this tells us is that we can continuous to increase our chips, increase our batch size, do more parameter scaling, and see reduced latency.

{% enddetails %}

<h3 markdown=1 class="next-section">That's all for Part 7! For Part 8, with a look at how we might serve LLaMA 3 on TPUs, click [here](../applied-inference).</h3>

## Appendix

### Appendix A: How real is the batch size > 240 rule?

The simple rule we provided above, that our batch size must be greater than 240 tokens to be compute-bound, is roughly true but ignores some ability of the TPU to prefetch the weights while other operations are not using all available HBM, like when doing inter-device communication.

Here's an empirical plot of layer time (in microseconds) for a small Transformer with d<sub>model</sub> 8192, d<sub>ff</sub> 32768, and only 2 matmuls per layer. This comes from [this Colab notebook](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing). You'll see that step time increases very slowly up until around batch 240, and then increases linearly.

{% include figure.liquid path="assets/img/batch-scaling-latency.png" class="img-fluid img-small" %}

Here's the actual throughput in tokens / us. This makes the argument fairly clearly. Since our layer is about 600M parameters sharded 4 ways here, we'd expect a latency of roughly 365us at minimum.

{% include figure.liquid path="assets/img/batch-scaling-throughput.png" class="img-fluid img-small" %}

So at least in this model, we do in fact see throughput increase until about BS240 per data parallel shard.

### Appendix B: 2D Weight Stationary sharding

As the topology grows, if we have access to higher dimensional meshes (like that of TPUs) it is possible to refine this further with "**2D Weight Sharding”**. By introducing a second sharding axis. We call this "**2D Weight Stationary**”, and was described in more detail in the [Efficiently Scaling Transformer Inference paper](https://arxiv.org/abs/2211.05102).

Because we're only sharding the hidden $$F$$ dimension in Megatron, it can become significantly smaller than $$E$$ (the $$d_\text{model}$$ dimension) once the number of chips grows large with 1D sharding. This means at larger batch sizes, it can be more economical to perform a portion of the collectives over the hidden dimension after the first layer of the MLP is applied.

{% include figure.liquid path="assets/img/2d-weight-stationary.png" class="img-fluid img-small" %}

This figure shows:

1. 1D weight-stationary sharding, a.k.a. Pure Megatron sharding, where activations are fully replicated after AllGather, and weights are fully sharded over the hidden F dimension.
2. 2D weight stationary sharding, where weights are sharded over both the hidden F and reduction E dimension, and activations are sharded over the E dimension. We perform an AllGather on the (yz) axis before the first layer, then ReduceScatter on the (x) axis.

For the attention layer, Megatron style sharding is also relatively simple for smaller numbers of chips. However, Megatron happens over the $$n_\text{heads}$$ dimension, which puts a limit on the amount of sharding that is possible. Modifying the 2D sharding with for (instead of sharding the hidden, we shard the $$n_\text{heads}$$ dimension), we gain the ability to scale further.

### Appendix C: Latency bound communications

As a recap, in [Section 3](../sharding) we derived the amount of time it takes to perform an AllGather into a tensor of size B on each TPU, over X chips on a 1D ring links of full duplex bandwidth of WICI and latency Tmin.

$$T_{total} = \max\left(\frac{T_{min} \cdot |X|}{2}, \frac{B}{W_{ICI}}\right)$$

For large B, the wall clock stays relatively constant because as you add more chips to the system, you simultaneously scale the amount of data movement necessary to perform the operation and the total bandwidth available.

{% include figure.liquid path="assets/img/all-gather.gif" class="img-fluid" %}

Because of the relatively low amounts of data being moved during latency optimized inference, collectives on activations are often bound by the latency term (especially for small batch sizes). One can visualise the latency quite easily, by counting the number of hops we need to complete before it is completed.

On TPUs, if the tensor size-dependent part of communication is less than 1 microsecond per hop (a hop is communication between two adjacent devices) we can be bottlenecked by the fixed overhead of actually dispatching the collective. With `4.5e10` unidirectional ICI bandwidth, ICI communication becomes latency bound when: $$(\text{bytes} / n_\text{shards}) / 4.5e10 < 1e-6$$. For 8-way Megatron sharding, this is when `buffer_size < 360kB`. **This actually is not that small during inference:** with `BS=16` and `D=8192` in int8, our activations will use `16*8192=131kB`, so we're already latency bound.

<p markdown=1 class="takeaway">**Takeaway:** our comms become latency bound when $$\text{total bytes} < W_{ICI} \times 1e-6$$. For instance, with model parallelism over $$Y$$, we become bound in int8 when $$Y > BD / 45,000$$.</p>

There's a parallel to be drawn here with the compute roofline — we are incurring the fixed cost of some small operations (latency for comms, memory bandwidth for matmuls).

### Appendix D: Speculative Sampling

When we *really* care about end to end latency, there is one extra trick we can employ called speculative sampling<d-cite key="spec1"></d-cite><d-cite key="spec2"></d-cite>. As a recap, we usually generate tokens from a large Transformer one by one:

{% include figure.liquid path="assets/img/spec-sampling1.png" class="img-fluid" %}

With speculative sampling, we use a smaller, cheaper model to generate tokens and then check the result with the big model. This is easiest to understand with *greedy decoding*:

{% include figure.liquid path="assets/img/spec-sampling2.png" class="img-fluid" %}

1. We sample greedily from some smaller, cheaper model. Ideally we use a model trained to match the larger model, e.g. by distillation, but it could be as simple as simply using n-grams or token matching a small corpus of text.
2. After we've generated K tokens, we use the big model to compute the next-token logits for all the tokens we've generated so far.
3. Since we're decoding greedily, we can just check if the token generated by the smaller model has the highest probability of all possible tokens. If one of the tokens is wrong, we take the longest correct prefix and replace the first wrong token with the correct token, then go back to (1). If all the tokens are correct, we can use the last correct logit to sample an extra token before going back to (1).

**Why is this a latency win?** This scheme still requires us to do the FLOPs-equivalent of one forward pass through the big model for every token, but because we can batch a bunch of tokens together, we can do all these FLOPs in one forward pass and take advantage of the fact that we're *not* *compute-bound* to score more tokens for free.

Every accepted token becomes more expensive in terms of FLOPs on average (since some will be rejected, and we have to call a draft model), but we wring more FLOPs out of the hardware, and the small model is cheap, so we win overall. Since everything has been checked by the big model, we don't change the sampling distribution at all (though the exact trajectory will differ for non-greedy).

For normal autoregressive sampling the token/s is the same as the step time. We are still beholden to the theoretical minimum step time according to the Arithmetic Intensity section here (in fact, Speculative Sampling step times are usually quite a bit slower than normal autoregressive sampling, but because we get more than 1 token out per step on average we can get much better tokens/s).

{% include figure.liquid path="assets/img/spec-sampling3.png" class="img-fluid" caption="<b>Figure:</b> this figure shows the per-step latency and speculation success rate for Chinchilla (a 70B model from DeepMind) with a 4B parameter drafter (small model). For XSum (a natural language dataset), the ideal amount of speculation is about 3-4 tokens ahead, while HumanEval (a coding dataset) is more predictable and sees wins from more aggressive speculation." %}

**How does this work for non-greedy decoding?** This is a bit more complicated, but essentially boils down to a Metropolis-Hastings inspired algorithm where have $$P_{\text{draft model}}(\text{chosen token})$$ and $$P_{\text{target model}}(\text{chosen token})$$ derived from the logits, and reject the chosen token probabilistically if the ratio of these probabilities is smaller than some threshold.

These [two](https://arxiv.org/abs/2211.17192) [papers](https://arxiv.org/abs/2302.01318) derived this concurrently and have good examples of how this works in practice.

<p markdown=1 class="takeaway">**Takeaway:** Speculative sampling is yet another powerful lever for trading throughput for better per token latency. However, in the scenario where batch size is limited (e.g. small hardware footprint or large KV caches), it becomes a win-win.</p>
