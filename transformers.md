---
layout: distill
title: "All the Transformer Math You Need to Know"
# permalink: /main/
description: "Here we'll do a quick review of the Transformer architecture, specifically how to calculate FLOPs, bytes, and other quantities of interest."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 4

previous_section_url: "../sharding"
previous_section_name: "Part 3: Sharding"

next_section_url: ../training
next_section_name: "Part 5: Training"

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

bibliography: main.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "Counting Dots"
  - subsections:
    - name: "Forward and reverse FLOPs"
  - name: "Transformer Accounting"
  - name: "Global FLOPs and Params Calculation"
  - name: "Miscellaneous Math"
  - subsections:
    - name: "Sparsity and Mixture-of-Experts"
    - name: "Gradient checkpointing"
    - name: "Key-Value (KV) caching"
  - name: "What Should You Take Away from this Section?"
  - name: "A Few Problems to Work"
  - name: "Appendix"
  - subsections:
    - name: "Appendix A: How does Flash Attention work?"

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

## Counting Dots

Let's start with vectors $$x$$,$$y$$ and matrices $$A$$,$$B$$ of the following shapes:

$$
\def \red#1{\textcolor{red}{#1}}
\def \green#1{\textcolor{green}{#1}}
\def \blue#1{\textcolor{blue}{#1}}
\def \purple#1{\textcolor{purple}{#1}}
\def \orange#1{\textcolor{orange}{#1}}
\def \gray#1{\textcolor{gray}{#1}}

\begin{array}{cc}
\textrm{array}  & \textrm{shape} \\ \hline
x               & \textrm{[P]}   \\
y               & \textrm{[P]}   \\
A               & \textrm{[N P]} \\
B               & \textrm{[P M]} \\
\hline
\end {array}
$$

- A dot product of $$x \cdot y$$ requires $$P$$ _adds_ and _multiplies_, or $$2P$$ floating-point operations total.
- A matrix-vector product $$Ax$$ does $$N$$ dot-products along the rows of $$A$$, for $$2NP$$ FLOPs.
- A matrix-matrix product $$AB$$ does a matrix-vector product for each of the $$M$$ columns of $$B$$, for $$2NPM$$ FLOPs total.
- In general, if we have two higher dimensional arrays $$C$$ and $$D$$, where some dimensions are <span style="color:red">CONTRACTING</span> and some are <span style="color:blue">BATCHING</span>.  (e.g. $$C[\blue{GH}IJ\red{KL}], D[\blue{GH}MN\red{KL}]$$) then the FLOPs cost of this contraction is two times the product of all of the $$C$$ and $$D$$ dimensions where the batch and contraction dimensions are only counted once, (e.g. $$2\blue{GH}IJMN\red{KL}$$). Note that a dimension is only batching if it occurs in both multiplicands. (Note also that the factor of 2 won't apply if there are no contracting dimensions and this is just an elementwise product.)

$$
\begin{array}{ccc}
\textrm{Operation} & \textrm{FLOPs} & \textrm{Data} \\
\hline
x \cdot y  & 2P   & 2P      \\
A x        & 2NP  & NP + P  \\
AB         & 2NPM & NP + PM \\
[c_0,...,c_N] \cdot [d_0,...,d_N] &
2 \prod c_i \times \prod_{\substack{d_j \notin \blue{BATCH} \\ d_j \notin \red{CONTRACT}}} d_j
&
  \prod c_i + \prod d_j \\
\hline
\end {array}
$$

Make note of the fact that for a matrix-matrix multiply, the *compute* scales cubically $$O(N^3)$$ while the data transfer only scales quadratically $$O(N^2)$$ \- this means that as we scale up our matmul size, it becomes *easier* to hit the compute-saturated limit. This is extremely unusual, and explains in large part why we use architectures dominated by matrix multiplication \- they're amenable to being scaled!

{% include figure.liquid path="assets/img/matmul-flops.gif" class="img-fluid" %}

### Forward and reverse FLOPs

During training, we don't particularly care about the result of a given matrix multiply; we really care about its derivative. That means we do significantly more FLOPs during backpropagation.

If we imagine **B** is just one matrix in a larger network and **A** are our input activations with **C = A B**, the derivative of the loss **L** with respect to **B** is given by the chain rule:

$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial C}\right)$$

which is an outer product and requires $2NPM$ FLOPs to compute (since it contracts over the $N$ dimension). Likewise, the derivative of the loss with respect to **A** is

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial C}\right) B^T$$

is again $2NPM$ FLOPs since **dL/dC** is a (co-)vector of size $$[N, M]$$. While this quantity isn't the derivative wrt. a parameter, it's used to compute derivatives for previous layers of the network (e.g. just as dL/dC is used to compute dL/dB above).

Adding these up, we see that **during training, we have a total of 6NPM FLOPs**, compared to 2NPM during inference: 2NPM in the forward pass, 4NPM in the backward pass. Since PM is the number of parameters in the matrix, this is the simplest form of the famous $$6 * \text{num parameters} * \text{num tokens}$$ approximation of Transformer FLOPs during training: each token requires $$6 * \text{num parameters}$$ FLOPs. We'll show a more correct derivation below.

## Transformer Accounting

Transformers are the future. Well, they're the present at least. Maybe a few years ago, they were one of many architectures. But today, it's worth knowing pretty much every detail of the architecture. We won't reintroduce the architecture but [this blog](https://jalammar.github.io/illustrated-transformer/) and the [original Transformer paper](https://arxiv.org/abs/1706.03762) may be helpful references.

Here's a basic diagram of the Transformer decoder architecture:

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>Figure:</b> this diagram shows one layer of a standard Transformer and flows from top-to-bottom. We use a single-letter convention to describe the shapes and layouts of arrays in a Transformer, again showing contracting dimensions in red, and batched dimensions in blue. In a given operation, the input shape is given on top-left and the parameter shape is given on the top-right, with the resulting shape below, e.g. BTD is the input shape for the gating einsum and DF is the weight shape." %}

**Note [gating einsum]**: The diagram above uses a "[gating einsums](https://arxiv.org/abs/2002.05202)”<d-cite key="glu"></d-cite> where we split the up-projection matrix into two matrices ($W_\text{In1}$ and $W_\text{In2}$ above) whose outputs are elementwise multiplied as a kind of "gating function”. Not all LLMs use this, so you will sometimes see a single $W_\text{In}$ matrix and a total MLP parameter count of 2DF instead of 3DF. Typically in this case, D and F will be scaled up to keep the parameter count the same as the 3 matrix case. With that said, some form of gating einsum is used by LLAMA, DeepSeek, and many other models.

**Note 2 [MHA attention]**: With self-attention, T and S are the same but for cross-attention they may be different. With vanilla Multi-Head Attention (MHA), N and K are the same while for [Multi-Query Attention](https://arxiv.org/abs/1911.02150) (MQA)<d-cite key="mqa"></d-cite> K=1 and for [Grouped MQA](https://arxiv.org/abs/2305.13245) (GMQA)<d-cite key="gmqa"></d-cite> K merely has to divide N.

## Global FLOPs and Params Calculation

For the below we're going to compute per-layer FLOPs to avoid having to stick factors of **L** everywhere.

### MLPs


The MLPs of a Transformer typically consist of 2 input matmuls that are element-wise combined and a single output matmul:

$$
\begin{array}{ccc}
\textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\
\hline \\
A[B,T,\red{D}] \cdot W_{in1}[\red{D}, F] & 6BTDF & DF \\[10pt]
A[B,T,\red{D}] \cdot W_{in2}[\red{D}, F] & 6BTDF & DF \\[10pt]
\sigma\left(A_{in1}\right)[B,T, F] * A_{in2}[B,T, F] & \gray{O(BTF)} \\[10pt]
A[B,T,\red{F}] \cdot W_{out}[\red{F}, D] & 6BTDF & DF \\[10pt]
\hline \\
& \approx 18BTDF & 3DF
\end{array}
$$

### Attention

For the generic grouped-query attention case with different **Q** and **KV** head numbers, let us assume equal head dimension H for **Q**,**K**,**V** projections, and estimate the cost of the **QKVO** matmuls:

$$
\begin{array}{ccc}
\textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\
\hline \\
A[B,T,\red{D}] \cdot W_{Q}[\red{D}, N, H] & 6BTDNH & DNH \\[10pt]
A[B,T,\red{D}] \cdot W_{K}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{D}] \cdot W_{V}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{N}, \red{H}] \cdot W_{O}[\red{N}, \red{H}, D] & 6BTDNH & DNH \\[10pt]
\hline \\ & 12BTD(N+K)H & 2D(N+K)H
\end{array}
$$

The dot-product attention operation is more subtle, effectively being a $$TH \cdot HS$$ matmul batched over the $$B$$, $$K$$ dimensions, a softmax, and a $$TS \cdot SH$$ matmul again batched over the $$B$$, $$K$$ dimensions. We highlight the batched dims in blue:

$$
\begin{array}{cc}
\textrm{operation} & \textrm{train FLOPs} \\
\hline \\[3pt]
Q[\blue{B}, T, \blue{K}, G, \red{H}] \cdot K[\blue{B}, S, \blue{K}, \red{H}]
& 6BTSKGH = 6BTSNH  \\[3pt]
\textrm{softmax}_S \;\; L[B, T, S, K, G] & \gray{O(BTSKG) = O(BTSN)} \\[3pt]
S[\blue{B}, T, \red{S}, \blue{K}, G] \cdot V[\blue{B}, \red{S}, \blue{K}, H]
& 6BTSKGH = 6BTSNH \\[3pt]
\hline \\
& \approx 12BTSNH = 12BT^2NH \\
\end{array}
$$

**Note [causal masking]**: Most recent transformers use a causal mask as opposed to full bidirectional attention. In this case the useful FLOPs of the dot product operations are reduced by a factor of 1/2. To achieve this reduction in practice we need to make use of an attention kernel, rather than a naive einsum.

### Other Operations

There are several other operations happening in a Transformer.  Layernorms are comparatively cheap and can be ignored for first-order cost estimates. There is also the final enormous (though not per-layer) unembedding matrix multiply.

$$
\begin{array}{ccc}
\textsf{operation} & \textsf{train FLOPs} & \textsf{params} \\
\hline \\
\textrm{layernorm}_D \;\; A[B,T,\red{D}] & \gray{O\left(BTD\right)} & \gray{D} \\[10pt]
A[B,T,\red{D}] \cdot W_{unembed}[\red{D}, V] & 6BTDV & DV \\
\end{array}
$$

### General rule of thumb for Transformer FLOPs

If we neglect the cost of dot-product attention for shorter-context training, then the total FLOPs across all layers is

$$
\begin{align*}
(18BTDF + 12BTD(N+K)H)L = 6 *BT * (3DF + 2D(N+K)H)L \\ = 6 * \textrm{num tokens} * \textrm{parameter count}
\end{align*}
$$

Leading to a famous rule of thumb for estimating dense Transformer FLOP count, ignoring the attention FLOPs. (Unembedding is another simple matmul with $6BSDV$ FLOPs and $DV$ params, and follows the same rule of thumb.)

### Fractional cost of attention with context length

If we do account for dot-product attention above and assume $$F=4D$$, $$D=NH$$ (as is typical) and $$N=K$$:

$$\small{\frac{\textrm{attention FLOPs}}{\textrm{matmul FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4*18 BTD^2 + 24 BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}$$

So the takeaway is that **dot-product attention FLOPs only become dominant during training once T>8D**. For D ~ 8k, this would be ~64K tokens. This makes some sense, since it means as the MLP size increases, the attention FLOPs become less critical. For large models, the quadratic cost of attention is not actually a huge obstacle to longer context training. However, for smaller models, even e.g. Gemma-27B, D=4608 which means attention becomes dominant around 32k sequence lengths. Flash Attention also helps alleviate the cost of long-context, which we discuss briefly [in Appendix A](#appendix-a-how-does-flash-attention-work).

## Miscellaneous Math

### Sparsity and Mixture-of-Experts

We'd be remiss not to briefly discuss Mixture of Experts (MoE) models<d-cite key="moe"></d-cite>, which replace the single dense MLP blocks in a standard Transformer with a set of independent MLPs that can be dynamically routed between. To a first approximation, **an MoE is just a normal dense model with E MLP blocks per layer**, instead of just one. Each token activates $k$ of these experts, typically $k=2$. This increases the parameter count by $O(E)$, while multiplying the total number of activated parameters per token by $k$, compared with the dense version.

{% include figure.liquid path="assets/img/moe.png" class="img-fluid img-small" caption="<b>Figure:</b> an example MoE layer with $n$ experts. The gating expert routes each token to $k$ of them, and the output of those $k$ MLPs get summed. Our parameter count is $n$ times the size of each expert, but only $k$ are used for each token. <a href=\"https://deepgram.com/learn/mixture-of-experts-ml-model-guide\">Source</a>." %}

Compared to a dense model, an MoE introduces new comms, primarily two AllToAlls (one before and one after the MoE block) that route tokens to the correct expert and bring them back to their home device.<d-footnote>Technically, this only happens if we are data or sequence sharded along the same axis as our experts.</d-footnote> However as we saw in the previous section, the cost of each AllToAll is only 1/4 that of a comparable AllGather along a single axis (for a bidirectional ring).

### Gradient checkpointing

Backpropagation as an algorithm trades memory for compute. Instead of a backward pass requiring $$O(n_\text{layers}^2)$$ FLOPs, **it requires $$O(n_\text{layers})$$ memory**, saving all intermediate activations generated during the forward pass. While this is better than quadratic compute, it's incredibly expensive memory-wise: a model with $$B * T=4M$$ (4M total tokens per batch), L=64, and D=8192 that avoids all unnecessary backward pass compute would have to save roughly $$2 * 20 * B * T * D * L = 84TB$$ of activations in bfloat16. 20 comes from (roughly) counting every intermediate node in the Transformer diagram above, since e.g.

$$f(x) = \exp(g(x))$$

$$\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}$$

so to avoid recomputing we need to save $$g(x)$$ and $$\exp(g(x))$$ from the forward pass. To avoid saving this much memory, we can choose to only save some fraction of the intermediate activations. Here are a few strategies we use.

* **Block remat**: only save the input to each layer. This is the most aggressive method we use and only saves 1 checkpoint per layer, meaning we'd only save 4.2TB in the example above. This forces us to repeat essentially all forward pass FLOPs in the backward pass, meaning we increase our FLOPs from $$6ND$$ to roughly $$8ND$$.
* **Big matmuls only:** another simple policy is to only save the outputs of large matmuls. This lets us avoid recomputing any large matmuls during the backward pass, but still makes us recompute other activation functions and parts of attention. This reduces 20 per layer to closer to 7 per layer.

This by no means comprehensive. When using JAX, these are typically controlled by `jax.remat`/`jax.checkpoint` (you can read more [here](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)).

### Key-Value (KV) caching

As we'll see in [Section 7](../inference), LLM inference has two key parts, prefill and generation.

* **Prefill** processes a long prompt and saves its attention activations in a Key-Value Cache (KV Cache) for use in generation, specifically the key-value projections in the attention block.
* **Generation** batches several of these KV caches together and samples tokens from each of them.

Each KV cache is then effectively an array of size $[2, S, L, K, H]$ where the 2 accounts for the keys and values. This is quite large! The total size of the Key-Value cache in int8 is $2SLKH$. For a moderately-sized model with 8k context length, 64 layers, and $KH = NH = D = 8192$, this is $2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}$. You can see why we would want to use GMQA with $K \ll N$.

## What Should You Take Away from this Section?

* The overall parameters and FLOPs of a Transformer are fairly easy to calculate, and are summarized here, assuming MHA (with batch size B, vocab size V, a sequence of length T, D=d<sub>model</sub>, and F=d<sub>ff</sub>):


<!-- $$
\begin{array}{ccc}
\textrm{Component} & \textrm{Params per layer} & \textrm{Training FLOPs per layer} \\
\hline \\
\textbf{MLP} & 3DF & 18BTDF \\[10pt]
\textbf{Attention} & 4DNH & 24BTDNH + 12BT^2NH \\[10pt]
\textbf{Other} & D & BTD \\[10pt]
\textbf{Vocab} & DB \text{ (total, not per-layer)} & 12BTDV \\[10pt]
\end{array}
$$ -->


| Component     | Params per layer          | Training FLOPs per layer      |
| :------------ | :------------------------ | :---------------------------- |
| **MLP**       | 3DF                       | 18BTDF                        |
| **Attention** | 4DNH                      | 24BTDNH \+ 12BT<sup>2</sup>NH |
| **Other**     | D                         | BTD                           |
| **Vocab**     | DV (total, not per-layer) | 12BTDV                        |

* The parameter count of the MLP block dominates the total parameter count and the MLP block also dominates the FLOPs budget as long as the sequence length $T < 8D$.
* The total FLOPs budget during training is well approximated by $$6 \cdot \text{num_params} \cdot \text{num_tokens}$$ for reasonable context lengths.
* During inference, our KV caches are roughly $$2 \cdot S \cdot L \cdot N \cdot H$$ per cache, although architectural modifications can often reduce this.

## A Few Problems to Work

**Question 1:** How many parameters does a model with $D=4096$, $F=4 \cdot D$, $V=32,000$, and $L=64$ have? What fraction of these are attention parameters? How large are our KV caches per token? *You can assume $N\cdot H=D$ and multi-head attention with int8 KVs.*

{% details Click here for the answer. %}

1. The total parameters is roughly $$L \cdot (3DF + 4DNH + D) + 2DV$$. For the given numbers, this is $$64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9$$, or 16B parameters.
2. The ratio of attention parameters to total parameters in general is $$4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4$$. This gives us roughly 1/4 of parameters are used in attention.
3. Per token, our KV caches are $$2 \cdot L \cdot N \cdot H = 2 \cdot 64 \cdot 4096$$ in int8, which is `512kB / token`.

{% enddetails %}

**Question 2:** How many total FLOPs are required to perform A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] on `{‘X': 4, ‘Y': 8, ‘Z': 4}`. How many FLOPs are performed by each TPU?

{% details Click here for the answer. %}

The total "theoretical” FLOPs of the operation is $$2 \cdot B \cdot D \cdot F$$. However, because the computation isn't sharded across the Z dimension, we're actually doing Z extra FLOPs, meaning $$2 \cdot B \cdot D \cdot F \cdot Z$$ total FLOPs. Since the computation is sharded across the other dimensions, the total per-device is roughly $$2 \cdot B \cdot D \cdot F / (X \cdot  Y)$$.

{% enddetails %}

**Question 3:** How many FLOPs are involved in performing $A[I,J,K,L] * B[I,J,M,N,O] \rightarrow C[K,L,M,N,O]$?

{% details Click here for the answer. %}

Following the rule above, we have I and J as contracting dimensions and K, L, M, N, and O as non-contracting dimensions. We have no "batching dimensions”, so this is just $$2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O$$, the sum of all the axes. If we had a shared axis, it would only be counted once.

{% enddetails %}

**Question 4:** What is the arithmetic intensity of self-attention (ignoring the Q/K/V/O projections)? *Give the answer as a function of the Q and KV lengths T and S.* At what context length is attention FLOPs-bound? Given the HBM bandwidth of our TPUs, plot the effective relative cost of attention to the FFW block as the context length grows.

{% details Click here for the answer. %}

Self-attention requires loading the $$Q$$, $$K$$, and $$V$$ activations, then computing $$\text{softmax}(Q \cdot K) \cdot V$$, then writing the result back to HBM. This will be done with Flash Attention so there are some caveats to this math, but basically in bf16 self-attention performs

$$\text{Q[B,T,N,H]} \rightarrow_\text{reshape} \text{Q[B, T, K, G, H]} \cdot \text{K[B, S, K, H]} \rightarrow \text{O[B, T, S, K, G]}$$

$$U=\text{softmax}_S(\text{O[B, T, S, K, G]})$$

$$\text{U[B, T, S, K, G]} \cdot \text{V[B, S, K, H]} \rightarrow \text{X[B, T, K, G, H]}$$

So our total bytes is $$2 * \text{sizeof}(Q) + 2 * \text{sizeof(K or V)} = 4BTNH + 4BSKH = 4BHK * (TG + S)$$, total FLOPs is $$4BTSNH + O(BTSN)$$ and the arithmetic intensity is $$4BTSKGH / (4BHK * (TG + S))$$.

So basically, during prefill we have $$S=T$$ so we have an arithmetic intensity of $$4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)$$. During generation, $$T=1$$ so we have $$4BSKGH / (4BHK \cdot (G + S)) = SG / (G + S) \rightarrow G$$ assuming $$S$$ is very large. Depending on how you interpret the question, during prefill or training self-attention is compute bound at S=240 assuming no sequence sharding. During generation, we are never compute bound because $$G$$ is small. Nonetheless, however, you can see that increasing $$G$$ leads to us being closer to compute bound.

{% enddetails %}

**Question 5:** At what sequence length are self-attention FLOPs equal to the QKVO projection FLOPs?

{% details Click here for the answer. %}

This is purely a question of when $$24BTDNH == 12BT^2NH$$. Simplifying we get $$2D = T$$, so e.g. for $$D=4096$$, this is $$8192$$. This tells us that for most reasonable context lengths, matmul FLOPs are greater.

{% enddetails %}

**Question 6:** Say we only save the output of each of the 7 main matmuls in a Transformer layer during our forward pass (Q, K, V, O \+ the three FFW matrices). How many extra FLOPs do we need to "rematerialize” during the backwards pass?

{% details Click here for the answer. %}

Saving only the seven matmul outputs (Q, K, V, O, W₁, W₂, W₃) means the backward pass must recompute the two attention matmuls

$$QK^{\top} \quad\text{and}\quad \operatorname{softmax}(QK^{\top})V.$$

Each is a $T \times T$ matmul batched over $B$ sequences and $N$ heads, so the additional FLOPs are

$$4 \; B \, T^{2} \, N \, H.$$

All other recomputed operations are only $O(BTD)$.

{% enddetails %}

**Question 7:** DeepSeek v3 says it was trained for 2.79M H800 hours on 14.8T tokens ([source](https://arxiv.org/pdf/2412.19437v1)). Given that it has 37B activated parameters, roughly what hardware utilization did they achieve? *Hint: note that they used FP8 FLOPs without structured sparsity.*

{% details Click here for the answer. %}

From the spec sheet [here](https://lenovopress.lenovo.com/lp1814.pdf), we find 3,026 TFLOPs/s of FP8 performance with sparsity, or typically half this (`1.513e15` FLOPs/s) without sparsity. 2.79M H800 hours means `2.79e6 * 1.513e15 * 60 * 60 = 1.52e25` total FLOPs. Given the activated parameter count of 37B, this training run should have used about `6 * 37e9 * 14.8e12 = 3.3e24` FLOPs. That means the FLOPs utilization is about `3.3e24 / 1.52e25 = 21.7%`.

{% enddetails %}

**Question 8:** Mixture of Experts (MoE) models have $E$ copies of a standard dense MLP block, and each token activates $k$ of these experts. What batch size in tokens is required to be compute-bound for an MoE with weights in int8 on TPU v5e? For DeepSeek, which has 256 (routed) experts and $k=8$, what is this number?

{% details Click here for the answer. %}

Because we have $E$ copies of each expert, in int8, we need to load $E \cdot D \cdot F$ bytes. Because each token activates $k$ experts, we have $2\cdot k \cdot B \cdot D \cdot F$ FLOPs. To be compute-bound with bfloat16 FLOPs, we need an arithmetic intensity over 240 which happens when $(2\cdot k \cdot BDF) / EDF > 240$ or $k \cdot B / E > 120$.

Therefore, we need $B > 120 \cdot E / k$ to be compute bound. For DeepSeek, this gives us $B > 120 \cdot 256 / 8 = 3840$. This is a remarkably large batch size at generation time.

{% enddetails %}

<h3 markdown=1 class="next-section">That's it for Part 4! For Part 5 (about scaling Transformer training), [click here](../training)!</h3>

## Appendix

### Appendix A: How does Flash Attention work?

The traditional objection to scaling Transformers to very long context is that the attention FLOPs and memory usage scale quadratically with context length. While it's true that the attention QK product has shape $[B, S, T, N]$ where B is the batch size, S and T are the Q and K sequence dims, and N is the number of heads, this claim comes with some serious caveats:

1. As we noted in Section 4, even though this is quadratic, the attention FLOPs only dominated when $$S > 8 \cdot D$$, and especially during training the memory of a single attention matrix is small compared to all of the weights and activation checkpoints living in memory, especially when sharded.
2. We don't need to materialize the full attention matrix in order to compute attention! We can compute local sums and maxes and avoid ever materializing more than a small chunk of the array. While the total FLOPs is still quadratic, we drastically reduce memory pressure.

This second observation was first made by [Rabe et al. 2021](https://arxiv.org/abs/2112.05682) and later in the [Flash Attention paper](https://arxiv.org/abs/2205.14135) (Dao et al. 2022). The basic idea is to compute the attention in chunks of K/V, where we compute the local softmax and some auxiliary statistics, then pass them onto the next chunk which combines them with its local chunk. Specifically, we compute

1. **M:** The running max of $$q \cdot k$$ over the sequence dimension
2. **O:** The running full attention softmax over the sequence dimension
3. **L:** The running denominator $$\sum_i (q \cdot k_i - \text{running max})$$

With these, we can compute the new max, the new running sum, and the new output with only a constant amount of memory. To give a sketchy description of how this works, attention is roughly this operation:

$$\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}$$

The max is subtracted for numerical stability and can be added without affecting the outcome since $$\sum_i \exp(a_i + b) = \exp(b) \sum \exp(a)$$. Looking just at the denominator above,  if we imagine having two contiguous chunks of key vectors, $$K^1$$ and $$K^2$$ and we compute the local softmax sums $$L^1$$ and $$L^2$$ for each

$$L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)$$

$$L^2 = \sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^2)$$

Then we can combine these into the full softmax sum for these two chunks together by using

$$L^\text{combined} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 - \max(M^1, M^2)) \cdot L^2$$

where

$$M^1 = \max_j Q \cdot K_j^1 \text{ and } M^2 = \max_j Q \cdot K_j^2$$

This can be done for the full softmax as well, giving us a way of accumulating arbitrarily large softmax sums. Here's the full algorithm from the Flash Attention paper.

{% include figure.liquid path="assets/img/flash-algo.png" class="img-fluid" %}

From a hardware standpoint, this lets us fit our chunk of Q into VMEM (what the algorithm above calls on-chip SRAM) so we only have to load the KV chunks on each iteration, reducing the arithmetic intensity. We can also keep the running statistics in VMEM.

One last subtle point worth emphasizing is an attention softmax property that's used to make the Flash VJP (reverse mode derivative) calculation practical for training.  If we define an intermediate softmax array as:

$$S_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_j}}$$

In attention, we obtain *dS* from reverse-mode *dO* and *V* arrays:

$$dS_{ij} = dO_{id} \cdot_d V_{jd} = \sum_d dO_{id} V_{jd}$$

During the backpropagation of this gradient to Q and K

$$d(q_i \cdot k_j) = (dS_{ij} - S_{ij} \cdot_j dS_{ij}) S_{ij}$$

We exploit an identity that allows us to exchange a contraction along the large key **length** dimension with a local contraction along the feature **depth** dimension.

$$\begin{align*}
S_{ij} \cdot_j dS_{ij} &= \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO_{id} V_{jd} \\
&= \sum_d dO_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} V_{jd} \\
&= \sum_d dO_{id} O_{id} \\
&= dO_{id} \cdot_d O_{id}
\end{align*}$$

This replacement is crucial for being able to implement a sequence-block *local* calculation for the VJP, and enables further clever sharding schemes like ring attention.
