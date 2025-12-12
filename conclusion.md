---
layout: distill
title: "Conclusions and Further Reading"
# permalink: /main/
description: "Thank you for reading! Here we'll include a few more references for further study."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 11

previous_section_url: "../jax-stuff"
previous_section_name: "Part 10: JAX"

next_section_url: "../gpus"
next_section_name: "Part 12: GPUs"

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
  - name: "Acknowledgments"
  - name: "Further Reading"
  - name: "Feedback"

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
  .algorithm {
    padding: 10px;
    margin-top: 5px;
    margin-bottom: 5px;
    border-style: dashed;
    background-color: #fffaf2;
  }

  .algorithm li {
    margin-bottom: 0px;
  }
---
**Thank you for reading this set of essays and congratulations on making it all the way to the end.** Before we conclude, a few acknowledgments:

## Acknowledgments

This document represents a significant collective investment from many people at Google DeepMind, who we'd like to briefly acknowledge!

* James Bradbury, Reiner Pope, and Blake Hechtman originally derived many of the ideas in this manuscript, and were early to understanding the systems view of the Transformer.
* Sholto Douglas wrote the first version of this doc and is responsible for kicking off the project. He is more than anyone responsible for the overall narrative of this doc.
* Jacob Austin led the work of transforming this first version from rough notes into a more polished and comprehensive artifact. He did much of the work of editing, formatting, and releasing this document, and coordinated contributions from other authors.
* Most of the figures and animations were made by Anselm Levskaya and Charlie Chen.
* Charlie Chen wrote the inference section and drew many of the inference figures.
* Roy Frostig helped with publication, editing, and many other steps of the journey.

We'd also like to thank many others gave critical feedback throughout the process, in particular Zak Stone, Nikhil Sethi, Caitlin Stanton, Alek Dimitriev, Sridhar Lakshmanamurthy, Albert Magyar, Diwakar Gupta, Jeff Dean, Corry Wang, Matt Johnson, Peter Hawkins, and many others. Thanks to Ruiqi Gao for help with the HTML formatting.

**Thank you all!**

<p markdown=1 class="announce">Before you go, you might also enjoy reading the new [Section 12](../gpus) on NVIDIA GPUs!</p>

## Further Reading

There is a bunch of related writing, including the following:

* [**TPU Deep Dive**](https://henryhmko.github.io/posts/tpu/tpu.html): a wonderful in-depth look at the TPU architecture in the spirit of this book.
* [**Domain specific architectures for AI inference**](https://fleetwood.dev/posts/domain-specific-architectures): a hardware and model deep dive in the spirit of this book.
* [**A Domain-Specific Supercomputer for Training Deep Neural Networks**](https://dl.acm.org/doi/pdf/10.1145/3360307): one of the OG TPU papers, this has a lot of great details about the Google TPU program not covered here.
* [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html): a more GPU and PyTorch-focused tutorial on LLM rooflines and performance engineering.
* [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html): increasingly, TPU programming involves writing custom kernels in Pallas. This series discusses how to write kernels and many lower level TPU details that aren't mentioned here.
* [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM): while GPU and CUDA specific, this is an excellent blog post showing how to optimize a matmul kernel in CUDA. This might be a good deep dive into how TPUs and GPUs are different.
* [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html): this is a really nice guide to parallelism APIs in JAX and is a good way to learn how to actually implement some of the ideas we've discussed here.
* [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024): our former colleague Rafi gave a great course on TPU performance engineering and the slides are all on GitHub. This covers a bunch of things in more depth than we do here.
* [**\[2211.05102\] Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102): a detailed paper on the mathematics of Transformer inference. This is the inspiration for a lot of this document.
* [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook): something of a GPU analog to this book, this talks more at depth about how PyTorch implements parallelism techniques and memory-saving techniques during training.
* [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/): a blog with many of the same ideas as this book and some excellent illustrations.
* [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework): a fantastic Stanford course covering many details of LLM training and serving, with some useful exercises. Assignments 1 and 2 are particularly relevant.
* [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering): a highly practical guide to ML infrastructure, covering topics not addressed in this book like how to negotiate with cloud providers, cluster management, and empirical measurements of GPU throughput.

There remains a lot of room for comprehensive writing in this area, so
we hope this manuscript encourages more of it! We also believe that
this is a fruitful area to study and research. In many cases, it can
be done even without having many hardware accelerators on hand.

## Feedback

Please leave comments or questions so that we can improve this
further. You can reach our corresponding author, Jacob Austin, at
jacobaustin123 [at] gmail [dot] com, or suggest edits by posting issues,
pull requests, or discussions [on GitHub](https://github.com/jax-ml/scaling-book).
