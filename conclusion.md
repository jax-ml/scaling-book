---
layout: distill
title: "Conclusions and Further Reading"
# permalink: /main/
description: "Thank you for reading! Here we'll include a few more references for further study."
date: 2025-02-03
future: true
htmlwidgets: true
hidden: false

section_number: 11

previous_section_url: "../jax-stuff"
previous_section_name: "Part 10: JAX"

next_section_url: ""
next_section_name: "..."

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

* James Bradbury and Reiner Pope originally derived many of the ideas in this doc, and were among the first to deeply understand the system's picture of the Transformer.
* Sholto Douglas wrote the first version of this doc and is responsible for kicking off the project. He is more than anyone responsible for the overall narrative of this doc.
* Jacob Austin led the work of transforming this first version from rough notes into a more polished and comprehensive artifact. He did much of the work of editing, formatting, and releasing this document, and coordinated contributions from other authors.
* Most of the figures and animations were made by Anselm Levskaya and Charlie Chen.
* Charlie Chen wrote the inference section and drew many of the inference figures.

We'd also like to thank many others gave critical feedback throughout the process, in particular Vinay Ramasesh, Zak Stone, Nikhil Sethi, Caitlin Stanton, Alex Dimitriev, Sridhar Lakshmanamurthy, Albert Magyar, Diwakar Gupta, Jeff Dean, Corry Wang, and many others. Thanks to Ruiqi Gao for help with the HTML formatting.

**Thank you all!**

## Further Reading

There is a bunch of related writing, including the following:

* [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html): increasingly, TPU programming involves writing custom kernels in Pallas. This series discusses how to write kernels and many lower level TPU details that aren't mentioned here.
* [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM): while GPU and CUDA specific, this is an excellent blog post showing how to optimize a matmul kernel in CUDA. This might be a good deep dive into how TPUs and GPUs are different.
* [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)**:** this is a really nice guide to parallelism APIs in JAX and is a good way to learn how to actually implement some of the ideas we've discussed here.
* [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024)**:** our former colleague Rafi gave a great course on TPU performance engineering and the slides are all on GitHub. This covers a bunch of things in more depth than we do here.
* [**\[2211.05102\] Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102): a detailed paper on the mathematics of Transformer inference. This is the inspiration for a lot of this document.

There is a real lack of comprehensive writing in this area, so I hope people will take this as encouragment to write more! And to study this area. I continue to believe this is a fruitful area for academic research, since you can do so much useful work without a large number of GPUs or TPUs.

## Feedback

Please leave comments or questions so we can improve this further. You can reach me at jaaustin [at] google [dot] com, or suggest edits directly in GitHub.