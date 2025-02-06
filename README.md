# How To Scale Your Model

This book aims to demystify the art of scaling LLMs on TPUs. We try to explain how TPUs work, how LLMs actually run at scale, and how to pick parallelism schemes during training and inference that avoid communication bottlenecks. The book is available at https://jax-ml.github.io/scaling-book.

### Acknowledgments

This book was written by Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh and Albert Webson at Google DeepMind. Many of the ideas were first derived by James Bradbury and Reiner Pope.

The website uses a Distill-style Jekyll theme created by https://github.com/alshedivat/al-folio and the Distill team. Thank you!

### Running Locally

To build this repo locally, run

```
git clone https://github.com/jax-ml/scaling-book.git
cd scaling-book
bundle install
bundle exec jekyll serve
```

To run on Mac OS, you may need to run some of the following as well: `brew install imagemagick`, `pip install jupyter`, `brew install ruby`, or `git config safe.bareRepository all`, depending on what errors you hit. Once you have run jekyll serve successfully, the book will be available at `localhost:4000/scaling-book`.

To deploy to the GitHub Pages site (with repo write permision), run `sh bin/deploy`, which will take about 3 minutes to run.

### Contributing and Contact

If you see any issues or have questions, please leave a comment on the website itself (powered by Giscus) or in the GitHub discussion. Feel free to send a PR if you want to contribute. You can also email jaaustin [at] google [dot] com.

![dragon](assets/img/dragon.png)

*This book was originally called "How To Train Your Dragon", after the Dreamworks film, hence the dragon imagery.*
