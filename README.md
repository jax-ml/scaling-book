# How To Scale Your Model

This book aims to demystify the art of scaling LLMs on TPUs. We try to explain how TPUs work, how LLMs actually run at scale, and how to pick parallelism schemes during training and inference that avoid communication bottlenecks. The book is available at https://jax-ml.github.io/scaling-book.

### Acknowledgments

This book was written by Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh and Albert Webson at Google DeepMind. Many of the ideas were first derived by James Bradbury and Reiner Pope.

The website uses a Distill-style Jekyll theme created by https://github.com/alshedivat/al-folio and the Distill team. Thank you!

### Running Locally

To build this repo locally, you will need Ruby, ImageMagick, and Jupyter installed, which for MacOS can be installed with Homebrew using

```
brew install imagemagick ruby
pip install jupyter
```

After this is installed, you should make sure the correct version of Ruby is found in PATH. You should have at least ruby 3.4.5 installed. You may need to add

```
if [ -d "/opt/homebrew/opt/ruby/bin" ]; then
  export PATH=/opt/homebrew/opt/ruby/bin:$PATH
  export PATH=`gem environment gemdir`/bin:$PATH
fi
```

to your .bashrc to get the right version. After this, you should be able to clone and run the repository.

```
git clone https://github.com/jax-ml/scaling-book.git
cd scaling-book
bundle install
bundle exec jekyll serve
```

Once you have run jekyll serve successfully, the book will be available at `http://127.0.0.1:4000/scaling-book`.

To deploy to the GitHub Pages site (with repo write permision), run `sh bin/deploy`, which will take about 3 minutes to run.

### Contributing and Contact

If you see any issues or have questions, please leave a comment on the website itself (powered by Giscus) or in the GitHub discussion. Feel free to send a PR if you want to contribute. You can also email jaaustin [at] google [dot] com.

To contribute on GitHub you will need to sign a Google "Contributor License Agreement" (CLA). You can do that here: https://cla.developers.google.com/clas.

### Citation

For attribution in academic contexts, please cite this work as

```Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.```

BibTeX citation

```
@article{scaling-book,
  title = {How to Scale Your Model},
  author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
  publisher = {Google DeepMind},
  howpublished = {Online},
  note = {Retrieved from https://jax-ml.github.io/scaling-book/},
  year = {2025}
}
```

![dragon](assets/img/dragon.png)

*This book was originally called "How To Scale Your Dragon", after the Dreamworks film, hence the dragon imagery.*
