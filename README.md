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

After this is installed, you should make sure the correct version of Ruby is found in PATH. You should have at least Ruby 3.4.5 installed. Then you ought to be able to run

```
git clone https://github.com/jax-ml/scaling-book.git
cd scaling-book
bundle install
bundle exec jekyll serve
```

Once you have run jekyll serve successfully, the book will be available at `http://127.0.0.1:4000/scaling-book`.

The Github Pages deployment is handled by a GitHub Action that runs automatically on new commits to the main branch.

### Generating a Single Document

To combine all chapters into a single markdown file with standardized formatting:

```bash
python bin/convert_to_single_md.py
```

This generates `scaling-book-combined.md` in the repository root. The script:
- Strips Jekyll frontmatter
- Converts `{% include figure.liquid %}` to standard markdown images
- Converts internal page links to anchor links
- Converts inline `$$` math to `$`
- Strips unsupported LaTeX commands (with warnings)

To convert the combined markdown to a Word document:

```bash
pandoc scaling-book-combined.md -o scaling-book.docx
```

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
