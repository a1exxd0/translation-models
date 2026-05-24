# Lightweight transformers for translation.

An experiment on how small we can make language models for translation between languages. Does it differ by language?

# Locally running

```sh
git clone --recurse-submodules https://github.com/a1exxd0/translation-models.git
make test
```

# Local development

Requires [`bear`](https://github.com/rizsotto/bear). Install on MacOS with:

```sh
brew install bear
```

```sh
git clone --recurse-submodules https://github.com/a1exxd0/translation-models.git
bear -- make
```

Run tests:
```sh
make test
```