I've written a lot of Bezier code. Here I am doing it again. Using De Casteljau where possible.

git submodule add $HOME/PycharmProjects/ttf_extractor
git submodule update --remote, Git will go into your submodules and fetch and update for you.
git submodule update --init --recursive # when starting work
git submodule update --remote --rebase  # when completing work


The 3-steps removal process would then be:
0. mv a/submodule a/submodule_tmp

1. git submodule deinit -f -- a/submodule
2. rm -rf .git/modules/a/submodule
3. git rm -f a/submodule
# Note: a/submodule (no trailing slash)

# or, if you want to leave it in your working tree and have done step 0
3.   git rm --cached a/submodule
3bis mv a/submodule_tmp a/submodule