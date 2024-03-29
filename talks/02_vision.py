# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] slideshow={"slide_type": "skip"} tags=[] jp-MarkdownHeadingCollapsed=true
# # Setup

# + tags=[] slideshow={"slide_type": "skip"} jupyter={"source_hidden": true}
# %matplotlib inline

# + tags=[] slideshow={"slide_type": "skip"} jupyter={"source_hidden": true}
import math
from matplotlib import pyplot as plt
from fastai.torch_core import show_image

def show_images(
    images, 
    ncols=None, 
    nrows=None, 
    cmap="coolwarm", 
    vmin=None, 
    vmax=None, 
    vrange=0.5, 
    figsize=None, 
    **kwargs
):
    ncols = ncols or len(images)
    nrows = math.ceil(len(images) / ncols)
    
    abs_max = max(image.abs().max() for image in images)
    vmin = vmin if vmin is not None else -abs_max * vrange
    vmax = vmax if vmax is not None else +abs_max * vrange

    figsize = figsize or (ncols, nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    
    for image, ax in zip(images, axs.flat):
        show_image(image, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)


# + tags=[] slideshow={"slide_type": "skip"} jupyter={"source_hidden": true}
def param_count(module):
    return sum(param.numel() for param in module.parameters())


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# # Vision models
#
# Intro to image recognition with neural networks

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Plan
#
# 1. Machine Learning
# 2. MNIST dataset
# 3. Preparing data for training
# 4. Evaluating a model
# 5. Testing various models

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Previously we've focused on how neural networks work at the low level and glossed over the overall machine learning process. In this talk we'll go over the whole process of building a model:
#
# - Downloading and exploring data
# - Preparing training and validation sets
# - Picking metrics to judge our models
# - Experiments with various models
# - Tweaks to make our models train faster
#
# We will try to create a model for handwritten digit recognition using the very popular [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).
#
# > **Note:**
# > For some reason the original link to MNIST dataset https://yann.lecun.com/exdb/mnist/ is asking for authentication, so I've linked to the Wikipedia article instead
#
# Links:
# - https://en.wikipedia.org/wiki/MNIST_database
# - https://yann.lecun.com/exdb/mnist/

# + [markdown] tags=[] slideshow={"slide_type": "slide"}
# ## Machine Learning
#
# Approach to solving tasks by learning an algorithm from examples.
#
# > Suppose we arrange for some automatic means of testing the effectiveness of any current weight assignment in terms of actual performance and provide a mechanism for altering the weight assignment so as to maximize the performance. We need not go into the details of such a procedure to see that it could be made entirely automatic and to see that a machine so programmed would "learn" from its experience.
# >
# > [Arthur L. Samuel, Artificial Intelligence: A Frontier of Automation, 1962](https://journals.sagepub.com/doi/abs/10.1177/000271626234000103)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Usually when we want the computer to do something we need to explain in detail what we want it to do. Computers are powerful, but really dumb, they will only do what we ask them to do... and that's fine for a lot of problems. However, there are problems we can't easily explain to a computer, because we don't really know what needs to be done to recognize a dog in an image.
#
# Arthur Samuel, an IBM researcher, proposed a different way of explaining tasks to computers, by providing examples of data and letting the computer figure the algorithm out.
#
# Links:
# - https://nbviewer.org/github/fastai/fastbook/blob/master/01_intro.ipynb
# - https://en.wikipedia.org/wiki/Machine_learning
# - https://en.wikipedia.org/wiki/Arthur_Samuel_(computer_scientist)

# + [markdown] tags=[] slideshow={"slide_type": "slide"}
# Machine learning loop
#
# ![Machine learning loop](./images/machine_learning_loop.png)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# First off we need some examples of data that will be passed to our model, in the case of our husky/wolf classifier from the previous talk that would be the set of pictures of huskies and wolves that we downloaded.
#
# Depending on whether we know what these examples represent, what the expected outputs are, we split machine learning into:
#
# - supervised learning
# - unsupervised learning
#
# We will focus on supervised learning for now and assume we have labeled examples. Once we have inputs and labels we need a way to measure the performance of any model we are going to try. Finally we can create a parametric model and optimize the parameters to maximize the performance on our examples.

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ...which for neural networks becomes
#
# ![Neural network training loop](./images/neural_network_training_loop.png)

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## MNIST dataset
#
# A dataset of grayscale images of handwritten digits. `fastai` has utilities for downloading various well-known datasets, including MNIST.

# + tags=[]
from pathlib import Path
from fastai.data.external import untar_data, URLs

data_path = (Path("..") / "data").resolve()
mnist_path = untar_data(url=URLs.MNIST, data=data_path)
mnist_path

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Links:
# - https://docs.fast.ai/data.external.html

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# Let's check what was downloaded

# + tags=[]
# !tree -L 2 -sh --du {mnist_path}

# + tags=[]
list((mnist_path / "training" / "5").iterdir())[:5]

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We have 2 directories `training` and `testing`, each of which has 10 directories, 1 for each digit. Each of these directories in turn contain 28x28px grayscale images of handwritten digits. We can use the Python Imaging Library to view these images.
#
# Links:
# - https://pillow.readthedocs.io/en/stable/

# + tags=[]
from PIL import Image

Image.open(mnist_path / "training" / "5" / "19858.png")

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Overfitting
#
# - Model that works great, but only on its training set
# - One of the biggest problem in ML
# - A model should learn general concepts
# - Complex models have more capacity to overfit

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# So what's up with these `training` and `testing` directories? We want our model to learn from our examples so that it can be applied to new data in future. For example we want to show it photos of wolves and huskies, train it and then use it to classify new photos. However, during training our model will look for the easiest way to achieve its goal. If we gave it enough parameters to play with it could memorize our training set and be 100% accurate on it, while being completely useless in the real world.
#
# This is called overfitting and it doesn't have to be as extreme. In general it means our model is doing better on the training samples than on data it has never seen. This is probably the biggest common problem in ML and we have to take care to avoid it.
#
# First of all we need to be able to detect overfitting and so we split our labeled data into at least 2 sets: training and validation. Training set is only used for training and validation set is used to evaluate our model's performance on data it has never seen. This way we can compare metrics of our model on these sets and if they are considerably worse on the validation set our model might be overfitting.
#
# The easiest way to split the data is random sampling, e.g. select random 20% of your data and mark that as a validation set and leave the rest for training. However, there's a good chance this won't work depending on the task:
#
# - if we are predicting future stock prices, we should ensure that the validation set contains dates after the training set
# - if we are classifying facial expressions, we might want to ensure the validation set contains pictures of different people than the training set
#
# MNIST dataset used to be used for comparing/benchmarking classifiers and so it provides ready to use training and validation (`testing`) sets, so we don't have to split the data ourselves. That said, if we had to, we should make sure that the validation set contains digits written by different people than the training set.

# + [markdown] tags=[] slideshow={"slide_type": "slide"}
# ## DataLoader

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Enough theory, let's load our training and validation sets. `pytorch` expects that datasets are defined using 2 classes:
#
# - `Dataset` - provides an iterator that returns pairs of raw inputs and labels
# - `DataLoader` - loads inputs and labels from a `DataSet` transforms them into `pytorch` `Tensor`s and returns an iterator of evenly sized batches, so we can train a model one batch at a time
#
# `fastai` builds on top of that and provides the `DataBlock` class, which given an explanation of what our inputs and labels are, creates training and validation `Dataset`s and `DataLoader`s with sensible defaults and common transforms. To specify the shape of inputs and labels `fastai` provides various `Block` classes, we'll use:
#
# - `ImageBlock` - for our input images
# - `CategoryBlock` - for digit labels
#
# Links:
# - https://docs.fast.ai/tutorial.datablock.html

# + tags=[]
from fastai.data.all import DataBlock, CategoryBlock, FuncSplitter
from fastai.vision.all import ImageBlock, PILImageBW

mnist_block = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
    n_inp=1,
    get_items=lambda data_path: list(data_path.glob("*/*/*.png")),
    get_y=lambda image_path: image_path.parent.name,
    splitter=FuncSplitter(lambda image_path: image_path.parent.parent.name == "testing"),
)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Now that we've explained to `fastai` how our dataset looks like...
#
# - `blocks` defines what kinds of data we have
# - `n_inp` tells how many inputs we have, the rest will be treated as labels
# - `get_items` describes how to get a list of all our images, provided a base path to our dataset
# - `get_y` defines how to infer labels from image paths
# - `spliter` defines how to split our images into training and validation sets
#
# ...we can ask it to create training and validation `DataLoader`s and explore how our samples look like

# + tags=[] slideshow={"slide_type": "slide"}
mnist_loaders = mnist_block.dataloaders(mnist_path, bs=64)
mnist_loaders.train.show_batch(max_n=16, ncols=8, figsize=(8, 2.5))

# + tags=[]
x, y = mnist_loaders.train.one_batch()
x.shape, y.shape

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# When creating a `DataLoader` we can specify a batch size `bs`, it defaults to `64`. We process data in batches, so that we don't have to hold the entire dataset in memory. For MNIST it's not that important, we could probably fit the entire set in memory without problems, but in general when working with bigger sets with more complex data we would quickly run out.
#
# `DataBlock.dataloaders()` creates both training and validation loaders, available under `.train` and `.valid` respectively. `fastai` `DataLoader`s also provide the `show_batch()` method to quickly inspect the data based on the `blocks` we passed to the `DataBlock`.
#
# Finally we can see that a batch contains 2 tensors each with 64 rows, which is our batch size. 
#
# The first tensor contains our image data, each image is 28x28 pixels and since they are grayscale we also have `1` for the number of color channels. If we worked with color, we would have `3` (RGB) channels.
#
# The second tensor contains our labels, not much to see here for now.

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Baseline

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Before we start experimenting with various networks let's prepare a quick baseline for performance we'd like to beat. We can make a simple classifier that will compare an image to the average image of each digit and return the digit that's closest, e.g. using mean squared error of pixel values.
#
# So let's start with calculating images of average digits.

# + tags=[] slideshow={"slide_type": "slide"}
import torch

def calc_mean_digits():
    device = mnist_loaders.device
    sums = torch.zeros(10, 1, 28, 28).to(device)
    counts = torch.zeros(10).to(device)
    
    for x, y in mnist_loaders.train:
        for category in range(10):
            sums[category] += x[y == category].sum(dim=0)
            counts[category] += (y == category).sum()
    
    return sums / counts.view(10, 1, 1, 1)


# + tags=[]
mean_digits = calc_mean_digits()
mnist_loaders.show_batch((mean_digits, range(10)), max_n=10, ncols=5, figsize=(5, 2.5))

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Now we can build a baseline model that will compare inputs to the means and pick the digit which was closest.

# + tags=[]
from fastai.torch_core import TensorCategory

def baseline(x):
    distances = ((x - mean_digits.view(1, 10, 28, 28)) ** 2).mean(dim=(2, 3))
    _, y = distances.min(dim=1)
    return TensorCategory(y)


# + tags=[]
x, y = mnist_loaders.train.one_batch()
mnist_loaders.show_results((x, y), baseline(x), max_n=16, ncols=8, figsize=(8, 2.5))


# -

# It seems to be working pretty well, let's calculate the accuracy over both the training and validation sets.

# + tags=[]
def model_accuracy(model, loader):
    correct = sum((y == model(x)).sum() for x, y in loader).item()
    total = len(loader.dataset)
    return correct / total

train_accuracy = model_accuracy(baseline, mnist_loaders.train)
valid_accuracy = model_accuracy(baseline, mnist_loaders.valid)
train_accuracy, valid_accuracy

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Now we know that whatever model we choose, it has to have at least 82% accuracy, otherwise we might as well use our baseline model. 
# -

# ## Regression

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Now that we have a baseline we can start experimenting with various models. We have a 28x28 image as input and 1 number (the digit) as the output, so it's tempting to go with a model like:

# + tags=[]
from torch import nn

linear_regression = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 1),
)
# -

# `Flatten` reshapes the input so instead of being `28x28` it's just a vector of length `28 * 28`.
#
# `Linear` creates a linear layer with random weights and biases `wx + b`, just like we created from scratch in the previous talk. Its arguments are numbers of inputs and outputs.
#
# Finally `Sequential` just combines layers, so that first we `Flatten` our images and then pass it through the `Linear` layer.
#
# The output of our network will be a single floating-point number, which we will round to the closest integer later. We could round it as part of the network, but that would result in our loss function being mostly flat with zero gradients. That's because changing parameters might result in the exact same output and as such, same loss as before the change, so gradient descent would not work.
#
# Predicting a continuous value is called regression and so I've named our first network the `linear_regression` model. Predicting a category is called classification and we'll get to that later.
#
# Links:
# - https://en.wikipedia.org/wiki/Regression_analysis
# - https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
# - https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
# - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

# + tags=[]
from fastai.learner import Learner
from fastai.losses import MSELossFlat

def mnist_regression_learner(model):
    def accuracy(pred, target):
        return (pred.round() == target).float().mean()
    
    return Learner(
        dls=mnist_loaders,
        model=model,
        loss_func=MSELossFlat(),
        metrics=[accuracy]
    )


# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# To train a network with `fastai` we use the `Learner` class, it has a bunch of different parameters, but the most important are:
#
# - `dls` - our `DataLoaders` instance, which will provide training and validation data
# - `model` - the model we want to train
# - `loss_func` - loss function we want to minimize, for regression mean squared error is a safe choice
# - `metrics` - additional metrics we want to measure on the validation set while training
#
# Once we've instantiated a `Learner`, we can start training using `fit` or `fit_one_cycle` methods. I won't go into the differences for now, but `fit_one_cycle` is faster 😉. We have to give it the number of epochs (how many times we want to repeat training over the full training set) and the learning rate.
#
# The `Learner` class provides a `lr_find()` method, which tries a bunch of learning rates and suggests one for us, so let's try that first.
#
# Links:
# - https://docs.fast.ai/learner.html#learner
# - https://docs.fast.ai/callback.schedule.html#learner.fit_one_cycle
# - https://arxiv.org/abs/1708.07120
# - https://docs.fast.ai/callback.schedule.html#learner.lr_find
# - https://arxiv.org/abs/1506.01186

# + tags=[]
mnist_regression_learner(linear_regression).lr_find()

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# In general we want to pick the point where the loss is smooth and pointing down, which should hopefully give us a smooth loss decrease during training.

# + tags=[]
mnist_regression_learner(linear_regression).fit_one_cycle(5, 0.005)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Well, that's not very encouraging, the model seems pretty terrible with less than 25% accuracy, so only slightly better than random choice. Let's review some results.

# + tags=[]
x, y = mnist_loaders.valid.new(shuffle=True, bs=16).one_batch()
y_pred = linear_regression(x).round().int().view(16)

mnist_loaders.show_results((x, y), y_pred.clip(0, 9), max_n=16, ncols=8, figsize=(8, 2.5))

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# So one of the problems we see here is that our model doesn't know that predictions have to be between `0` and `9`, so every now and then it will return an invalid prediction of `-1` or `10`... the latter has a side effect of `show_results` crashing as well...
#
# We could move the `.clip(0, 9)` to the model, but again, that would introduce regions with zero gradients, so we need a smooth function that would do something similar. Luckily there's a `sigmoid` function, which maps any value to `(0, 1)` range.
#
# $$
# \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
# $$
#
# Links:
# - https://en.wikipedia.org/wiki/Sigmoid_function
# - https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html

# + tags=[]
from matplotlib import pyplot as plt

x = torch.linspace(-6, 6, 100)
plt.plot(x, torch.sigmoid(x))
plt.grid(True)
plt.show()

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We need `[0, 9]` instead of `(0, 1)`, but that's fairly easy, we can just multiply the output, that's what the `SigmoidRange` layer from `fastai` does.
#
# Links:
# - https://docs.fast.ai/layers.html#sigmoidrange

# + tags=[]
from fastai.layers import SigmoidRange

linear_regression_sigmoid = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 1),
    SigmoidRange(-0.5, 9.5),
)

# + [markdown] tags=[] slideshow={"slide_type": "notes"}
# Let's train it and see if that helps

# + tags=[]
mnist_regression_learner(linear_regression_sigmoid).fit_one_cycle(5, 0.005)

# + tags=[]
x, y = mnist_loaders.valid.new(shuffle=True, bs=16).one_batch()
y_pred = linear_regression_sigmoid(x).round().int().view(16)

mnist_loaders.show_results((x, y), y_pred, max_n=16, ncols=8, figsize=(8, 2.5))

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Well it's still just as terrible as it was, but at least we no longer have to `.clip()`... maybe we just need a more complex model, let's add a hidden layer.

# + tags=[]
linear_deep_regression = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    SigmoidRange(-0.5, 9.5),
)

# + tags=[]
mnist_regression_learner(linear_deep_regression).fit_one_cycle(5, 0.005)

# + tags=[]
x, y = mnist_loaders.valid.new(shuffle=True, bs=16).one_batch()
y_pred = linear_deep_regression(x).round().int().view(16)

mnist_loaders.show_results((x, y), y_pred, max_n=16, ncols=8, figsize=(8, 2.5))

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Now we're getting somewhere! That said, we are still very much under our baseline.
#
# We could probably make the model even more complex, with more neurons and/or layers, but we might be able to do better by changing our approach instead.

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Classification

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# The problem with our approach so far is that we have been using a regression model for classification. We don't need to estimate the value of the digit, we just want to know which digit it is. Predicting the value makes it harder for the model to figure out similar looking numbers as their values might not be similar at all. For example `1` and `7` can look very similar, but if our model is 50/50 about which it is it will land at `4`, which looks completely different.
#
# What we could do instead is try to predict 10 probabilities, one for each digit, and pick the one with the highest probability. Let's go back to our simplest model and just change the number of outputs to 10.

# + tags=[]
linear1 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 10),
)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We need to also change our training procedure.
#
# First off, we need to change our target, so its shape matches our output. We want the model to learn to predict probabilities and during training we know exactly which digit it is, so we can turn each target digit into a vector with a single 1 (100% probability) and 9 0s for other digits. This is called one-hot encoding.
#
# Links:
# - https://en.wikipedia.org/wiki/One-hot
# - https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html

# + tags=[]
from torch.nn.functional import one_hot

one_hot(torch.tensor(3), num_classes=10)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Let's try training our new model.

# + tags=[]
from torch.nn.functional import mse_loss, one_hot
from fastai.metrics import accuracy

def mnist_mse_learner(model):
    def loss(pred, target):
        encoded_target = one_hot(target, num_classes=10).float()
        return mse_loss(pred, encoded_target)
    
    return Learner(
        dls=mnist_loaders,
        model=model,
        loss_func=loss,
        metrics=[accuracy],
    )


# + tags=[]
mnist_mse_learner(linear1).fit_one_cycle(5, 0.005)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# With this tiny change we have now beaten our baseline model which had accuracy of around 82% 🏆

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Mean-squared error works great for regression, but for classification, where we predict probabilities there's cross-entropy loss, and it looks like:
#
# $$
# \text{CrossEntropyLoss}(x,t) = \text{NLLLoss}(\text{Softmax}(x), t)
# $$
#
# where:
#
# $$
# \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
# $$
#
# $$
# \text{NLLLoss}(x, t) = -log(x_t)
# $$
#
# `Softmax` makes sure that our predictions are in the `[0,1]` range and that they sum up to `1` like real probabilities.
#
# `NLLLoss` is negative log likelihood loss and all it does is just take the `-log()` of our prediction for the target class. That also means that it ignores our predictions for the other classes, as opposed to mean-squared error loss.
#
# Links:
# - https://en.wikipedia.org/wiki/Cross_entropy
# - https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# - https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
# - https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html

# + tags=[]
from matplotlib import pyplot as plt

x = torch.linspace(0, 1, 100)
plt.plot(x, -x.log())
plt.grid(True)
plt.show()

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Enough math, let's see if changing the loss function helps.

# + tags=[]
from fastai.losses import CrossEntropyLossFlat

def mnist_learner(model, dls=mnist_loaders, cbs=None):
    return Learner(
        dls=dls,
        model=model,
        loss_func=CrossEntropyLossFlat(),
        metrics=[accuracy],
        cbs=cbs,
    )


# + tags=[]
linear1 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 10),
)

mnist_learner(linear1).fit_one_cycle(5, 0.05)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Another step in the right direction 🏆

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Because we have a single weight for each pixel for each digit, we can view the learned weights to see what our network looks for in each digit.

# + tags=[]
show_images(linear1[1].weight.view(10, 1, 28, 28), ncols=5, vrange=0.2)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Warm colors represent positive weights and cold negative ones. This means that the network expects lines to be where the red spots are and doesn't want lines in blue spots.
#
# You can clearly see blue outlines of numbers and some red spots, which are characteristic to each digit. For example it seems that when there's a line in the top right that does not go down, that's a good predictor of a `5`.

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Let's see where we can get with a deeper network.

# + tags=[]
linear2 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

mnist_learner(linear2).fit_one_cycle(5, 0.01)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Just 2% error rate, another huge jump in accuracy. Now that we are getting very close to 100%, let's review cases where the model has problems. `fastai` offers a couple helpers for intepreting the results. We can start with showing the samples that we are most incorrect about, that have the highest losses.
#
# Links:
# - https://docs.fast.ai/interpret.html

# + tags=[]
from fastai.interpret import Interpretation

Interpretation.from_learner(mnist_learner(linear2)).plot_top_losses(k=24, ncols=6, figsize=(12, 5))

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# As you can see we are entering the MNIST hard mode. Some of these a human would have problems with as well.
#
# We can also plot a confusion matrix, which shows how many digits we are mislabeling and how. On the diagonal we see correct labels, everywhere else there are errors.

# + tags=[]
from fastai.interpret import ClassificationInterpretation

ClassificationInterpretation.from_learner(mnist_learner(linear2)).plot_confusion_matrix()

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Again, we can try brute-forcing the problem and just adding more complexity to the network.

# + tags=[]
linear3 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

mnist_learner(linear3).fit_one_cycle(5, 0.01)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# However, it seems that we need to change our approach again if we want to solve these last 2%.
# -

# ## Convolutions

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# There are a couple problems with our previous models:
#
# 1. The model will only work with 28x28 images
# 2. The number of parameters is proportional to the number of pixels
#
# `fastai` `Learner` has a `summary()` method to review inputs, outputs and numbers of parameters of each layer

# + tags=[]
mnist_learner(linear2).summary()

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Even though we have a relatively simple model and just 28x28px images, we've already gathered over 100k parameters. This would clearly not scale to normal images or photos, where we might want to use 512x512px images with 3 color channels.

# + tags=[]
format(param_count(
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 512 * 512, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
), ",")

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# 100M parameters even without increasing the number of neurons in the hidden layer. We need a different way to process images of arbitrary size. 

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Luckily people have been processing images for decades and so we can draw inspiration from that. Common image filters such as blur or sharpen use convolution under the hood. Filters like these calculate the color of the output pixel by multiplying the corresponding input pixel and its surroundings by a matrix and summing the values up. This matrix is called a convolution kernel. 
#
# Let's visualize how that works with a simple blur kernel, which just averages out surrounding pixels.
#
# Links:
# - https://en.wikipedia.org/wiki/Convolution
# - https://en.wikipedia.org/wiki/Kernel_(image_processing)
# - https://arxiv.org/abs/1603.07285

# + tags=[] slideshow={"slide_type": "skip"} jupyter={"source_hidden": true}
import ipywidgets as widgets
from torch.nn.functional import conv2d
from torchvision.transforms import ToTensor
from IPython.display import display
from matplotlib import pyplot as plt

def interactive_conv2d(image, kernel):
    image_height, image_width = image.shape[1:]
    
    def render(padding=0, stride=1, step=1):
        output, = conv2d(
            image.view(1, *image.shape), 
            kernel.view(1, 1, *kernel.shape).float(),
            padding=padding,
            stride=stride,
        )
        output_height, output_width = output.shape[1:]
        step_slider.max = output.numel()
        
        step = min(step, step_slider.max)
        index = step - 1
        
        fig, (image_ax, output_ax) = plt.subplots(1, 2)
        
        image_ax.set_xlim(-padding, image_width + padding)
        image_ax.set_ylim(image_height + padding, -padding)
        image_ax.set_facecolor("black")
        
        image_ax.imshow(
            image.permute(1, 2, 0), 
            cmap="gray", 
            extent=(0, image_width, image_height, 0),
            vmin=0,
            vmax=1,
        )
        image_ax.add_patch(plt.Rectangle(
            xy=(
                (index % output_width) * stride - padding,
                (index // output_width) * stride - padding,
            ),
            width=kernel.shape[1],
            height=kernel.shape[0],
            fill=False,
            color="red",
        ))
        image_ax.set_title("input")
        
        output_ax.imshow(
            output.permute(1, 2, 0),
            cmap="gray",
            extent=(0, output_width, output_height, 0),
            vmin=0,
            vmax=1,
        )
        output_ax.add_patch(plt.Rectangle(
            xy=(index % output_width, index // output_width),
            width=1,
            height=1,
            fill=False,
            color="red",
        ))
        output_ax.set_title("output")

    padding_slider = widgets.IntSlider(min=0, max=2, value=0, description="padding")
    stride_slider = widgets.IntSlider(min=1, max=3, value=1, description="stride")
    step_slider = widgets.IntSlider(min=1, max=1, value=1, description="step")
    
    display(widgets.HBox([
        widgets.VBox([padding_slider, stride_slider]),
        step_slider,
    ]))
    
    interactive_output = widgets.interactive_output(render, dict(
        padding=padding_slider,
        stride=stride_slider,
        step=step_slider,
    ))
    interactive_output.layout.height = '350px'
    
    display(interactive_output)


# + tags=[]
kernel = torch.tensor([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
]) / 9

# + tags=[]
interactive_conv2d(
    image=ToTensor()(PILImageBW.create("./images/codequest_tiny.png")),
    kernel=kernel,
)

# + tags=[] slideshow={"slide_type": "skip"}
from torch.nn.functional import conv2d

def simple_conv2d(image, kernel):
    return torch.cat([
        conv2d(
            channel.view(1, 1, *channel.shape), 
            kernel.view(1, 1, *kernel.shape).float(),
        )[0]
        for channel in image
    ]).clip(0, 1)


# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Now that we know how convolution works, let's experiment with a few simple kernels on a picture of a [giant rubber duck](https://thebigduck.us/)... beacause why not 😉
#
# Links:
# - https://thebigduck.us/
# - https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html?highlight=totensor#torchvision.transforms.ToTensor
# - https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html

# + tags=[]
from fastai.vision.all import PILImage
from torchvision.transforms import ToTensor

big_duck = ToTensor()(PILImage.create("./images/big-duck.jpg").to_thumb(400))
show_image(big_duck, figsize=(8, 4))
big_duck.shape


# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# First off, let's generalize our blur to any kernel size and see how that works.

# + tags=[]
def blur_kernel(size=3):
    return torch.ones(size, size) / size / size

show_image(simple_conv2d(big_duck, blur_kernel(5)), figsize=(8, 4))


# -

# Sharpening is a bit more complicated, we need to boost the value for the center pixel and then subtract the blur. A kernel that just returns the original pixel and ignores the surroundings is called an identity kernel and we'll use it to boost the center pixel.

# + tags=[]
def identity_kernel(size=3):
    identity = torch.zeros(size, size)
    identity[size // 2, size // 2] = 1
    return identity

def sharpen_kernel(size=3):
    return 2 * identity_kernel(size) - blur_kernel(size)

show_image(simple_conv2d(big_duck, sharpen_kernel(5)), figsize=(8, 4))

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Now this is fun and all, but how does that fit into neural networks? Well, we can also do feature detection with convolutions, for example these kernels detect horizontal and vertical edges respectively.

# + tags=[]
horizontal_detector = torch.tensor([
    [-1, -1, -1],
    [+2, +2, +2],
    [-1, -1, -1],
])
vertical_detector = torch.tensor([
    [-1, +2, -1],
    [-1, +2, -1],
    [-1, +2, -1],
])

show_images([
    simple_conv2d(big_duck, horizontal_detector),
    simple_conv2d(big_duck, vertical_detector),
], figsize=(10, 5))

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# There's many many more such kernels that people have already figured out, but what we can do with neural networks is let gradient descent create kernels for us. Instead of using a linear/dense layer we can use a convolutional layer, which uses kernel weights as parameters.
#
# Links:
# - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# - https://en.wikipedia.org/wiki/Convolutional_neural_network

# + tags=[]
conv_layer = nn.Conv2d(
    in_channels=1, 
    out_channels=10, 
    kernel_size=3,
    padding=1,
).to(mnist_loaders.device)

show_images(conv_layer.weight, ncols=5, cmap="gray", vrange=1)
# -

# As you can see a convolutional layer starts with random kernels. We can apply these kernels to an example image and see what that looks like.

# + tags=[]
x, _ = mnist_loaders.train.one_batch()
show_image(x[0], cmap="gray")
show_images(conv_layer(x[:1]).view(10, 1, 28, 28), ncols=5, cmap="gray", vrange=1)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Now the problem is how do we get from here to our 10 probabilities? With linear/dense layers we could just specify numbers of inputs and outputs, with convolution it's not that simple. We can control the number of outputs with 
#
# - the number of kernels - which defines how many output images the layer will produce, called output channels
# - convolution stride - we can decrease the resolution by skipping pixels in the input image
# - pooling - taking the average and/or maximum values to decrease the resolution
#
# We can create a purely convolutional network by using `stride=2` to decrease the resolution by half with each layer until we get to `1x1` outputs. We also have to make sure the final layer has `10` outputs, one for each digit.

# + tags=[]
conv1 = nn.Sequential(
    nn.Conv2d(1, 8, 3, stride=2),
    nn.ReLU(),
    nn.Conv2d(8, 16, 3, stride=2),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, stride=2),
    nn.ReLU(),
    nn.Conv2d(32, 10, 2),
    nn.Flatten(),
)

display(mnist_learner(conv1).summary())

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We start with `28x28` images with a single channel. 
#
# We are not using padding, so the output of the first layer would be `26x26`, but we are using `stride=2`, so we're skipping every other column and row, and we end up with `13x13`. We continue this process until we get to `1x1`. This single pixel will aggregate information about the entire image, because every time we halve the resolution we also double the area of the input image used to create the output pixel. We could also make the network completely image size independent by replacing the final convolution with a pooling layer, that would just average out whatever pixels are left after our convolutions into a single pixel.
#
# To compensate for resolution loss after each layer, which would leave just a quarter of the information, we also double the number of output channels, that way the network has to distill the information, but can do that more gradually.
#
# This all looks much more complicated than what we saw with our linear models, however, it uses much less parameters.

# + tags=[]
param_count(linear2), param_count(conv1)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Great, but does it work?

# + tags=[]
mnist_learner(conv1).fit_one_cycle(5, 0.01)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Well, yes, it easily beats our best linear model while using less than 10% of parameters. 
#
# We still have 1.4% error rate to figure out though. Before we move on, let's just have a look at digits we are still having problems with.

# + tags=[]
Interpretation.from_learner(mnist_learner(conv1)).plot_top_losses(k=24, ncols=6, figsize=(12, 5))
# -

# ## Data Augmentation

# If you look at the losses from our last training you will notice that the training loss is much lower than our validation loss. That is a sign we might be overfitting. Fundamentally overfitting happens when a model is complex enough to be able to learn things that are specific to the training set, which might not be true in real use, which we measure with the validation set. The best option to resolve that is to obtain more training data.
#
# Gathering new data might be time consuming and/or costly, you don't always have students at hand to extend your handwritten digits set.
#
# For images, and some other domains, we can however create infinite amount of data from what we already have by randomly transforming samples in our training set. We can for example rotate our digits at random to ensure our model is capable of classifying a digit regardless of its orientation. That is as long as your transforms wouldn't change the label, e.g. rotating a `6` by 180 degrees would turn it into a `9`.
#
# Applying such transforms during training to automatically create more data is called data augmentation.
#
# `fastai` provides some common augmentations we can take advantage of, we just need to create new data loaders with the added augmentation.
#
# Links:
# - https://en.wikipedia.org/wiki/Data_augmentation
# - https://pytorch.org/vision/main/transforms.html
# - https://docs.fast.ai/vision.augment.html

# + tags=[]
from fastai.vision.augment import Rotate, Zoom, Warp, PadMode, setup_aug_tfms

def augment_mnist_loaders(transforms):
    return mnist_block.new(batch_tfms=setup_aug_tfms(transforms)).dataloaders(mnist_path)

mnist_augmented_loaders1 = augment_mnist_loaders([
    Warp(p=0.5, magnitude=0.1, pad_mode=PadMode.Zeros),
    Zoom(p=0.5, min_zoom=0.9, max_zoom=1.1, pad_mode=PadMode.Zeros),
    Rotate(p=0.5, max_deg=30, pad_mode=PadMode.Zeros),
])
# -

# This will add some random warping, zooming and rotation to our training batches, here's a sample of what that looks like.

# + tags=[]
mnist_augmented_loaders1.train.show_batch(max_n=16, ncols=8, figsize=(8, 2.5))
# -

# Let's try the same model on this augmented training set

# + tags=[]
conv1_augmented = nn.Sequential(
    nn.Conv2d(1, 8, 3, stride=2),
    nn.ReLU(),
    nn.Conv2d(8, 16, 3, stride=2),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, stride=2),
    nn.ReLU(),
    nn.Conv2d(32, 10, 2),
    nn.Flatten(),
)

mnist_learner(conv1_augmented, dls=mnist_augmented_loaders1).fit_one_cycle(5, 0.01)


# -

# We're not doing much better than the model trained without augmentations, but now we are no longer overfitting. That means we can increase the models complexity so it can do better on our new training data and, hopefully, on our validation set.

# +
def conv2_block(channels_in, channels_out):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 3, stride=2),
        nn.ReLU(),
        nn.Conv2d(channels_out, channels_out, 3, padding=1),
        nn.ReLU(),
    )

conv2 = nn.Sequential(
    conv2_block(1, 16),
    conv2_block(16, 32),
    conv2_block(32, 64),
    
    nn.Conv2d(64, 10, 2),
    nn.Flatten(),
)

mnist_learner(conv2, dls=mnist_augmented_loaders1).fit_one_cycle(5, 0.01)
# -

# We've doubled the number of convolution layers, after every convolutional layer, we've added another one without `stride`, so it doesn't change the output shape. We've also doubled the number of filters in each layer.
#
# Results seem to be promising, we are now over the 99% threshold.

Interpretation.from_learner(mnist_learner(conv2)).plot_top_losses(k=24, ncols=6, figsize=(12, 5))

# ## Batch normalization

# As I mentioned earlier `fastai` provides us with useful tools to inspect our model, one of these tools is the `ActivationStats` callback, which gathers statistics about activations of each trainable layer of our network during training. This let's us look into the training process.
#
# Links:
# - https://docs.fast.ai/callback.hook.html#activation-graphs

# + tags=[]
from fastai.callback.all import ActivationStats

debug_conv2 = nn.Sequential(
    conv2_block(1, 16),
    conv2_block(16, 32),
    conv2_block(32, 64),
    
    nn.Conv2d(64, 10, 2),
    nn.Flatten(),
)

debug_learner = mnist_learner(debug_conv2, cbs=[ActivationStats(every=4, with_hist=True)])
debug_learner.fit_one_cycle(1, 0.01)

# + tags=[]
for i in range(len(debug_learner.activation_stats.stats[0])):
    debug_learner.activation_stats.color_dim(i)


# -

# `color_dim()` stacks histograms of activations of each batch on a single diagram, x axis is the batch index (or time) and the y axis represents the histogram of a particular batch. Color is used to mark how many activations there are with a particular value, from blue (few) to yellow (many).
#
# What we can see here is that the deeper we go in our network the more erratic the training becomes. The weights start near zero, shoot up and then collapse again and this cycle continues until it finally smooths itself out. This happens because activations in deeper levels are dependent on the previous levels, so small changes at the start of the network are multiplied through it, either shooting the activations up or zeroing them. The later layers have to compensate for the changes and so we get these oscilations.

# One way to deal with this problem is introducing normalization layers, which will scale the activations to a standard range, e.g. `(-1, 1)` or `(0, 1)`. 
#
# This would however prevent our network from outputing high activations when necessary to notify deeper layers of something important, so what we end up doing is adding a layer which normalizes the output range first then scales the outputs back up using a linear transform. Such a layer is provided by `pytorch` as different `BatchNorm` layers, since we have 2-dimensional inputs we need to use `BatchNorm2d`.
#
# Links:
# - https://nbviewer.org/github/fastai/fastbook/blob/master/13_convolutions.ipynb
# - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

# + tags=[]
def conv3_block(channels_in, channels_out):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 3, stride=2),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(),
        nn.Conv2d(channels_out, channels_out, 3, padding=1),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(),
    )

conv3 = nn.Sequential(
    conv3_block(1, 16),
    conv3_block(16, 32),
    conv3_block(32, 64),
    
    nn.Conv2d(64, 10, 2),
    nn.Flatten(),
)

conv3_learner = mnist_learner(conv3, cbs=[ActivationStats(every=4, with_hist=True)])
conv3_learner.fit_one_cycle(1, 0.01)

# + tags=[]
for i in range(0, len(conv3_learner.activation_stats.stats[0]), 2):
    conv3_learner.activation_stats.color_dim(i)
# -

# Much smoother, let's see if that actually improves our results.

# + tags=[]
mnist_learner(conv3, dls=mnist_augmented_loaders1).fit_one_cycle(5, 0.01)
# -

# Another huge improvement in accuracy.

# ## Custom data augmentations

# + tags=[]
Interpretation.from_learner(mnist_learner(conv3)).plot_top_losses(k=24, ncols=6, figsize=(12, 5))
# -

# Looking at the digits our network has problems with I see 2 patterns:
#
# - digits with missing parts of the lines
# - narrow and bold lines
#
# We can add custom augmentations that would introduce more such digits into the training set.

# + tags=[]
from torchvision.transforms import Resize
from fastai.vision.augment import RandTransform
from fastai.vision.data import TensorImageBW

class NoiseMask(RandTransform):
    split_index = 0
    
    def encodes(self, x: TensorImageBW):
        bs, _, h, w = x.shape
        nh, nw = h // 4, w // 4
        resize = Resize((h, w))
        noise = (8 * resize(torch.rand(bs, 1, nh, nw)) - 2).clip(0, 1)
        return x * noise.to(x.device)


# + tags=[]
x, y = mnist_loaders.train.new(bs=8).one_batch()
mask = NoiseMask(p=1)(torch.ones_like(x), split_idx=0)
x_masked = x * mask

mnist_loaders.show_batch((x, y), ncols=8, figsize=(8, 2.5))
mnist_loaders.show_batch((1 - mask, y), ncols=8, figsize=(8, 2.5))
mnist_loaders.show_batch((x_masked, y), ncols=8, figsize=(8, 2.5))


# -

# `NoiseMask` will:
# - generate a `7x7` image with random noise
# - scale it up to `28x28` so it matches our image dimensions
# - increase the contrast of the noise
# - apply the mask to the original image to hide random bits of the digit

# + tags=[]
class Bolden(RandTransform):
    split_index = 0
    order = 30
    
    def __init__(self, p=0.5, min_amount=-0.8, max_amount=1.5):
        super().__init__(p=p)
        self.min_amount, self.max_amount = min_amount, max_amount
        self.kernel = torch.tensor([
            [0.5, 0.8, 0.5],
            [0.8, 1.0, 0.8],
            [0.5, 0.8, 0.5],
        ]).view(1, 1, 3, 3)
        self.kernel /= self.kernel.sum()
    
    def encodes(self, x: TensorImageBW):
        bs = x.shape[0]
        amount = self.min_amount + (self.max_amount - self.min_amount) * torch.rand(bs, 1, 1, 1).to(x.device)
        return (4 * conv2d(x, self.kernel.to(x.device), padding=1) - 1.5 + amount).clip(0, 1)


# + tags=[]
x_bold = Bolden(p=1)(x, split_idx=0)

mnist_loaders.show_batch((x, y), ncols=8, figsize=(8, 2.5))
mnist_loaders.show_batch((x_bold, y), ncols=8, figsize=(8, 2.5))
# -

# `Bolden` on the other hand blurs the original image and then adjusts the black level to randomly make the lines lighter or bolder.

# Both transforms are based on `fastai` `RandTransform`, which gives us 2 things:
# - randomly switching the transform on or off for a particular batch, with the probability `p`
# - using the transform only for training, training has `split_idx=0` and validation has `split_idx=1` 
#
# We can now create data loaders with all the previous augmentations and our custom ones.
#
# Links:
# - https://docs.fast.ai/vision.augment.html#randtransform-

# + tags=[]
mnist_augmented_loaders2 = augment_mnist_loaders([
    Bolden(p=0.25),
    Warp(p=0.5, magnitude=0.1, pad_mode=PadMode.Zeros),
    Zoom(p=0.5, min_zoom=0.9, max_zoom=1.1, pad_mode=PadMode.Zeros),
    Rotate(p=0.5, max_deg=30, pad_mode=PadMode.Zeros),
    NoiseMask(p=0.25),
])

# + tags=[]
mnist_augmented_loaders2.show_batch(max_n=16, ncols=8, figsize=(8, 2.5))
# -

# ...and see if it actually helps

# + tags=[]
conv4 = nn.Sequential(
    conv3_block(1, 16),
    conv3_block(16, 32),
    conv3_block(32, 64),
    
    nn.Conv2d(64, 10, 2),
    nn.Flatten(),
)

mnist_learner(conv4, dls=mnist_augmented_loaders2).fit_one_cycle(10, 0.01)

# + tags=[]
Interpretation.from_learner(mnist_learner(conv4)).plot_top_losses(k=24, ncols=6, figsize=(12, 5))
# -

# Sadly, there's not much difference in accuracy, however, it does seem like the `NoiseMask` augmentation worked and the model classifies images with missing details better.
#
# ~99.5% accuracy is the best I got so far and it has to be enough for this talk.

# + [markdown] tags=[]
# ## Confidence
# -


# The accuracy looks great, but in the real world we want to use a classifier to automate a process, e.g. to parse digits on a tax form. Depending on the domain, classifying something badly, even very rarely might have significant consequences. For example, let's say our income was suddenly read as 900,000 instead of 100,000, that's just a single digit misclassified, but it will increase the tax that should be collected 9x (assuming a linear tax). 
#
# There are multiple solutions to this, on a tax form there will probably be calculations we could do to find inconsistencies, e.g. if the declared tax is 9x smaller than the declared income suggests, the form could be sent for manual review.
#
# However, if we're just classifying a single digit, without any other context, how confident should we be about the prediction? Which images should we send for manual review?
#
# The model in theory gives us its confidence in the prediction, since it returns an estimated probability for each digit. We could set a threshold under which predictions will be sent for review. We can plot the histogram of the models estimations to see how confident the model is in general.

# + tags=[] slideshow={"slide_type": "skip"} jupyter={"source_hidden": true}
from fastai.vision.utils import show_images as fastai_show_images

def plot_probabilities(model, indices, ncols=8):
    x, y_pred, y = mnist_learner(model).get_preds(with_input=True)
    
    nrows = len(indices) // ncols
    axs = plt.subplots(nrows, ncols, figsize=(12, nrows * 2.5))[1].flat
    
    for idx, ax in zip(indices, axs): 
        label = y[idx]
        top_probs, top_labels = y_pred[idx].sort(dim=0, descending=True)
        top_titles = [f"{label.item()}: {prob.item():.3f}" for prob, label in zip(top_probs, top_labels)]
        title = "\n".join([
            f"{label.item()}", 
            *top_titles[:3],
        ])
        
        show_image(1 - x[idx], ax=ax, title=title, cmap="gray", vmin=0, vmax=1)

def plot_top_loss_probabilities(model, descending=True, k=24, **plot_kwargs):
    interpretation = Interpretation.from_learner(mnist_learner(conv4), dl=mnist_loaders.valid.new(shuffle=False))
    _, top_losses = interpretation.top_losses(k=k, largest=descending)
    plot_probabilities(model, top_losses)

def plot_top_unsure_probabilities(model, descending=False, k=24, **plot_kwargs):
    y_pred, _ = mnist_learner(model).get_preds()
    top_unsure = y_pred.max(dim=1).values.sort(descending=descending).indices[:k]
    plot_probabilities(model, top_unsure, **plot_kwargs)
    
def plot_probability_histogram(model, max_probability=1):
    y_pred, y = mnist_learner(model).get_preds()
    y_pred_top = y_pred.sort(dim=1, descending=True).values[:,:2]
    indices = y_pred_top[:,0] <= max_probability
    
    plt.hist(y_pred_top[indices].T, bins=torch.linspace(0, 1, 21), stacked=True)
    plt.xlim(0, 1)
    plt.xticks(torch.linspace(0, 1, 11))
    plt.show()
    
    print(f"{indices.sum():,} samples")


# + tags=[]
plot_probability_histogram(conv4)
# -

# Blue columns represent the highest probability for each validation image and the orange one is the 2nd highest probability. We can see 2 huge spikes, 1st choices are almost always near 100% probability and 2nd almost always at 0%. The model is extremely confident in its choices. To see the middle part we need to zoom in as the values are so small they can barely be seen on the above histogram.

# + tags=[]
plot_probability_histogram(conv4, max_probability=0.95)
# -

# Out of all the validation images we only have ~200 where the model is less than 95% certain. We could use 95% as the threshold for manual review, which would mean ~2% of images would have to be reviewed, but that way we could also extend our training data and improve our model. It wouldn't however help for some cases, which we have seen earlier, on the top loss charts, where the model is wrong, but very confident.
#
# We can review some examples where the model is the most uncertain about its predictions.

# + tags=[]
plot_top_unsure_probabilities(conv4, k=24)
# -

# ## Label smoothing

# The reason why our model is very confident is that we've told it to 😉. We were training the model expecting that it will always give us 100% probability for the correct label... and so the model is trying really hard to do that, even when it's wrong.
#
# Another problem with expecting a 100% probability is that mathematically that's hard to do for the model, we are using sigmoid or softmax as the final activation, which force the output values to the `(0, 1)` range and the result is that the model has to have very high activations to reach something near `1`.
#
# We could make the job easier for the model by targeting a slightly lower probability, e.g. 90%. This is called Label Smoothing and it will make the model less confident overall, but also should help with both training and spotting uncertainty.
#
# `fastai` offers label smoothing as a custom loss function, `LabelSmoothingCrossEntropy` and we need to just specify by how much we want to lower the target.
#
# Links
# - https://nbviewer.org/github/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb
# - https://arxiv.org/pdf/1906.02629.pdf
# - https://docs.fast.ai/losses.html#labelsmoothingcrossentropy

# + tags=[]
from fastai.losses import LabelSmoothingCrossEntropyFlat

conv5 = nn.Sequential(
    conv3_block(1, 16),
    conv3_block(16, 32),
    conv3_block(32, 64),
    
    nn.Conv2d(64, 10, 2),
    nn.Flatten(),
)

Learner(
    dls=mnist_augmented_loaders2,
    model=conv5,
    loss_func=LabelSmoothingCrossEntropyFlat(eps=0.075),
    metrics=[accuracy],
).fit_one_cycle(10, 0.01)

# + tags=[]
plot_probability_histogram(conv5)

# + tags=[]
plot_probability_histogram(conv5, max_probability=0.9)

# + tags=[]
plot_top_loss_probabilities(conv5, k=16)

# + tags=[]
plot_top_unsure_probabilities(conv5, k=16)


# -

# ## Convolutional weights vizualization

# + [markdown] tags=[]
# Understanding what neural networks learn is one of the major research topics in machine learning. [Matthew Zeiler and Rob Fergus](https://arxiv.org/abs/1311.2901) analyzed convolutional networks trained on [ImageNet](https://www.image-net.org/about.php) and provided great visualizations of layer weights and kernels learned.
#
# Links
# - https://arxiv.org/abs/1311.2901
# - https://nbviewer.org/github/fastai/fastbook/blob/master/01_intro.ipynb

# + [markdown] tags=[]
# <img src="./images/cnn_layer1.png" alt="Layer 1" width="50%">

# + [markdown] tags=[]
# <img src="./images/cnn_layer2.png" alt="Layer 2" width="100%">

# + [markdown] tags=[]
# <img src="./images/cnn_layer3.png" alt="Layer 3" width="100%">

# + [markdown] tags=[]
# <img src="./images/cnn_layer4.png" alt="Layer 4" width="100%"> 

# + [markdown] tags=[]
# <img src="./images/cnn_layer5.png" alt="Layer 5" width="100%">
# -

# ## Thanks

# Thanks for reading/listening/watching. 
#
# Now that we more or less have the basics covered, in the next talk I plan to cover something more complicated, like implementing your own google translate.
#
# See you next time 👋

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ## Residual / skip connections experiments

# + tags=[]
class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, 3, padding=1),
            nn.BatchNorm2d(channels_out),
        )
        self.conv[-1].weight.data.fill_(0)
        
        self.skip = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1),
            nn.BatchNorm2d(channels_out),
            nn.AvgPool2d(2, ceil_mode=True),
        )
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return self.activation(self.conv(x) + self.skip(x))

res1 = nn.Sequential(
    ResBlock(1, 16),
    ResBlock(16, 32),
    ResBlock(32, 64),
    
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 10),
)

mnist_learner(res1).summary()

# + tags=[]
mnist_learner(res1, dls=mnist_augmented_loaders2).fit_one_cycle(5, 0.01)

# + tags=[]
Interpretation.from_learner(mnist_learner(res1)).plot_top_losses(k=24, ncols=6, figsize=(12, 5))