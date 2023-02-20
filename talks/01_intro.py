# ---
# jupyter:
#   jupytext:
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

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# # Neural networks
#
# Intro

# + [markdown] tags=[] slideshow={"slide_type": "slide"}
# ## Plan
#
# 1. Fitting a simple function
# 3. Handwritten digit recognition (MNIST)
# 4. Plans for future talks

# + slideshow={"slide_type": "skip"} tags=[]
# %matplotlib inline

import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import torch


# + [markdown] tags=[] slideshow={"slide_type": "slide"}
# ## Fitting a simple function

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# For a simple example of how neural networks work, let's try to guess parameters of a quadratic function. 
#
# If you don't remember from school, a quadratic looks like this:
# -

def quad(a, b, c):
    return lambda x: a * x**2 + b * x + c


# + slideshow={"slide_type": "skip"} tags=[]
def interactive_quad_sliders():
    slider_args = dict(
        min=-2, 
        max=2, 
        step=0.05, 
        continuous_update=False
    )
    
    a = widgets.FloatSlider(description="a", value=1, **slider_args)
    b = widgets.FloatSlider(description="b", value=1, **slider_args)
    c = widgets.FloatSlider(description="c", value=1, **slider_args)
    return a, b, c


# + slideshow={"slide_type": "skip"}
def interactive_quad():
    def render(a, b, c):
        x = np.linspace(-2, 2, 200)
        _, ax = plt.subplots(figsize=(4,3))
        ax.plot(x, quad(a,b,c)(x))
        ax.grid(True)
        plt.show()
    
    a, b, c = interactive_quad_sliders()
    
    display(
        widgets.HBox([a, b, c]),
        widgets.interactive_output(render, dict(a=a, b=b, c=c))
    )


# -

interactive_quad()

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# Let's generate a random quadratic function.
# -

target_params = tuple(np.random.uniform(-2, 2, 3))
target = quad(*target_params)
print("(a, b, c) =", target_params)


# + slideshow={"slide_type": "skip"}
def plot_target():
    x = np.linspace(-2, 2, 200)
    _, ax = plt.subplots(figsize=(4,3))
    ax.plot(x, target(x))
    ax.grid(True)
    plt.show()


# -

plot_target()

# + [markdown] slideshow={"slide_type": "slide"}
# ...and sample some points from it with noise, so the data isn't perfect.
# -

x, _ = torch.sort(torch.from_numpy(np.random.uniform(-2, 2, 100)))
y = target(x) + np.random.uniform(-0.1, 0.1, x.shape[0])


# + slideshow={"slide_type": "skip"}
def plot_samples():
    target_x = np.linspace(-2, 2, 200)
    target_y = target(target_x)
    
    _, ax = plt.subplots(figsize=(4,3))
    ax.scatter(x, y, s=3, c="C0", label="samples")
    ax.plot(target_x, target_y, c="C1", label="target")
    ax.grid(True)
    ax.legend()
    
    plt.show()


# -

plot_samples()


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# Guessing the parameters visually

# + slideshow={"slide_type": "skip"}
def fit_visually():
    def render(a, b, c):
        _, ax = plt.subplots(figsize=(4,3))
        ax.grid(True)
        ax.scatter(x, y, s=3, c="C0", label="samples")
        ax.plot(x, quad(a,b,c)(x), c="C1", label="guess")
        ax.legend()
        plt.show()
    
    a, b, c = interactive_quad_sliders()
    
    display(
        widgets.HBox([a, b, c]),
        widgets.interactive_output(render, dict(a=a, b=b, c=c))
    )


# + slideshow={"slide_type": "-"}
fit_visually()


# + [markdown] slideshow={"slide_type": "slide"}
# How can we automate this process?

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# First off, we need to define how far away we are from the target. 
#
# We need a function, which will take the samples and our predictions and give us a score. This is called a **loss function**.
#
# Let's use mean squared error:
# -

def mean_squared_error(target, prediction):
    return ((target - prediction) ** 2).mean()


# + slideshow={"slide_type": "skip"} tags=[]
def fit_with_loss():
    def render(a, b, c):
        predict = quad(a, b, c)
        y_predicted = predict(x)

        loss = mean_squared_error(y, y_predicted)
        print("loss =", loss.item())

        _, ax = plt.subplots(figsize=(4,3))
        ax.scatter(x, y, s=3, c="C0", label="samples")
        ax.plot(x, y_predicted, c="C1", label="guess")
        ax.grid(True)
        ax.legend()
        plt.show()
    
    a, b, c = interactive_quad_sliders()
    
    display(
        widgets.HBox([a, b, c]),
        widgets.interactive_output(render, dict(a=a, b=b, c=c))
    )


# -

fit_with_loss()


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# Now that we know how far we are, we need to figure out how to change the parameters to decrease the loss.
#
# Let's have a look at how our loss looks like.

# + tags=[] slideshow={"slide_type": "skip"}
def loss_line(predict):
    A = np.linspace(-2, 2, 50)    
    Y_predicted = predict(A)(np.broadcast_to(x, A.shape + x.shape).transpose())
    Y = np.broadcast_to(y, A.shape + y.shape).transpose()

    return A, ((Y - Y_predicted)**2).mean(axis=0)
    
def fit_with_loss_plots():
    def render(a, b, c):
        y_predicted = quad(a,b,c)(x)
        loss = mean_squared_error(y, y_predicted)

        print("loss =", loss.item())

        _, (a_ax, b_ax, c_ax) = plt.subplots(1, 3, figsize=(12,2), sharey=True)

        a_loss_line = loss_line(lambda a: quad(a, b, c))
        a_ax.plot(*a_loss_line, c="C0")
        a_ax.scatter(a, loss, c="C1")
        a_ax.set_xlabel("a")
        a_ax.set_ylabel("loss")
        a_ax.set_ylim([0, None])
        a_ax.grid(True)

        b_loss_line = loss_line(lambda b: quad(a, b, c))
        b_ax.plot(*b_loss_line, c="C0")
        b_ax.scatter(b, loss, c="C1")
        b_ax.set_xlabel("b")
        b_ax.grid(True)

        c_loss_line = loss_line(lambda c: quad(a, b, c))
        c_ax.plot(*c_loss_line, c="C0")
        c_ax.scatter(c, loss, c="C1")
        c_ax.set_xlabel("c")
        c_ax.grid(True)

        _, fit_ax = plt.subplots(figsize=(3,2))
        fit_ax.scatter(x, y, s=3, c="C0", label="samples")
        fit_ax.plot(x, y_predicted, c="C1", label="guess")
        fit_ax.set_xlabel("x")
        fit_ax.set_ylabel("y")
        fit_ax.grid(True)
        fit_ax.legend()

        plt.show()
        
    a, b, c = interactive_quad_sliders()
    
    display(
        widgets.HBox([a, b, c]),
        widgets.interactive_output(render, dict(a=a, b=b, c=c))
    )


# + tags=[]
fit_with_loss_plots()


# + [markdown] slideshow={"slide_type": "slide"}
# We can calculate the slope of the loss function and move the parameters in the direction of decreasing loss.
#
# The slope at a given point is called a **gradient** and so we have re-invented **stochastic gradient descent**.

# + tags=[] slideshow={"slide_type": "skip"}
def loss_with_gradient(a, b, c):
    params = torch.tensor([a, b, c], requires_grad=True)
    y_predicted = quad(*params)(x)
    
    loss = mean_squared_error(y, y_predicted)
    loss.backward()
    
    return (
        y_predicted.detach().numpy(), 
        loss.detach().numpy(), 
        params.grad.detach().numpy()
    )

def fit_with_gradient():
    def render(a, b, c):
        y_predicted, loss, grad = loss_with_gradient(a, b, c)
        lr = (loss / grad ** 2).min()

        print("loss     =", loss)
        print("gradient =", tuple(grad))

        _, (a_ax, b_ax, c_ax) = plt.subplots(1, 3, figsize=(12,2), sharey=True)

        a_ax.axline([a, loss], [a+1, loss+grad[0]], c="C0")
        a_ax.quiver(a, loss, -lr * grad[0], 0, color="C1", angles='xy', scale_units='xy', scale=1)
        a_ax.scatter(a, loss, c="C1")
        a_ax.set_xlabel("a")
        a_ax.set_ylabel("loss")
        a_ax.set_xlim([-2, 2])
        a_ax.set_ylim([0, None])
        a_ax.grid(True)

        b_ax.axline([b, loss], [b+1, loss+grad[1]], c="C0")
        b_ax.quiver(b, loss, -lr * grad[1], 0, color="C1", angles='xy', scale_units='xy', scale=1)
        b_ax.scatter(b, loss, c="C1")
        b_ax.set_xlabel("b")
        b_ax.set_xlim([-2, 2])
        b_ax.grid(True)

        c_ax.axline([c, loss], [c+1, loss+grad[2]], c="C0")
        c_ax.quiver(c, loss, -lr * grad[2], 0, color="C1", angles='xy', scale_units='xy', scale=1)
        c_ax.scatter(c, loss, c="C1")
        c_ax.set_xlabel("c")
        c_ax.set_xlim([-2, 2])
        c_ax.grid(True)

        _, fit_ax = plt.subplots(figsize=(3,2))
        fit_ax.scatter(x, y, s=3, c="C0", label="samples")
        fit_ax.plot(x, y_predicted, c="C1", label="guess")
        fit_ax.set_xlabel("x")
        fit_ax.set_ylabel("y")
        fit_ax.grid(True)
        fit_ax.legend()

        plt.show()

    a, b, c = interactive_quad_sliders()
    
    display(
        widgets.HBox([a, b, c]),
        widgets.interactive_output(render, dict(a=a, b=b, c=c))
    )


# + tags=[]
fit_with_gradient()


# + [markdown] slideshow={"slide_type": "slide"}
# We can now write the code which will do SGD for us and find our function parameters.
# -

def fit_sgd(epochs, learning_rate, init_params):
    # generate random parameters between -2 and 2
    params = torch.tensor(
        np.random.uniform(-2, 2, 3) if init_params is None else init_params, 
        requires_grad=True
    )
    
    loss_history = []
    for _ in range(epochs):
        # evaluate our model
        y_predicted = quad(*params)(x)

        # calculate the loss and the loss gradient
        loss = mean_squared_error(y, y_predicted)
        loss.backward()

        # move towards decreasing loss
        with torch.no_grad():
            params.add_(-params.grad * learning_rate)
            params.grad.zero_()
            
        loss_history.append(loss.item())
    
    # calculate final loss
    y_predicted = quad(*params)(x)
    loss = mean_squared_error(y, y_predicted)
    loss_history.append(loss.item())
    
    return params.tolist(), loss.item(), loss_history


# + slideshow={"slide_type": "skip"}
def interactive_fit_sgd():
    init_params = np.random.uniform(-2, 2, 3)
    
    def render(epochs, learning_rate):
        params, loss, loss_history = fit_sgd(epochs, learning_rate, init_params)
        
        _, (loss_ax, fit_ax) = plt.subplots(1, 2, figsize=(8,3))
        
        loss_ax.plot(loss_history)
        loss_ax.set_xlabel("iteration")
        loss_ax.set_ylabel("loss")
        loss_ax.grid(True)
        
        fit_ax.scatter(x, y, s=3, c="C0", label="samples")
        fit_ax.plot(x, quad(*params)(x), c="C1", label="guess")
        fit_ax.set_xlabel("x")
        fit_ax.set_ylabel("y")
        fit_ax.grid(True)
        fit_ax.legend()
        
        plt.show()
        
        print("params =", params)
        print("target =", list(target_params))
        print("loss   =", loss)
        
    epochs = widgets.IntSlider(
        description="epochs", 
        value=15, 
        min=1, 
        max=50, 
        step=1, 
        continuous_update=False
    )
    learning_rate = widgets.FloatLogSlider(
        description="learning rate", 
        base=10,
        value=0.2, 
        min=-3, 
        max=0, 
        step=0.05,
        continuous_update=False,
    )
    
    display(
        widgets.HBox([epochs, learning_rate]),
        widgets.interactive_output(render, dict(epochs=epochs, learning_rate=learning_rate))
    )


# + [markdown] slideshow={"slide_type": "slide"}
# ...and if we run our SGD implementation it approximates the parameters pretty well.

# + slideshow={"slide_type": "-"}
interactive_fit_sgd()


# + [markdown] slideshow={"slide_type": "slide"}
# OK... but what if I don't know what the target function is?
#
# How do I know what parameters there are and how they are used?

# + [markdown] slideshow={"slide_type": "slide"}
# We can approximate any function with a bunch of linear segments.
# -

def linear(a, b):
    return lambda x: a*x + b


def linear_model():
    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    
    def predict(x):
        return linear(a, b)(x)
    
    return predict, [a, b]


# + slideshow={"slide_type": "-"}
def fit_model_sgd(model, epochs, learning_rate):
    predict, params = model
    
    loss_history = []
    for _ in range(epochs):
        # evaluate our model
        y_predicted = predict(x)

        # calculate the loss and the loss gradient
        loss = mean_squared_error(y, y_predicted)
        loss.backward()

        # move towards decreasing loss
        with torch.no_grad():
            for param in params:
                param.add_(-param.grad * learning_rate)
                param.grad.zero_()

        loss_history.append(loss.item())
    
    # calculate final loss
    y_predicted = predict(x)
    loss = mean_squared_error(y, y_predicted)
    loss_history.append(loss.item())
    
    return loss.item(), loss_history


# + slideshow={"slide_type": "skip"}
def interactive_fit_model_sgd(model):
    predict, params = model
    init_params = [param.detach().clone() for param in params]
    
    def render(epochs, learning_rate):
        with torch.no_grad():
            for param, init_value in zip(params, init_params):
                param.zero_()
                param.add_(init_value)
        
        loss, loss_history = fit_model_sgd(model, epochs, learning_rate)
        
        _, (loss_ax, fit_ax) = plt.subplots(1, 2, figsize=(8,3))
        
        loss_ax.plot(loss_history)
        loss_ax.set_xlabel("iteration")
        loss_ax.set_ylabel("loss")
        loss_ax.grid(True)
        
        fit_ax.scatter(x, y, s=3, c="C0", label="samples")
        fit_ax.plot(x, predict(x).detach(), c="C1", label="guess")
        fit_ax.set_xlabel("x")
        fit_ax.set_ylabel("y")
        fit_ax.grid(True)
        fit_ax.legend()
        
        plt.show()
        
        print("loss   =", loss)
        
    epochs = widgets.IntSlider(
        description="epochs", 
        value=15, 
        min=0, 
        max=100, 
        step=1, 
        continuous_update=False
    )
    learning_rate = widgets.FloatLogSlider(
        description="learning rate", 
        base=10,
        value=0.2, 
        min=-6, 
        max=0, 
        step=0.05,
        continuous_update=False,
    )
    
    display(
        widgets.HBox([epochs, learning_rate]),
        widgets.interactive_output(render, dict(epochs=epochs, learning_rate=learning_rate))
    )


# -

interactive_fit_model_sgd(linear_model())


# +
def relu(x):
    return torch.max(x, torch.tensor(0))

def nonlinear_model():
    a1 = torch.randn((1, 50), requires_grad=True)
    b1 = torch.randn(50, requires_grad=True)
    a2 = torch.randn((50, 1), requires_grad=True)
    b2 = torch.randn(1, requires_grad=True)
    
    def predict(x):
        p = x.float().reshape(*x.shape, 1)
        p = relu(p @ a1 + b1)
        p = p @ a2 + b2
        return p.reshape(*x.shape)
        
    return predict, [a1, b1, a2, b2]


# -

interactive_fit_model_sgd(nonlinear_model())


def interactive_relu():
    def render(a1, b1, a2, b2, a3, b3):
        x = np.linspace(-2, 2, 200)
        l1 = a1 * x + b1
        l2 = a2 * np.maximum(l1, 0) + b2
        l3 = a3 * np.maximum(l2, 0) + b3
        
        _, ax = plt.subplots(figsize=(4,3))
        ax.plot(x, l1)
        ax.plot(x, l2)
        ax.plot(x, l3)
        ax.grid(True)
        plt.show()
    
    slider_args = dict(
        min=-10, 
        max=10, 
        step=0.05, 
        continuous_update=False
    )
    
    a1 = widgets.FloatSlider(description="a1", value=1, **slider_args)
    b1 = widgets.FloatSlider(description="b1", value=1, **slider_args)
    a2 = widgets.FloatSlider(description="a2", value=1, **slider_args)
    b2 = widgets.FloatSlider(description="b2", value=1, **slider_args)
    a3 = widgets.FloatSlider(description="a3", value=1, **slider_args)
    b3 = widgets.FloatSlider(description="b3", value=1, **slider_args)
    
    display(
        widgets.HBox([a1, b1]),
        widgets.HBox([a2, b2]),
        widgets.HBox([a3, b3]),
        widgets.interactive_output(render, dict(a1=a1, b1=b1, a2=a2, b2=b2, a3=a3, b3=b3))
    )


interactive_relu()


