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
        x = torch.linspace(-2, 2, 200)
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

target_params = (torch.rand(3) * 4 - 2).tolist()
target = quad(*target_params)
print("(a, b, c) =", target_params)


# + slideshow={"slide_type": "skip"}
def plot_target():
    x = torch.linspace(-2, 2, 200)
    _, ax = plt.subplots(figsize=(4,3))
    ax.plot(x, target(x))
    ax.grid(True)
    plt.show()


# -

plot_target()

# + [markdown] slideshow={"slide_type": "slide"}
# ...and sample some points from it with noise, so the data isn't perfect.

# +
x, _ = torch.sort(torch.rand(200) * 4 - 2)
x = x.reshape(200, 1)
y = target(x) + torch.randn_like(x) * 0.02

x.shape, y.shape


# + slideshow={"slide_type": "skip"}
def plot_samples():
    target_x = torch.linspace(-2, 2, 200)
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
    return ((target - prediction) ** 2).mean(axis=0)


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
    a = torch.linspace(-2, 2, 50)
    y_predicted = predict(a)(x)
    
    return a, mean_squared_error(y, y_predicted)
    
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
        y_predicted.detach(), 
        loss.detach(), 
        params.grad.detach()
    )

def fit_with_gradient():
    def render(a, b, c):
        y_predicted, loss, grad = loss_with_gradient(a, b, c)
        lr = (loss / grad ** 2).min()
        
        print("loss     =", loss.item())
        print("gradient =", grad.tolist())

        _, (a_ax, b_ax, c_ax) = plt.subplots(1, 3, figsize=(12,2), sharey=True)

        a_ax.axline([a, loss.item()], [a+1, (loss+grad[0]).item()], c="C0")
        a_ax.quiver(a, loss, -lr * grad[0], 0, color="C1", angles='xy', scale_units='xy', scale=1)
        a_ax.scatter(a, loss, c="C1")
        a_ax.set_xlabel("a")
        a_ax.set_ylabel("loss")
        a_ax.set_xlim([-2, 2])
        a_ax.set_ylim([0, None])
        a_ax.grid(True)

        b_ax.axline([b, loss.item()], [b+1, (loss+grad[1]).item()], c="C0")
        b_ax.quiver(b, loss, -lr * grad[1], 0, color="C1", angles='xy', scale_units='xy', scale=1)
        b_ax.scatter(b, loss, c="C1")
        b_ax.set_xlabel("b")
        b_ax.set_xlim([-2, 2])
        b_ax.grid(True)

        c_ax.axline([c, loss.item()], [c+1, (loss+grad[2]).item()], c="C0")
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

def fit_sgd(model, init_params, epochs, learning_rate):
    # prepare params for calculating loss gradient
    params = [param.clone().requires_grad_(True) for param in init_params]
    
    loss_history = []
    for _ in range(epochs):
        # evaluate our model
        y_predicted = model(*params)(x)

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
    y_predicted = model(*params)(x)
    loss = mean_squared_error(y, y_predicted)
    loss_history.append(loss.item())
    
    params = [param.detach() for param in params]
    return params, loss.item(), loss_history


# + slideshow={"slide_type": "skip"}
def interactive_fit_quad_sgd():
    init_params = torch.rand(3) * 4 - 2
    
    def render(epochs, learning_rate):
        params, loss, loss_history = fit_sgd(quad, init_params, epochs, learning_rate)
        
        _, (loss_ax, fit_ax) = plt.subplots(1, 2, figsize=(8,3))
        
        loss_ax.plot(loss_history)
        loss_ax.set_xlabel("iteration")
        loss_ax.set_ylabel("loss")
        loss_ax.grid(True)
        
        fit_ax.scatter(x, y, s=3, c="C0", label="samples")
        fit_ax.plot(x, quad(*params)(x), c="C1", label="prediction")
        fit_ax.set_xlabel("x")
        fit_ax.set_ylabel("y")
        fit_ax.grid(True)
        fit_ax.legend()
        
        plt.show()
        
        print("params =", [param.item() for param in params])
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
        value=0.1, 
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
interactive_fit_quad_sgd()


# + [markdown] slideshow={"slide_type": "slide"}
# OK... but what if I don't know what the target function is?
#
# How do I know what parameters there are and how they are used?

# + [markdown] slideshow={"slide_type": "slide"}
# We can approximate any function with a bunch of linear segments.

# + slideshow={"slide_type": "skip"}
def interactive_linear_approximation():
    def render(segment_count):
        segment_i = [*range(0, len(x) - 1, round((len(x) - 1) / segment_count)), len(x) - 1]
        
        _, ax = plt.subplots(figsize=(4,3))
        
        ax.scatter(x, y, s=3, c="C0", label="samples")
        ax.plot(x[segment_i], y[segment_i], c="C1", label="approximation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.legend()
        
        plt.show()
        
    segment_count = widgets.IntSlider(
        description="segments", 
        value=3, 
        min=1, 
        max=30, 
        step=1, 
        continuous_update=False
    )
    
    display(
        segment_count,
        widgets.interactive_output(render, dict(segment_count=segment_count))
    )


# -

interactive_linear_approximation()


# + [markdown] slideshow={"slide_type": "slide"}
# ...so what we should be able to do is generate a bunch of random linear functions and somehow combine them into a non-linear result.
#
# To introduce non-linear behavior we can use a Rectified Linear Unit (ReLU), which is just
# -

def relu(x):
    return torch.max(x, torch.tensor(0))


# + slideshow={"slide_type": "skip"}
def plot_relu():
    x = torch.tensor([-1, 0, 1]).reshape(3, 1)
    y = relu(x)
    
    _, ax = plt.subplots(figsize=(4,3))
    ax.plot(x, y)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.grid(True)
    plt.show()


# -

plot_relu()


# + [markdown] slideshow={"slide_type": "slide"}
# We can pass our linear functions through ReLU and sum them up. 
#
# A weighted sum will give us more flexibility when combining our linear segments.

# + slideshow={"slide_type": "skip"}
def interactive_relu_weighted_sum():
    def render(a1, b1, a2, b2, w1, w2, c):
        x = torch.linspace(-2, 2, 200)
        y1 = relu(a1 * x + b1)
        y2 = relu(a2 * x + b2)
        y = w1 * y1 + w2 * y2 + c
        
        fig, ax = plt.subplots(figsize=(8,3), layout='constrained')
        ax.plot(x, y1, label="relu(a1*x + b1)", c="C0")
        ax.plot(x, y2, label="relu(a2*x + b2)", c="C2")
        ax.plot(x, y, label="w1 * relu(a1*x + b1) + w2 * relu(a2*x + b2) + b", c="C1")
        ax.grid(True)
        ax.legend()
        plt.show()
    
    slider_args = dict(
        min=-10, 
        max=10, 
        step=0.05, 
        continuous_update=False
    )
    
    a1 = widgets.FloatSlider(description="a1", value=0.25, **slider_args)
    a2 = widgets.FloatSlider(description="a2", value=1, **slider_args)
    b1 = widgets.FloatSlider(description="b1", value=0.25, **slider_args)
    b2 = widgets.FloatSlider(description="b2", value=-0.75, **slider_args)
    w1 = widgets.FloatSlider(description="w1", value=1, **slider_args)
    w2 = widgets.FloatSlider(description="w2", value=1, **slider_args)
    c = widgets.FloatSlider(description="c", value=0.5, **slider_args)
    
    display(
        widgets.HBox([a1, b1]),
        widgets.HBox([a2, b2]),
        widgets.HBox([w1, w2, c]),
        widgets.interactive_output(render, dict(a1=a1, b1=b1, a2=a2, b2=b2, w1=w1, w2=w2, c=c))
    )


# -

interactive_relu_weighted_sum()


# + [markdown] slideshow={"slide_type": "slide"}
# Time to build our universal function.
# -

def universal_fn(a, b, w, c):
    def f(x):
        y = a * x + b
        y = relu(y)
        y = (y * w).sum(dim=1, keepdim=True) + c
        return y
        
    return f


# where `a`, `b` and `w` are vectors, which contain sets of parameters for each of our linear segments.

# + [markdown] slideshow={"slide_type": "slide"}
# ...and here's how our model performs

# + slideshow={"slide_type": "skip"}
def interactive_fit_universal_sgd():
    last_segment_count = None
    init_params = None
    
    def render(segment_count, epochs, learning_rate):
        nonlocal last_segment_count, init_params
        
        if init_params is None or last_segment_count != segment_count:
            init_params = [
                torch.randn(segment_count),
                torch.randn(segment_count),
                torch.randn(segment_count),
                torch.randn(1),
            ]

        last_segment_count = segment_count
        
        params, loss, loss_history = fit_sgd(universal_fn, init_params, epochs, learning_rate)
        
        _, (loss_ax, fit_ax) = plt.subplots(1, 2, figsize=(8,3))
        
        loss_ax.plot(loss_history)
        loss_ax.set_xlabel("iteration")
        loss_ax.set_ylabel("loss")
        loss_ax.grid(True)
        
        fit_ax.scatter(x, y, s=3, c="C0", label="samples")
        fit_ax.plot(x, universal_fn(*params)(x), c="C1", label="prediction")
        fit_ax.set_xlabel("x")
        fit_ax.set_ylabel("y")
        fit_ax.grid(True)
        fit_ax.legend()
        
        plt.show()
        
        print("loss =", loss)
        
    segment_count = widgets.IntSlider(
        description="segments", 
        value=5, 
        min=1, 
        max=30, 
        step=1, 
        continuous_update=False
    )
    epochs = widgets.IntSlider(
        description="epochs", 
        value=100, 
        min=0, 
        max=1000, 
        step=1, 
        continuous_update=False
    )
    learning_rate = widgets.FloatLogSlider(
        description="learning rate", 
        base=10,
        value=0.05, 
        min=-6, 
        max=0, 
        step=0.05,
        continuous_update=False,
    )
    
    display(
        widgets.HBox([segment_count, epochs, learning_rate]),
        widgets.interactive_output(render, dict(
            segment_count=segment_count, 
            epochs=epochs, 
            learning_rate=learning_rate
        ))
    )


# -

interactive_fit_universal_sgd()


# + [markdown] slideshow={"slide_type": "slide"}
# We've just built the simplest neural network!
# -

def network(x):
    y = x                                     # input layer
    y = a * x + b                             # linear layer
    y = relu(y)                               # activation
    y = (y * w).sum(dim=1, keepdim=True) + c  # output layer
    return y


# + slideshow={"slide_type": "skip"}
import graphviz

def network_graph():
    return graphviz.Source("""
        digraph network {
            rankdir=LR
            splines=line
            

            node [fixedsize=true, shape=circle, style=solid];

            subgraph cluster_input {
                label="Input\nLayer"
                labelloc="b"
                color=transparent
                
                x;
            }

            subgraph cluster_hidden {
                label="Hidden\nLinear\nLayer"
                labelloc="b"
                color=transparent
                node[label=""]
                nodesep=-1
                
                h1
                h2
                hellipsis [height=0.25, label="⋮", color="transparent"]
                hn
            }

            subgraph cluster_output {
                label="Output\nLayer"
                labelloc="b"
                color=transparent
                
                y;
            }
            
            x -> h1
            x -> h2
            x -> hellipsis [style=invis]
            x -> hn
            
            h1 -> y
            h2 -> y
            hellipsis -> y [style=invis]
            hn -> y
        }
    """)


# -

network_graph()
