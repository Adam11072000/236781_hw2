r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
1.
  a.For a singular output we need "in_features"*"out_features"*"Num_of_samples", in our case that is 
  1024*512*64.Therefore for all the output samples (64) we get 1024*512*64*64.
  b.In our case the Jacobian matrix is not sparse. Considering every output elements is affected by all input
  elements (because the graph is fully connected) it is not possible to get a sprase matrix unless all the 
  elements are zero.
  c.we do not need to materialize the above Jacobian in order to calculate the downstream gratdient w.r.t. to 
    the input, to do that we need to Transpose W to have appropriate dimensions for dy/dx, then we 
    perform dy/dx * W_transpose. The result will have the same dimensions as the input tensor.
    


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
2.
  a.In this case considering the Jacobian is with respect to W, it will have the same dimension as the weight
    tensor,therefore it will be 512*1024.
  b.In this case the Jacobian matrix IS sparse,that is due to the fact that the output is affected only by 
    the weights.To be more preciese each output element in y depends on the corresponding row of w.
    Therefore, for a given row in W, all elements except the ones in that row will be zero in
    the Jacobian dy/dw. 
  c.again there is no need to materialize, to do that we need to transpose X, then we calculate 
    x_transpose*dy/dx, as mentioned above the result will have the same dimensions as the input.
 



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.5, 0.05, 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.002
    lr_momentum = 0.003
    lr_rmsprop = 0.00015
    reg = 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr =  0.002
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
Yes, the results do meet our expectations, as we can see from the results (the epochs) the accuracy when going over the train set is highest when
the dropout value is zero, followed by 0.4 and lastly 0.8, that is due to picking the hyperparameters in way that gives is higher overfitting with lower dropout values, we can also see that using dropout regularization gives us higher test set accuracy although that is achieved by the later epochs as expected.



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**
Yes that is possible.
The cross entropy loss takes into account all the probabilities' distributions, therefore the train loss depends on all the scores of each class.
On the other hand the prediction is the highest scoring class.
Given the following scenario we can see an increase in both the test loss and accuracy:
Suppose in a given epoch (not the first one) the number of correct prediction increased from the previous one, and the scores of the highest scoring class do not differ from the other classes (which did not happen before considering we have highest scoring class) the accuracy will increase (better predictions) and so will the loss.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**
1. Gradient descent is an optimization algorithm used to update parameters,on the other hand backpropagation is the algorithm used to calculate the gradients needed for parameter updates in neural networks.
2. We will firstly go over the differnces quickly then we will go into detail about each comparable attribute.
Gradient descnet updates parameters based on the average gradient over the entire training dataset.
Stochastic gradient descent updates parameters using the gradient computed for a single training example at a time.
Time Complexity:
Regular GD requires much more time to run compared to SGD that is due to the fact that SGD computes the gradient for a singular training example at a given moment while GD calculates all the gradients for all the training set, making it significantlly slower when using larger batches of training sets.
this also affects the time needed for the algorithm to converge, GD takes more time to converge while SGD takes less time.
Bias,Variance and Generalization:
GD goes over the entire dataset therefore it provides a smoother therefore having less bias, while on the other hand SGD calculates over a singular example therefore it may have more bias and noise.
Adding to that logic, GD has better generalizaiton due to the fact that it takes all the data set therefore having a broader range of calculations in contrast to SGD which does does not share ability.
Updating:
In GD we update using the average gradient over all the dataset, on the other hand in SGD we update using a single training example iteratively.
3. To put it shortly, we can take the "Time Complexity" from above because we strive for faster computations and more effecient computing, moreover the noise we talked about in the second point which is added from taking into account a singular training example each time, acts as a regularization term which will prevent us from creating an overfitting model.
4.
A. 
In our opinion the suggested method does not produce an equivalent to regular GD on all the data, the main difference here is that GD works on all the data, meaning the gradiant in a singular epoch is calculated using an average of all the data in hand, the suggested method does an independent calculation on every batch then calculates the loss using the information exclusive to that batch then we do the backward pass. Mathematically summing then performing backward pass is the same as optimizing the sum of batches, however the difference lies in the optimization, where optimizing the losses compared to optimizing the sum.
In GD the gradient is with respect to the AVG loss over all the dataset, while in our case the respect is to the sum of losses of all batches.
B.
The more proccesing we do to batches the more memory will be used and piled up, this may be cause by all the secondary calculations that need to be done and stored for future use between the batches.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 2
    hidden_dims = 14
    activation = "relu"
    out_activation = "relu"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr, weight_decay, momentum = 0.02, 0.0005,0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ===== 
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.00275
    weight_decay = 0.00005
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**
1.The number of parameters for a CNN layer is C_in * k * k * C_out, in our case the number of parameters in a regular block is:
(256*3*3*256)+(256*3*3*256) = 1179648
The bottleneck block has:
(256*1*1*64) + (64*3*3*64) + (64*1*1*256) = 69632.
As we can clearly see the bottleneck has significantly less parameters.
2.The floating point operations 


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
we can see that the best results gotten with L = 2. this may be due to the fact that the network is very expressive, thus having more layers will give major overfitting.
1) theoritically, the deeper the network the more accurate it can become, due to the fact that
mathematically, the function that the network defines can become more expressive.
practically, the deeper the network the more overfitting we get, thus we can reach 100% accuracy on the training set but quite a low accuracy on the test set.
the best results were produced by L = 2, since it is not that deep, thus no overfitting, but not very shallow as to not being expressive.
2) 8, 16. this was caused because of vanishing/exploding gradients, thus the network was not trainable.
two things helped, one of them is weight_decay, which prevents exploding gradients, and early stoppings which stops the network from training if no better results on test set, thus preventing vanishing/exploding gradients.
"""

part5_q2 = r"""
for a non-expressive network (not very deep like L = 2), best results gotten were from K = 128, as these kernels can detect a much bigger overview of the pictures, that can classify better.
for a mid-expressive netowkr (L = 4), best results were still from K = 128 due to the same reason as above.
for a very expressive network, L = 8, a small kernel can now get very complex results from smaller features, thus K = 64 gave the best results.

We can see that the best results where gotten for L = 4, K = 128.
we got better score than q1 because of the fact that the network could detect more complex patterns.

"""

part5_q3 = r"""
the best results were gotten for L = 2, and K = [64, 128]
combining two kernels with different sizes can help detect very complex patterns in pictures, thus we see that when we deepen the network, we already see overfitting.

"""

part5_q4 = r"""
resnets usually can classify pictures much better than CNNs as we saw in lectures.
we can see that we got the best results for L=16 AND K=32.
small difference was gotten for L = 8 and  k = 32, this is because resnets can be trained with deeper networks since they address the problem of vanishing/exploding gradients.
as for L = 32, we see a very noticable degrading in test score due to the fact of overfitting.
the training took much more time than CNN's.

as for varying sizes of Kernels, this could make the network too much expressive as to induce overfitting and get worse results.
Although theoritically, if hyper parameters are tweaked good enough, with varying number of kernels, the network can get very good results, as we have no
vanishing/exploding gradients in ResNets.
"""

# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""