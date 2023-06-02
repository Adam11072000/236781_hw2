r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
1.
  a.
  For a singular output we need "in_features"x"out_features"x"Num_of_samples", in our case that is 
  1024x512x64.Therefore for all the output samples (64) we get 1024x512x64x64.
  
  b.
  In our case the Jacobian matrix is sparse. Considering an output element y(i,j) this element only depends on the i'th sample therefore only the derivative by     the i'th sample will give us a non zero answer while all the others will result in zeros, which in turn will make the matrix sparse since each element has one     non zero value while all the others are zeros.
  
  c.
    we do not need to materialize the above Jacobian in order to calculate the downstream gradient w.r.t. to the input, to do that we need to Transpose W to have     appropriate dimensions for dy/dx, then we perform dy/dx * W_transpose. The result will have the same dimensions as the input tensor.
    
2.
  a.
    In this case considering the Jacobian is with respect to W, it will have the same dimension as the weight
    tensor,therefore it will be 512x1024x64x512.
    
  b.
    In this case the Jacobian matrix is sparse,that is due to the fact that the output is affected only by 
    the weights.To be more preciese each output element in y depends on the corresponding row of w.
    Therefore, for a given row in W, all elements except the ones in that row will be zero in
    the Jacobian dy/dw. 
    
  c.
    again there is no need to materialize, to do that we need to transpose X, then we calculate 
    x_transpose*dy/dx, as mentioned above the result will have the same dimensions as the input.


"""

part1_q2 = r"""
Back propagation is NOT required for training neural networks, that is due to the fact that we can manually compute all the gradients of the loss with respect to each paramater independently.
However it needs to be said that not using back propagation is not scalable because not using it can increase time complexity to a degree where it is virtually impossible to train neural networks without it.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.5, 0.03, 0
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
    lr_vanilla = 0.003
    lr_momentum = 0.003
    lr_rmsprop = 0.00016
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
1.
   Yes, the results do meet our expectations, as we can see from the results (the epochs) the accuracy when going over the train set is highest when the dropout      value is zero, followed by 0.4 and lastly 0.8, that is due to picking the hyperparameters in way that gives is higher overfiting with lower dropout values, we    can also see that using dropout regularization gives us higher test set accuracy although that is achieved by the later epochs as expected.
2.
   We must note that in dropout - 0.8 the model performed worse than 0.4 because of the fact that 0.8 turned out to be way too close to 1 which lead to too many      neurons being dropped, which in turn made an underfitting model.


"""

part2_q2 = r"""
Yes that is possible.
The cross entropy loss takes into account all the probabilities' distributions, therefore the train loss depends on all the scores of each class.
On the other hand the prediction is the highest scoring class.
Given the following scenario we can see an increase in both the test loss and accuracy:
Suppose in a given epoch (not the first one) the number of correct predictions increased from the previous one, and the scores of the highest scoring class do not differ from the other classes (which did not happen before considering we have highest scoring class) the accuracy will increase (better predictions) and so will the loss.

"""

part2_q3 = r"""
1.
Gradient descent is an optimization algorithm used to update parameters,on the other hand backpropagation is the algorithm used to calculate the gradients needed for parameter updates in neural networks.

2.
We will firstly go over the differnces quickly then we will go into detail about each comparable attribute.
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

3.
To put it shortly, we can take the "Time Complexity" from above because we strive for faster computations and more effecient computing, moreover the noise we talked about in the second point which is added from taking into account a singular training example each time, acts as a regularization term which will prevent us from creating an overfitting model.
4.

A. 

In our opinion the suggested method does not produce an equivalent to regular GD on all the data, the main difference here is that GD works on all the data, meaning the gradiant in a singular epoch is calculated using an average of all the data in hand, the suggested method does an independent calculation on every batch then calculates the loss using the information exclusive to that batch then we do the backward pass. Mathematically summing then performing backward pass is the same as optimizing the sum of batches, however the difference lies in the optimization, where optimizing the losses compared to optimizing the sum.
In GD the gradient is with respect to the AVG loss over all the dataset, while in our case the respect is to the sum of losses of all batches.

B.

The more proccesing we do to batches the more memory will be used and piled up, this may be cause by all the secondary calculations that need to be done and stored for future use between the batches.


"""

part2_q4 = r"""



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

1.High Optimization error: Yes.
The training loss did not stabilze after the given timeframe of epochs, we can give the system more epochs to train the model or give it a better suited optimizer
We can see that the training loss have not stabalized

2.High Generalization error: No.
the generalization factor in our case is decent enough not to decrease.
that is seen in by the face that the training and test losses are similar and there is no appearant under/over-fitting.

3.High Approximation error: Yes.
As seen the training accuracy can be a bit higher and the red labels can be classified better by chosing another model. (that can be achieved by chosing a better hypothesis class.



"""

part3_q2 = r"""
considering the original training set is split into the train and validation set, it is expected that the FPR and FNR rates are going to be close but as seen in the results that is not the case, therefore our next guess is that the FPR is higher.


"""

part3_q3 = r"""
1. 
Considering the symptoms are non-lethal and further testing is High-Risk. It is better to classify a person as non-sick when he is actually sick than classify him as sick when he is not sick and potentially risk his life in further testing.
therefore we want to keep the FPR as low as possible regardless of the TPR.


2.
in this case we want prioritize classifying as much suck people as possible especially when the disease is asymptomatic, therefore we want to keep the TPR as high as possible while also making sure the FPR stays down as well because we dont want to misdiagnose healthy people and risk their life with the expensive testing.



"""


part3_q4 = r"""
1.
As seen, increasing the width gives us the ability to get more from the dataset and in turn helps us make more accurate models to increase prediction percentage, however this is relevant only to a ceratin point, in which increasing the width even more will cause in a lower test accuracy which in turn means the model starts to overfit itself.

2.
The same that was said above can be said here, we do expect to see improvments when increasing the depth however to a certain point we start to overfit which in turn gives us an all around worse model.

3.
Firstly let us present the results:

Depth 1 - Width 32:
Test Accuracy - 84.8%
Validation Accuracy - 81.1%

Depth 4 - Width 8:
Test Accuracy - 86.0%
Validation Accuracy - 84.1%
As we can clearly see the later model (4,8) has better results in both the Test and the validation sets, even though the test accuracy is close to the validation accuracy we can still see that there are major differences between the two models. 

4.
Selecting our threshold based on the validation set makes the most sense since it helps us incorporate generalization into our model (that is something we strive to achieve) kinda like a hyper-parameter, that can be seen in the improved test scores we get in the model.



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
1.The number of parameters for a CNN layer is (C_in * k * k +bais)*C_out, in our case the number of parameters in a regular block is:
(256*3*3+1)*256+(256*3*3+1)*256 = 1180160
The bottleneck block has:
(256*1*1+1)*64 + (64*3*3+1)*64 + (64*1*1+1)*256 = 70016.
As we can clearly see the bottleneck has significantly less parameters.
2.The floating point operations 
The floating point operations for a CNN layer is C_in * k * k * C_out, in our case the number of parameters in a regular block is:
(256*3*3*256)+(256*3*3*256) = 1179648
The bottleneck block has:
(256*1*1*64) + (64*3*3*64) + (64*1*1*256) = 69632.
As we can clearly see the bottleneck still has significantly less parameters.
3.The ability to combine the input spatially is superior in a regular block due to the fact that applying the kernel convulotion twice gives us a wider recption field than doing it on a bottleneck block.
The ability to combine the input across feature maps is also superior in a regular block since we gain 2*256 feature maps instead of 64*2 +256 in the bottleneck



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
1.
The model did not do so good considering it only got 1 of the 6 objects correctly and it did not even identify some of the objects
2.
for the dolphin's case, we can argue that we did not have a dolphin class to begin with, therefore the model did not even get the chance to learn the class and therefore it detects other things (the model falsely detected a dolphin as a person with 0.9 which gives the indication that it did not even have a dolphin class)
for the cats/dogs case we know for ceratin that there is a dog class considering the model detected a dog once, however the training set may not have been good enough to train the model to detect accurately, maybe the model resorted to a color scheme that is resinates with cats and therefore when dogs of the same color were shown the model thought that they were cats.


"""


part6_q2 = r"""


"""


part6_q3 = r"""

The object in question - the screen monitor (classified as TV)
We will analyze each of the images we showed.
1- original image: as we can see the model is pretty good with detecting and classifying objects, the model only gets the remote and sharpie wrong but they are not relevant to our object.

2- occlusion: The model yet again detects the monitor (and most of the items) right, this shows that even with occlusion the model is still pretty good with detection, our guess is that the model used the rectangular edges of the monitor and the lighting coming out of it it detects it, however it is worth noting that the confidence level is lower now (in the 70's range) compared to the regular image

3- distortion: Again the model is successful in detection however the confidence seems to drop lower than before (in the 40's range) we can really see how distorting an image and giving the model something that is not typical or ideal may hurt the decisiveness of the program.

4- bias: This example was chosen carefully to trick the program and it worked! Using a setup that is very similar to a laptop's (keyboard directly under screen with contact between them) the model detected the monitor and keyboard as a laptop this shows that given a weird setup the program can wrongly detect.


"""

part6_bonus = r"""
There are 3 main changes we decided to impelment in order to improve our results:
1- resizing the image
2- adding contrast
3- adding sharpness

1- resizing the image (especially making it smaller dimensions) may improve the results becuase of many reasons including, preventing smearing (the bigger the image's dimensions the more smearing and fogginess the image will have), enhanced features because of the resolution and many more.

2- increasing the contrast may help with image detection because it makes details and little features shine more which may be a deciding factor in image detection (for example dogs nose vs cats nose) 

3- increasing sharpness, for similar reason of contrast, increasing sharpness will lead to more defined edges which may give us an edge in detecting something that was wrongly detected.

Overall we did see an improvement becuase the detected the cat (was not detected before) and the top right dog was rightly detected after previously being detected as a cat.


"""