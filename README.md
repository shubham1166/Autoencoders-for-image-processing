# Autoencoders
---
## Autoencoders
**Autoencoders**: An autoencoder is a special type of feed-forward neural network which does the following:

-  *Encodes* its input into a hidden representation

-  *Decodes* its input again from this hidden representation.
So the input and output is same in a simple autoencoder.The model is trained to minimize a certain loss function which will ensure that input is close to output
![](https://lh6.googleusercontent.com/kohYji2VOZmTHMvCizq0crmFdnJPghFJPWCh4U8FjP1wE_QYG93d_Q-qm4UGrEFZP_PORT-vVkwalztc1-Sj58uWWRmqJs8eta3fiq6B4CwTixFy6cbs8NP5Uf5yi8_RVIz6hKTH)

Depending on the dimension of the hidden layer h, an autoencoder can be of two types:

- **Under complete autoencoder**: An autoencoder where dimension of the hidden layer is less than that of dimension of input/output layer is called an under complete autoencoder.

- **Over complete autoencoder**: An autoencoder where dimension of the hidden layer is more than that of dimension of input/output layer is called an over complete autoencoder. In such cases, the autoencoder could learn a trivial encoding by simply copying input into h and then by copying h into output. So, we usually have to apply some regularization in these type of autoencoders in order to avoid those identity mappings.

*g* and *f* are activation functions that we can change as per the demand of network like sigmoid, RELU, leakly RELU, tanh, linear etc. There can be different loss functions that we can use like cross_entropy, mean square error but i have used mean square error in most of my codes.

---
## Link b/w PCA and autoencoders
We'll show that the encoder part of an autoencoder is equivalent to PCA if we 

- use a linear autoencoder
- use a linear decoder
- use squared error loss function
- normalize the inputs 
**![](https://lh5.googleusercontent.com/k5GJGRI4Qyf-0SL-6PYxTzVFUWTcMMQhrsI8q14bfVwGRJmPSEH4TMnvxlGdeMIcJBaRfLlyg4GL4jdkaDLkh5AaEieHpmZBxCdJEtEf2ly5Mp5g7L2UtFYvkPZUXQ972SzNOlOw)**

These are the conditions under which PCA and autoencoders can behave similarly(theoritical proof can be found [here](https://www.youtube.com/watch?v=0ZQxPIwuA4o&list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT&index=53))

---
## Regularization in autoencoders
While poor generalization can happen in over complete and undercomplete autoencoders, it is more serious problem for overcomplete autoencoders. Here the model can simply learn to copy input into hidden layers and then to output layers. We can do regularization in many ways:


- We can add L2 norm or L1 norm
- Adding dropouts
- Adding forbeneous norm of the jacobian of nodes of hidden layers with respect to input layers

---
## Convolutional autoencoders
Convolutional autoencoders are the autoencoders in which we use convolutional layers instead of dense layers. I have implemented a convolutional autoencoder in this project trying to replicate VGG model in downsampling and upsampling.

We use convolutional autoencoders in case of images. It can be used for anomaly detection in images as that of i have done my other project and one of the major use of convolutional autoencoders can be that we can train the autoencoder and then use those weights/model in segmentation or other techniques(transfer learning)


---
## Variational autoencoders
![](https://lh4.googleusercontent.com/TMmPiAbkBu0Q_CJhkXw5nw_-mEsQP92ngGck-QR5PuRuoMSF9oG4IxWH7wmW2Hb2mxRHy072CH5BsbxgaLBHlCQtVkHnHrjCVRNEeNgypaqqaU7Lb8zi-v5WHb4NUvYY98Ulu6vD)

Variational autoencoder is a generative kind of autoencoder. The main idea of a variational autoencoder is to add some noise in the encoded output of encoder and then see the results that we get from that noise latent variable. Below is the image with some glance of how variational autoencoder behaves. Also this a good resource for further details of [variational autoencoder](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)


![Animated GIF](https://media.giphy.com/media/26ufgj5LH3YKO1Zlu/giphy.gif)

---
## Denoising autoencoder

A denoising autoencoder simply corrupts the input data using a probabilistic process before feeding it to the network. The out remains same as the noise free data. This helps the autoencoder to learn to remove noise from the data.


**![](https://lh4.googleusercontent.com/3M7OZ5v4uicxmLsb-iWZOyCTgCYZjFR38AmMCJtKucXVHAmHcT-KWfeQb9nI0dETZFJuQBPE4TJDPWDYsZdSFlKYKPmehapzRdJQgzxiPC9fwh-wky74Pc1o6MYNdwBxdnff_5Ty)**\


---
## Contractive autoencoder
A Contractive autoencoder tries to prevent an overcomplete autoencoder from learning the identity function. It does so by adding the frobenius norm of the  jacobian of the encoder w.r.t the input to the loss function.

**![](https://lh3.googleusercontent.com/CCwwNWDkljS8VJx5xliVFBE9ZwW8hTQrsy89AoK7R3UIaiF_1K76aqWwZMqWTJbIaRz_xjBe_CS8CxKGoorFMqQhpxYfoxj1DDZS9Kc1x1xDIb9VhiKyMypdkNfyW81bll-I95-8)**
**![](https://lh6.googleusercontent.com/_iKun7RWkUthafeaZ6UBNj4DRtzORAe8LhcE3PrAUNB-z-ZXb6I9nSOTMKppfpkbcdN_r63nQVwHgVS6PXTLDERhnRaZYNKOcq4U1xzNorm002Vmlp84923mXOEVclbzBZ18Sd23)**

---
## Sparse autoencoders
Like that of contractive autoencoders, we mostly use sparse autoencoders for regularization in overcomplete autoencoders. The basic idea of sparse autoencoder is it tries to ensure that the neurons is inactive most of the times. A sparse autoencoder uses a sparsity parameter \rho (typically very close to 0, say 0.005). The regularization term is as follows:

**![](https://lh5.googleusercontent.com/AFzShBO3nTR_l-XtM76eHBnf2j4X2OUuZ7fw2uZ0AfZtUqDt1s7je0xjqyy0gaGbly2k4taW_rlmNS2dwdGeT2ohQwfy68DXJc76Hqr9Z8ee31a8JXArHszdL2gXnctCAn8bo9Lm)**
**![](https://lh5.googleusercontent.com/BykFRQ137HHeN_UeZtfRDDa7T-Ztr70y-mUj5i3ZMIIsDsGTcu1jfvPcEAyE7GUPcEp6cFMypPJchVVGJ6dC_PDyTFXgSp4747UPw2TnODhPp_ftWvIOt6yxzKucn_hjQtFMRlCw)**
