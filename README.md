# DenoisingAutoEncoders-StackedEncoders
Autoencoders are feedforward neural networks where input are same as the output. These are dimensionality reduction /compression Algorithm
## AUTO ENCODERS
There are certain important properties:
<b>Data-specific:</b> Autoencoders are only able to meaningfully compress data similar to what they have been trained on. Since they learn features specific for the given training data, they are different than a standard data compression techniques. So we can’t expect an autoencoder trained on handwritten digits to compress landscape photos.
<b>Lossy:</b> The output of the autoencoder will not be exactly the same as the input, it will be a close but degraded representation. If you want loss less compression they are not the way to go.</br>

## Denoising Autoencoder
Compressed data is easier to handle. you can learn automatic features from few seconds of data instead of extracting heuristic based features. Denoising is good because you distort the data and add some noise in it that can help in generalizing.
<img width="796" alt="screen shot 2019-01-24 at 6 44 56 pm" src="https://user-images.githubusercontent.com/12842789/51720076-4ff8a300-2009-11e9-9f48-8f224142d068.png">

<img width="396" alt="screen shot 2019-01-24 at 6 54 14 pm" src="https://user-images.githubusercontent.com/12842789/51720148-a5cd4b00-2009-11e9-95bb-9c802898d64c.png">

The principle behind denoising autoencoders is to be able to reconstruct data from an input of corrupted data. After giving the autoencoder the corrupted data, we force the hidden layer to learn only the more robust features using a loss function and updating parameters by backpropagation. The output will then be a more refined version of the input data.
## Stacked Autoencoder
To "pretrain" a Deep Network. Stacked AE are trained in an unsupervised manner to obtain the parameters. This pretraining procedure creates an initialisation point, to which the parameters are restricted to a near optimal local minimum.
Typically, autoencoders are trained in an unsupervised, greedy, layer-wise fashion.

Each layer is trained in unsupervised way as an autoencoder to get parameters. All these trained layers are stacked together to form a Stacked Autoencoder. For Classification we finally add a classifier at the end and fine tune with only a few labelled samples using softmax and backpropagation to update parameters.

<img width="911" alt="screen shot 2019-01-24 at 6 58 48 pm" src="https://user-images.githubusercontent.com/12842789/51720285-399f1700-200a-11e9-8b0a-5bbe93c4feb0.png">

## Model Description

#### Preprocessing:
pre-processing the images by scaling them down by a factor of 255.
#### Denoising Autoencoder:
Input = Noisy images(784), hidden layers = 1, hidden neurons = 1000, output = same dim as input
Activation functions = RELU
Loss function = MSE/BCE
optimizers = SGD/ ADAM
#### Stacked Autoencoder(for classification):
Input = Noisy images(784), hidden layers = 3, hidden neurons = (500,300,100), output = numClasses(10)
Activation functions = RELU
Loss function = MSE/BCE, Softmax (classification) optimizers = SGD/ ADAM

## Experimentation
<pre>• We have total control over the architecture of the autoencoder. We can make it very powerful by increasing the number of layers, nodes per layer and the number of nodes in the middle layer. Increasing these hyperparameters will let the autoencoder to learn more complex coding. But we should be careful to not make it too powerful. Otherwise the autoencoder will simply learn to copy its inputs to the output, without learning any meaningful representation. It will just mimic the identity function. The autoencoder will reconstruct the training data perfectly, but it will be overfitting without being able to generalize to new instances
• Classification accuracy with stacked autoencoder weight initialization: 1 labelled sample per class – 70% (around)
5 labelled sample per class – 70% (around)
• Classification accuracy with random weight initialization:
1 labelled sample per class – 30% (around) 5 labelled sample per class – 30% (around)
As the samples are very low for deep network it just highly overfits and makes random guesses.</pre>

<img width="513" alt="screen shot 2019-01-24 at 7 04 42 pm" src="https://user-images.githubusercontent.com/12842789/51720484-16c13280-200b-11e9-86cb-513e42a219f9.png">

<img width="586" alt="screen shot 2019-01-24 at 7 09 23 pm" src="https://user-images.githubusercontent.com/12842789/51720633-b8e11a80-200b-11e9-9eaa-0afa7688099c.png">

<img width="626" alt="screen shot 2019-01-24 at 7 12 20 pm" src="https://user-images.githubusercontent.com/12842789/51720748-12494980-200c-11e9-9e93-f824ed436154.png">

<img width="585" alt="screen shot 2019-01-24 at 7 13 09 pm" src="https://user-images.githubusercontent.com/12842789/51720771-3147db80-200c-11e9-9877-f6655ce200e2.png">

<img width="553" alt="screen shot 2019-01-24 at 7 14 25 pm" src="https://user-images.githubusercontent.com/12842789/51720820-63f1d400-200c-11e9-8940-e19ba53340ed.png">

<img width="548" alt="screen shot 2019-01-24 at 7 15 23 pm" src="https://user-images.githubusercontent.com/12842789/51720844-897edd80-200c-11e9-9dfa-e24a85346064.png">
