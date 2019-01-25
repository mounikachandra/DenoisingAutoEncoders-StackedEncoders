# DenoisingAutoEncoders-StackedEncoders
Autoencoders are feedforward neural networks where input are same as the output. These are dimensionality reduction /compression Algorithm
## AUTO ENCODERS
There are certain important properties:
<b>Data-specific:</b> Autoencoders are only able to meaningfully compress data similar to what they have been trained on. Since they learn features specific for the given training data, they are different than a standard data compression techniques. So we can’t expect an autoencoder trained on handwritten digits to compress landscape photos.
<b>Lossy:</b> The output of the autoencoder will not be exactly the same as the input, it will be a close but degraded representation. If you want loss less compression they are not the way to go.</br>

## Denoising Autoencoder
Compressed data is easier to handle. you can learn automatic features from few seconds of data instead of extracting heuristic based features. Denoising is good because you distort the data and add some noise in it that can help in generalizing.
  
