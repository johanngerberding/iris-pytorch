# IRIS 

Here we want to reproduce the [IRIS paper](https://arxiv.org/pdf/2209.00588.pdf).

We have to build three components: 
* experience collection 
* world model (autoencoder, vqvae with an additional perceptual loss) 
* behavior model 

## World Model 

* basically a VQVAE with an additional perceptual loss 
* loss = reconstruction_loss + commitment_loss + perceptual_loss


