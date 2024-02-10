# IRIS 

Here we want to reproduce the [IRIS paper](https://arxiv.org/pdf/2209.00588.pdf).

We have to build three components: 
* experience collection 
* world model (autoencoder, vqvae with an additional perceptual loss) 
* behavior model 

## World Model 

* first part is basically a VQVAE with an additional perceptual loss  
* loss = reconstruction_loss + commitment_loss + perceptual_loss
* second part is a transformer that will be able to image future observations, rewards, etc.
* transformer input {z01, z02, z03, a0} -> next z, reward r, termination t 

## Behavior Model 

* actor-critic like DreamerV2

## References 

Code is based/stolen from these:

- [original implementation](https://github.com/eloialonso/iris)
- [minGPT](https://github.com/karpathy/minGPT)
- [VQVAE](https://github.com/ritheshkumar95/pytorch-vqvae/tree/master)
- [VQGAN](https://github.com/CompVis/taming-transformers)
- [perceptual loss](https://github.com/richzhang/PerceptualSimilarity/tree/master)