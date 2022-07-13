# gan-resnet-cat
## Purpose
This is to test whether I can generate some cat pictures based on GAN + resnet

## Raw data
The raw cat photos are from [Marek Zyla](https://github.com/zylamarek/cat-dataset), which is in turn modifeid from 

[Weiwei Zhang, Jian Sun, and Xiaoou Tang, "Cat Head Detection - How to Effectively Exploit Shape and Texture Features", Proc. of European Conf. Computer Vision, vol. 4, pp.802-816, 2008](https://www.microsoft.com/en-us/research/wp-content/uploads/2008/10/ECCV_CAT_PROC.pdf)

I did some cropping around the head, which is defined by the ear and mouth landmarks, to make the face larger. The code to do the cropping and the cropped dataset are in cropped.dir

## Sanity checks
I first check whether I can get some reasonable MNIST pictures. This is to make sure I understand the many loss functions in GAN. The model is standard DCGAN. This part is inspired by [Emilien Dupont](https://github.com/EmilienDupont/wgan-gp) and I have added the SNGAN models. To make the first few epoches converge faster, I also experimented with add "kick", which simply adds the standard deviations in the loss function. It helps a little in the MNIST dataset, but not much in the cat dataset.

This is a gif without "kick", from left to right is GAN, WGAN-GP, SNGAN, SNGAN with hinge loss.
![Results without "kick"](./pics/nokick.gif)

Same as above, but with "kick", seems it helps to converge faster for the MNIST dataset.
![Reesults with "kick"](./pics/kick.gif)
