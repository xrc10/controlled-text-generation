# Required packages
# pip install torchtext
# conda install -c conda-forge spacy;
# python -m spacy download en

# This script runs the example from the original README
# export CUDA_VISIBLE_DEVICES=1;

# This will create vae.bin. Essentially this is the base VAE as in Bowman, 2015 [2].
# python train_vae.py --save --gpu;

# This will create `ctextgen.bin`. The discriminator is using Kim, 2014 [3] architecture and the training procedure is as in Hu, 2017 [1].
python train_discriminator.py --save --gpu

# basic evaluations, e.g. conditional generation and latent interpolation.
# python test.py --model ctextgen --gpu
