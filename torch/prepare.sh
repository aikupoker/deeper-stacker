#/usr/bin/env bash

# install guide for docker image:
#   nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# update apt source
apt-get update

# install some tools we need
apt-get install -y sudo git gcc g++ make vim

# get this repo
mkdir work && cd work
git clone https://github.com/aikupoker/deeper-stack.git ./deeper-stack

# unzip the hand rank file
cd deeper-stack/Source/Game/Evaluation
unzip HandRanks.zip
cd ../../../..

# install the latest CMake
git clone https://github.com/Kitware/CMake.git
cd CMake
./bootstrap && make && make install
cd ..

# get torch (for CUDA 10, CuDNN 7)
git clone https://github.com/torch/distro.git ./torch --recursive
cd ./torch && bash install-deps
rm -fr cmake/3.6/Modules/FindCUDA*

# add some patches
cp ../deeper-stack/torch/extra/cutorch/* ./extra/cutorch
cp ../deeper-stack/torch/pkg/torch/* ./pkg/torch
cd extra/cutorch
patch -p1 < automic.patch
cd ../..

# start build torch
export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"
TORCH_LUA_VERSION=LUA52 ./install.sh
source $HOME/.bashrc

# install some packages we need
luarocks install luasocket
luarocks install graphviz
