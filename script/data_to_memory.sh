#!/usr/bin/env bash
if [ "$1" == "imagenet" ]; then
mount -t tmpfs -o size=160G tmpfs /userhome/temp_data
mkdir /userhome/temp_data/ImageNet2012
mkdir /userhome/temp_data/ImageNet2012/train
tar xvf /gdata/ImageNet2012/ILSVRC2012_img_train.tar -C /userhome/temp_data/ImageNet2012/train
cp /userhome/project/pytorch-cls/script/unzip_imagenet.sh /userhome/temp_data/ImageNet2012/train/
cd /userhome/temp_data/ImageNet2012/train/
./unzip_imagenet.sh
cp -r /gdata/ImageNet2012/val/ /userhome/temp_data/ImageNet2012/val
elif [ "$1" == 'cifar10' ];
then
mount -t tmpfs -o size=1G tmpfs /userhome/temp_data
cp -r /gdata/cifar10/ userhome/temp_data/
elif [ "$1" == 'cifar100' ];
then
mount -t tmpfs -o size=1G tmpfs /userhome/temp_data
cp -r /gdata/cifar100 /userhome/temp_data/
elif [ "$1" == 'fashionmnist' ];
then
mount -t tmpfs -o size=1G tmpfs /userhome/temp_data
cp -r /userhome/data/fashionmnist /userhome/temp_data/
fi