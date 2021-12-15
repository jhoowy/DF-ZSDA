#!/bin/sh

echo "Generating MNIST dataset..."
python data_gen/dataset_gen.py --name MNIST

echo "Generating FashionMNIST datset..."
python data_gen/dataset_gen.py --name FashionMNIST

echo "Generating EMNIST datset..."
python data_gen/dataset_gen.py --name EMNIST
