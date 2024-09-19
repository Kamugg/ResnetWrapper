# ResnetWrapper

This is a wrapper I built for Resnet18 fine-tuned on the Flowers102 dataset.

## The dataset

[Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) is a dataset of flowers provided by the Oxford University. The dataset has varying sized images belonging to one of 102 classes. The training and validation split have 1020 images, while the test set has 6149.

## Contents

The repository is comprised of:

1. **Training.ipynb**: A notebook that shows the transfer learning process.
2. **FlowersResnetWrapper.py**: The wrapper class for resnet18.
3. **resnet.pth**: The pythorch checkpoint to usethe wrapper.
4. **labels.txt**: The list of human-readable classes needed for the wrapper to work.

## The wrapper:

### Initialize:

To initialize the wrapper you just need to provide the checkpoint's path, either via string or Path object.

### Feed images:

The wrapper accepts either PIL images or tensors, if a tensor is provided, the model expects a tensor of shapes either $B \times 3 \times H \times W$ or $3 \times H \times W$ of values in the range $[0, 255]$. You can call the preprocess() method to make the input digestible for resnet. Once the input is processed, you can call the wrapper forward method to obtain your predictions.

### Predictions

The predictions are provided through a dictionary with 3 keys:

1. **logits**: The logits of the model
2. **class_ids**: The class id of each prediction, computed as $argmax(softmax(predictions))$
3. **class_names**: The human-readable name of the class id of each prediction

## Requirements (wrapper only)

- **torch**==2.4.1+cu124
- **torchvision**==0.19.1+cu124
- **pillow**==10.4.0
