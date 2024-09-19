from pathlib import Path

import torch
from PIL.Image import Image
from torch.nn import Module, Linear
from torchvision.models import resnet18
from torchvision.transforms.v2 import Compose, PILToTensor, Normalize, CenterCrop, Resize


class FlowersResnetWrapper(Module):

    def __init__(self, checkpoint_path: str | Path | None = None, *args, **kwargs):
        """
        Initializes the custom ResNet18 wrapper for inference on the Flower102 dataset.

        Parameters:
        -----------
        checkpoint_path : str | Path | None, optional
            Path to a checkpoint file containing the finetuned model weights. If None, the model is initialized without loading weights.
            If the provided path is invalid, the model will use random initialization.
        *args, **kwargs :
            Additional arguments and keyword arguments to pass to the parent class initializer.

        Behavior:
        ---------
        - Loads a pre-finetuned ResNet18 model, modifying the final fully connected layer to output 102 classes for the Flower102 dataset.
        - Loads a checkpoint if a valid path is provided; otherwise, the model is initialized randomly.
        - Freezes all model parameters for inference and sets the model to evaluation mode.
        - Moves the model to GPU if available.
        - Loads class labels for the Flower102 dataset from a text file.
        - Sets the mean and standard deviation values for input normalization based on ImageNet statistics.

        Returns:
        --------
        None
        """
        super().__init__(*args, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resnet = resnet18()
        self.resnet.fc = Linear(self.resnet.fc.in_features, 102)
        if checkpoint_path is not None:
            if isinstance(checkpoint_path, str):
                checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                print(f'Provided checkpoint {checkpoint_path} does not exist. ResNet will be randomly initialized.')
            else:
                self.resnet.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=self.device))
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.eval()
        self.resnet = self.resnet.to(device=self.device)
        with open('./models/labels/labels.txt', 'r') as f:
            labeltext = f.read()
            f.close()
        self.labels = [l.replace('\'', '').strip() for l in labeltext.split('\n')]
        self.mean = torch.Tensor([0.485, 0.456, 0.406])
        self.std = torch.Tensor([0.229, 0.224, 0.225])

    def preprocess(self, x: Image|torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the input image or tensor for inference on the Flower102 dataset using the ResNet18 model.

        Parameters:
        -----------
        x : Image | torch.Tensor
            Input image as a PIL Image or a torch Tensor. If a PIL Image is provided, it will be converted to a Tensor.

        Behavior:
        ---------
        - Converts a PIL Image to a torch Tensor if necessary.
        - Ensures the input has a batch dimension (adds one if missing).
        - Normalizes the pixel values by dividing by 255.
        - Applies normalization using the mean and standard deviation specific to the dataset (based on ImageNet stats).
        - Resizes the image to 512x512 and performs a center crop to 256x256.

        Returns:
        --------
        torch.Tensor
            A preprocessed torch Tensor ready for input into the model.
        """
        needed_transforms = []
        if isinstance(x, Image):
            needed_transforms.append(PILToTensor())
        needed_transforms.append(lambda i: i[None, :, :, :] if len(i.shape) == 3 else i)
        needed_transforms.append(lambda i: i / 255)
        needed_transforms.append(Normalize(self.mean, self.std))
        needed_transforms.append(Resize((512, 512)))
        needed_transforms.append(CenterCrop((256, 256)))
        transforms = Compose(needed_transforms)
        return transforms(x)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Performs a forward pass through the ResNet18 model and returns the logits, predicted class IDs, and class names.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor representing a batch of images, expected to be preprocessed and of shape [batch_size, channels, height, width].

        Behavior:
        ---------
        - Moves the input tensor to the appropriate device (CPU or GPU).
        - Passes the input through the ResNet18 model to obtain logits.
        - Computes the predicted class IDs by applying softmax to the logits and taking the class with the highest probability.
        - Maps the predicted class IDs to corresponding class names using the loaded Flower102 labels.

        Returns:
        --------
        dict
            A dictionary containing:
            - 'logits': The raw output from the model (logits).
            - 'class_ids': The predicted class IDs for each input in the batch.
            - 'class_names': The predicted class names corresponding to the class IDs.
        """
        x = x.to(device=self.device)
        logits = self.resnet(x)
        out = {'logits': logits, 'class_ids': torch.argmax(torch.softmax(logits, dim=-1), dim=-1)}
        out['class_names'] = [self.labels[i] for i in out['class_ids']]
        return out
