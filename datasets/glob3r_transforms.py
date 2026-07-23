"""Image augmentation used by Glob3R Appendix B."""

import torchvision.transforms as transforms


class Glob3RImageTransform:
    """Color jitter, Gaussian blur, and random grayscale followed by ToTensor."""

    def __init__(self) -> None:
        # Torchvision draws from the worker-seeded torch RNG, so distributed
        # dataloader workers do not replay an identical NumPy augmentation stream.
        self.transform = transforms.Compose(
            [
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.1),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
