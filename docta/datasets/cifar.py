from torchvision.datasets import CIFAR10
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from .data_utils import load_label, noisify_general, get_T_true_from_data



class Cifar10_noisy(CIFAR10):
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, cfg, train = True, preprocess = None) -> None:
        if preprocess is None:
            preprocess = self.train_transform if train else self.test_transform
        super(Cifar10_noisy, self).__init__(cfg.data_root, train=train, transform=preprocess,
                                      target_transform=None, download=True)
        self.cfg = cfg
        if train:
            self.load_label()
        else:
            self.label = self.targets
        self.feature = self.data

    
    def load_label(self):
        if self.cfg.label_path is None:
            self.noisy_label, actual_noise = noisify_general(self.targets, self.cfg.noise_rate, random_state = self.cfg.seed)
        else:
            self.noisy_label = load_label(label_path = self.cfg.label_path, clean_label = self.targets, key = self.cfg.noisy_label_key, clean_key = self.cfg.clean_label_key)
        self.T_true = get_T_true_from_data(clean_label = self.targets, noisy_label = self.noisy_label)
        print(f'True T is \n{np.round(self.T_true * 100, 1 )}')
        self.label = np.stack((self.targets, self.noisy_label)).transpose()


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, (target, noisy_label), index). 
            target: clean label
            noisy_label: loaded/synthesized noisy label
        """
        img, label = self.data[index], self.label[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index


class Cifar10N(Cifar10_noisy):
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, cfg, train=True, preprocess=None, noisy_label_key=None, clean_label_key=None) -> None:
        print(f"CIFAR-10N includes five sets of noisy labels: random_label1, random_label2, random_label3, aggre_label, worse_label.\nPlease set cfg.noisy_label_key to one of them.")
        if preprocess is None:
            preprocess = self.train_transform if train else self.test_transform
        self.cfg = cfg
        self.cfg.label_path = "./data/cifar/CIFAR-10_human.pt"
        if noisy_label_key is not None:
            self.cfg.noisy_label_key = noisy_label_key
        if clean_label_key is not None:
            self.cfg.clean_label_key = clean_label_key
        super(Cifar10N, self).__init__(cfg, train, preprocess)

class Cifar10_clean(CIFAR10):
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, cfg, train = True, preprocess = None) -> None:
        if preprocess is None:
            preprocess = self.train_transform if train else self.test_transform
        super(Cifar10_clean, self).__init__(cfg.data_root, train=train, transform=preprocess,
                                      target_transform=None, download=True)
        self.cfg = cfg
        self.label = self.targets
        self.feature = self.data

    

class Cifar100_noisy(Cifar10_noisy):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])


class Cifar100_clean(Cifar10_clean):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])


class Cifar100N(Cifar100_noisy):

    def __init__(self, cfg, train=True, preprocess=None) -> None:
        if preprocess is None:
            preprocess = self.train_transform if train else self.test_transform
        self.cfg = cfg
        self.cfg.label_path = "./data/cifar/CIFAR-100_human.pt"
        self.cfg.noisy_label_key = "noisy_label"
        self.cfg.clean_label_key = "clean_label"
        super(Cifar100N, self).__init__(cfg, train, preprocess)