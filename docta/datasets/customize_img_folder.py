from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
class Customize_Image_Folder(datasets.ImageFolder):

    def __init__(self, root, transform, target_transform=None):
        super(Customize_Image_Folder, self).__init__(root, transform)
        self.root = root
        self.data = np.array(self.samples)
        self.path_all = self.data[:, 0].tolist()
        self.targets_all = self.data[:,1].astype(int)
        self.target_transform = target_transform
        # Get features
        feature_all = []
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ])
        else:
            self.transform = transform
        for i in range(len(self.targets_all)):
            tmp_feature, _ = self.__getitem__(i)
            feature_all.append(tmp_feature.permute(1, 2, 0).numpy().astype(np.uint8))  # require that the last dimension is the # of channels
        self.feature = np.array(feature_all)
        # Get labels
        self.label = self.targets_all
        
    def __getitem__(self, index):
        path = self.path_all[index]
        target = self.targets_all[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.path_all)