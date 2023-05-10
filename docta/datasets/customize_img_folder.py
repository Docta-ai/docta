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
        self.transform = transform
        for i in range(len(self.targets_all)):
            tmp_feature, _ = self.__getitem__(i)
            feature_all.append(np.asarray(tmp_feature))
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