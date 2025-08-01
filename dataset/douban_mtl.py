import numpy as np
import pandas as pd
import torch.utils.data
'''
class Douban_mtl(torch.utils.data.Dataset):

    def __init__(self, mode, path_base='/gpfs1/scratch/yhwang25/dataset/Douban/Data/'):
        #mode: train / val / test
        dataset_name = ['douban_music','douban_book','douban_movie']
        
        path = path_base + mode + '/' + dataset_name[0] + '_' + mode + '.csv'
        data = pd.read_csv(path).to_numpy()[:, :4]
        
        for i in range(1,len(dataset_name)):
            path = path_base + mode + '/' + dataset_name[i] + '_' + mode + '.csv'
            data_on = pd.read_csv(path).to_numpy()[:, :4]
            data = np.concatenate((data,data_on),0)
        
        self.items = data[:, :3].astype(np.int)
        self.targets = data[:,3].astype(np.int)
        self.field_dims = np.ndarray(shape=(3,), dtype=int)

        self.field_dims[0] = 3
        self.field_dims[1] = 2718
        self.field_dims[2] = 5567 + 6777 + 9565
        
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]
'''
class Douban_mtl(torch.utils.data.Dataset):

    def __init__(self, mode, path_base='/gpfs1/scratch/yhwang25/dataset/Douban/Data/'):
        #mode: train / val / test
        dataset_name = ['douban_music','douban_book','douban_movie']
        
        path = path_base + mode + '/' + dataset_name[0] + '_sparse_' + mode + '.csv'
        data = pd.read_csv(path).to_numpy()[:, :4]
        
        for i in range(1,len(dataset_name)):
            path = path_base + mode + '/' + dataset_name[i] + '_' + mode + '.csv'
            data_on = pd.read_csv(path).to_numpy()[:, :4]
            data = np.concatenate((data,data_on),0)
        
        self.items = data[:, :3].astype(np.int)
        self.targets = data[:,3].astype(np.int)
        self.field_dims = np.ndarray(shape=(3,), dtype=int)

        self.field_dims[0] = 3
        self.field_dims[1] = 2718
        self.field_dims[2] = 5567 + 6777 + 9565
        
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]
                
class Douban_mtl_2(torch.utils.data.Dataset):

    def __init__(self, mode, indexx=0, path_base='/gpfs1/scratch/yhwang25/dataset/Douban/Data/'):
        #mode: train / val / test
        dataset_name = ['douban_music','douban_book','douban_movie']
        del dataset_name[indexx]
        
        path = path_base + mode + '/' + dataset_name[0] + '_' + mode + '.csv'
        data = pd.read_csv(path).to_numpy()[:, :4]
        
        for i in range(1,len(dataset_name)):
            path = path_base + mode + '/' + dataset_name[i] + '_' + mode + '.csv'
            data_on = pd.read_csv(path).to_numpy()[:, :4]
            data = np.concatenate((data,data_on),0)
        
        self.items = data[:, :3].astype(np.int)
        self.targets = data[:,3].astype(np.int)
        self.field_dims = np.ndarray(shape=(3,), dtype=int)

        self.field_dims[0] = 3
        self.field_dims[1] = 2718
        self.field_dims[2] = 5567 + 6777 + 9565
        
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]


        
class Douban_mtl_classifier(torch.utils.data.Dataset):

    def __init__(self, mode, path_base='/gpfs1/scratch/yhwang25/dataset/Douban/Data/'):
        #mode: train / val / test
        dataset_name = ['douban_music','douban_book','douban_movie']
        
        path = path_base + mode + '/' + dataset_name[0] + '_' + mode + '.csv'
        data = pd.read_csv(path).to_numpy()[:, :4]
        
        for i in range(1,len(dataset_name)):
            path = path_base + mode + '/' + dataset_name[i] + '_' + mode + '.csv'
            data_on = pd.read_csv(path).to_numpy()[:, :4]
            data = np.concatenate((data,data_on),0)
        
        self.items = data[:, :3].astype(np.int)
        self.targets = data[:,0].astype(np.int)
        self.field_dims = np.ndarray(shape=(3,), dtype=int)

        self.field_dims[0] = 3
        self.field_dims[1] = 2718
        self.field_dims[2] = 5567 + 6777 + 9565
        
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]
        
class Douban_mtl_classifier_binary(torch.utils.data.Dataset):

    def __init__(self, mode, path_base='/gpfs1/scratch/yhwang25/dataset/Douban/Data/'):
        #mode: train / val / test
        dataset_name = ['douban_music','douban_book','douban_movie']
        
        #path = path_base + mode + '/' + dataset_name[0] + '_' + mode + '.csv'
        path = path_base + mode + '/' + dataset_name[0] + '_sparse_' + mode + '.csv'
        data = pd.read_csv(path).to_numpy()[:, :4]
        
        for i in range(1,len(dataset_name)):
            path = path_base + mode + '/' + dataset_name[i] + '_' + mode + '.csv'
            data_on = pd.read_csv(path).to_numpy()[:, :4]
            data = np.concatenate((data,data_on),0)
        
        self.items = data[:, :3].astype(np.int)
        self.targets = data[:,0].astype(np.int)
        self.targets = self.__preprocess_target(self.targets).astype(np.int)
        
        self.field_dims = np.ndarray(shape=(3,), dtype=int)

        self.field_dims[0] = 3
        self.field_dims[1] = 2718
        self.field_dims[2] = 5567 + 6777 + 9565
        
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]
    
    def __preprocess_target(self, target):
        #target[target <= 3] = 0
        target[target >= 1] = 1
        return target