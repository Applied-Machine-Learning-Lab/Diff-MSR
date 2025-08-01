import numpy as np
import pandas as pd
import torch.utils.data


class Douban(torch.utils.data.Dataset):
    """
    Douban Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    Reference:
        https://github.com/FengZhu-Joey/GA-DTCDR/tree/main/Data
    """

    def __init__(self, dataset_path='e:/code/Douban/Data/', sep='\t', engine='c', header=None):
        
        dataset_path1 = dataset_path + 'douban_music/ratings.dat'
        dataset_path2 = dataset_path + 'douban_book/ratings.dat'
        dataset_path3 = dataset_path + 'douban_movie/ratings.dat'
        
        data1 = pd.read_csv(dataset_path1,sep='\t',header=None).to_numpy()[:, :3]
        data2 = pd.read_csv(dataset_path2,sep='\t',header=None).to_numpy()[:, :3]
        data3 = pd.read_csv(dataset_path3,sep='\t',header=None).to_numpy()[:, :3]
        
        self.items1 = data1[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        self.items2 = data2[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        self.items3 = data3[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        self.items2[:,1] = self.items2[:,1] + 5567
        self.items3[:,1] = self.items3[:,1] + 5567 + 6777
        self.items = np.concatenate((self.items1, self.items2, self.items3), axis=0)
        
        self.targets1 = self.__preprocess_target(data1[:, 2]).astype(np.int)
        self.targets2 = self.__preprocess_target(data2[:, 2]).astype(np.int)
        self.targets3 = self.__preprocess_target(data3[:, 2]).astype(np.int)
        self.targets = np.concatenate((self.targets1, self.targets2, self.targets3), axis=0)
        
        self.field_dims = np.ndarray(shape=(3,), dtype=int)
        self.field_dims[0] = 2718
        self.field_dims[1] = 5567 + 6777 + 9565
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target


class DoubanMusic(torch.utils.data.Dataset):

    def __init__(self, dataset_path='/mnt/ufs18/home-135/zhaoxi35/WangYuhao/Douban/Data/douban_music/ratings.dat', sep='\t', engine='c', header=None):
        data = pd.read_csv(dataset_path,sep=sep,header=header).to_numpy()[:, :3]
        
        self.items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        domain_id = np.full((self.items.shape[0],1),0)
        self.items = np.concatenate([domain_id,self.items],1)
        
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.int)
        self.field_dims = np.ndarray(shape=(3,), dtype=int)
        #self.field_dims[0] = np.max(self.items[:,0])+1
        #self.field_dims[1] = np.max(self.items[:,1])+1
        self.field_dims[0] = 3
        self.field_dims[1] = 2718
        self.field_dims[2] = 5567 + 6777 + 9565
        #print(self.field_dims)
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)
        self.items[:,1] = self.items[:,1] + 0

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target
        
class DoubanMusic_sparse(torch.utils.data.Dataset):

    def __init__(self, dataset_path='/gpfs1/scratch/yhwang25/dataset/Douban/Data/douban_music/douban_music_sparse.csv', sep='\t', engine='c', header=None):
        #torch.manual_seed(0)
        #random.seed(0)
        
        
        data = pd.read_csv(dataset_path).to_numpy()[:, :4]
        '''
        #indexx = random.sample(range(data.shape[0]), 6000)
        indexx = torch.randint(0, data.shape[0], (6000,1),generator=torch.Generator().manual_seed(5))
        indexx = indexx.view(6000)
        data = data[list(indexx),:]
        '''
        
        self.items = data[:, :3].astype(np.int) # -1 because ID begins from 1
        self.items[:,1]-=1
        self.items[:,2]-=1
        self.targets = data[:, 3].astype(np.int)
        self.field_dims = np.ndarray(shape=(3,), dtype=int)
        #self.field_dims[0] = np.max(self.items[:,0])+1
        #self.field_dims[1] = np.max(self.items[:,1])+1
        self.field_dims[0] = 3
        self.field_dims[1] = 2718
        self.field_dims[2] = 5567 + 6777 + 9565
        #print(self.field_dims)
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)
        self.items[:,1] = self.items[:,1] + 0

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

class DoubanBook(torch.utils.data.Dataset):

    def __init__(self, dataset_path='/mnt/ufs18/home-135/zhaoxi35/WangYuhao/Douban/Data/douban_book/ratings.dat', sep='\t', engine='c', header=None):
        data = pd.read_csv(dataset_path,sep=sep,header=header).to_numpy()[:, :3]
        self.items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        domain_id = np.full((self.items.shape[0],1),1)
        self.items = np.concatenate([domain_id,self.items],1)
        
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.int)
        self.field_dims = np.ndarray(shape=(3,), dtype=int)
        #self.field_dims[0] = np.max(self.items[:,0])+1
        #self.field_dims[1] = np.max(self.items[:,1])+1
        self.field_dims[0] = 3
        self.field_dims[1] = 2718
        self.field_dims[2] = 5567 + 6777 + 9565
        
        #print(self.field_dims)
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)
        self.items[:,1] = self.items[:,1] + 5567
        
    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target

class DoubanMovie(torch.utils.data.Dataset):

    def __init__(self, dataset_path='/mnt/ufs18/home-135/zhaoxi35/WangYuhao/Douban/Data/douban_movie/ratings.dat', sep='\t', engine='c', header=None):
        data = pd.read_csv(dataset_path,sep=sep,header=header).to_numpy()[:, :3]
        self.items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        domain_id = np.full((self.items.shape[0],1),2)
        self.items = np.concatenate([domain_id,self.items],1)
        
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.int)
        self.field_dims = np.ndarray(shape=(3,), dtype=int)
        #self.field_dims[0] = 2712
        #self.field_dims[1] = 34893
        #self.field_dims[0] = np.max(self.items[:,0])+1
        #self.field_dims[1] = np.max(self.items[:,1])+1
        self.field_dims[0] = 3
        self.field_dims[1] = 2718
        self.field_dims[2] = 5567 + 6777 + 9565
        #print(self.field_dims)
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)
        self.items[:,1] = self.items[:,1] + 5567 + 6777

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target