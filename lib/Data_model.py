
import numpy as np
from itertools import combinations
import collections 
from glob import glob


class Data_model(object):
    def __init__(self, dataset, model, resolution, data_dir, connection_type, num_agent, distribution, imbalance):
        self.dataset = dataset
        self.model = model
        self.resolution = resolution
        self.data_dir = data_dir
        self.connection_type = connection_type
        self.num_agent = num_agent
        self.distribution = distribution
        self.imbalance = imbalance
        self.dist_list = None
        self.trainx = None
        self.trainy = None

    def dataset_and_model(self):
        
        # Loading datasets and importing appropriate networks
        #============================================================================== 
        if self.dataset == 'mnist':
            from keras.datasets import mnist
            from models.model_mnist import generator,discriminator
            (trainx, trainy), (testx, testy) = mnist.load_data()
            trainx = np.reshape(trainx,[-1,28,28,1])/127.5 - 1
            #testx = np.reshape(testx,[-1,28,28,1])/127.5 - 1
            
        elif self.dataset == 'mnist_fashion':
            from keras.datasets import fashion_mnist
            from models.model_mnist import generator,discriminator
            (trainx, trainy), (testx, testy) = fashion_mnist.load_data()
            trainx = np.reshape(trainx,[-1,28,28,1])/127.5 - 1
            #testx = np.reshape(testx,[-1,28,28,1])/127.5 - 1
            
        elif self.dataset == 'cifar10':
            from keras.datasets import cifar10
            if self.model == 'dcgan':
                from models.model_32_dcgan import generator,discriminator
            elif self.model == 'resnet':
                from models.model_32_resnet import generator,discriminator
            (trainx, trainy), (testx, testy) = cifar10.load_data()
            trainx = trainx/127.5 - 1
            #testx = testx/127.5 - 1
            
        elif (self.dataset in ['omniglot','omniglot_CG','omniglot_CL','omniglot_GL']) and self.resolution==28:
            from lib.utils import read_omniglot
            from models.model_mnist import generator,discriminator
            if self.resolution == 28:
                convert = 'grayscale'
            else:
                convert = 'RGB'
            trainx, trainy = read_omniglot(self.data_dir + 'omniglot-master/python/images_background',self.resolution, convert)
            if self.resolution == 28:
                trainx = np.expand_dims(trainx,axis=3)/127.5 - 1
            else:
                trainx = trainx/127.5 - 1
            
        elif self.dataset in ['celeba','ffhq']:# and args.resolution==64:
            if self.model == 'dcgan':
                from models.model_64_dcgan import generator,discriminator
            elif self.model == 'resnet':
                from models.model_64_resnet import generator,discriminator
            trainx = ['celebA_males_64', 'celebA_females_64']
            trainy = None
            
        elif self.dataset in ['celeba_ffhq']:# and args.resolution==64:
            if self.model == 'dcgan':
                from models.model_64_dcgan import generator,discriminator
            elif self.model == 'resnet':
                from models.model_64_resnet import generator,discriminator
            trainx = ['celebA_males_64', 'celebA_females_64']
            trainy = None
        
        self.trainx = trainx
        self.trainy = trainy
        return trainx, generator, discriminator
    
    
    def cmap_and_dlist(self):
        
        # connetion map
        #==============================================================================
        if self.connection_type == 'v1' and self.num_agent == 2:
            connection_map = [[0,2],
                              [1,2]]
        elif self.connection_type == 'v1' and  self.num_agent == 3:
            connection_map = [[0,3,5,6],
                              [1,4,5,6],
                              [2,3,4,6]]
        elif self.connection_type == 'v3' and self.num_agent == 2:
            connection_map = [[0,2],
                              [  2]]
        elif self.connection_type == 'v3' and self.num_agent == 3:
            connection_map = [[0,5,6],
                              [  5,6],
                              [    6]]
        else:
            raise NotImplementedError
            
        relevant_idx = list(set([item for sublist in connection_map for item in sublist]))
        
        # gen custom ditstibutions
        #==============================================================================+
        if self.dataset =='omniglot':
            dist_list = [[0],
                         [1],
                         [2]]
            cc = []
        elif self.dataset == 'omniglot_CG':
            dist_list = [[0],
                         [1]]
            cc = []
        elif self.dataset == 'omniglot_CL':
            dist_list = [[0],
                         [2]]
            cc = []
        elif self.dataset == 'omniglot_GL':
            dist_list = [[1],
                         [2]]
            cc = []
        elif self.dataset in ['comnist','cifar10and100']:
            dist_list = [[0],
                         [1]]
            cc = []
        elif self.distribution =='big' and self.num_agent == 3 and self.connection_type == 'v1':
            dist_list = []
            cc = []
        elif self.distribution =='small' and self.num_agent == 3 and self.connection_type == 'v1':
            dist_list = [[0,3,5,6],
                         [1,4,5,6],
                         [2,3,4,6]]
            cc = [[0],[1],[2],[3],[4],[5],[6]]
        elif self.distribution =='small' and self.num_agent == 2 and self.connection_type == 'v1':
            dist_list = [[0,2],
                         [1,2]]
            cc = [[0],[1],[2]]
        elif self.distribution =='big' and self.num_agent == 2 and self.connection_type == 'v1':
            dist_list = [[0,1,2,3,4,5,6],
                         [3,4,5,6,7,8,9]]
            cc = [[0,1,2],[7,8,9],[3,4,5,6]]
        elif self.distribution =='small' and self.num_agent == 2 and self.connection_type == 'v3': 
            dist_list = [[0,1],
                         [  1]]
            cc = [[0],[1]]
        elif self.distribution =='big' and self.num_agent == 2 and self.connection_type == 'v3': 
            dist_list = [[0,1,2,3,4,5,6,7,8,9],
                         [          5,6,7,8,9]]
            cc = [[0,1,2,3,4],[5,6,7,8,9]]
        elif self.distribution =='small' and self.num_agent == 3 and self.connection_type == 'v3': 
            dist_list = [[0,1,2],
                         [  1,2],
                         [    2]]
            cc = [[0],[1],[2]]
        elif self.distribution =='big' and self.num_agent == 3 and self.connection_type == 'v3': 
            dist_list = [[0,1,2,3,4,5,6,7,8,9],
                         [      3,4,5,6,7,8,9],
                         [            6,7,8,9]] 
            cc = [[0,1,2],[3,4,5],[6,7,8,9]]
        else:
            raise NotImplementedError
    
        self.dist_list = dist_list
        
        return connection_map, dist_list, cc, relevant_idx 
    
    
    def process_distribution(self):
        
        # not a generic code, aims 3 agents games 
        #==============================================================================    
        if self.connection_type == 'v1' and self.num_agent == 3 and self.dataset in ['mnist', 'mnist_fashion','cifar10']:
            trainx_list = []
            trainy_list = []  
            if self.imbalance == 'type1':
                b_list = [500,2000,6000] # ==>[500,1000,2000]
            elif self.imbalance == 'type2':
                b_list = [6000,6000,6000] # ==>[6000,3000,2000]
            else:
                b_list = [2000,4000,6000] # ==>[2000,2000,2000]
            
            for digit in list(set([item for sublist in self.dist_list for item in sublist])):
                if digit in [0,1,2]:
                    trainx_list.append(self.trainx[np.squeeze(self.trainy==digit)][:b_list[0]])
                    trainy_list.append(self.trainy[np.squeeze(self.trainy==digit)][:b_list[0]])
                elif digit in [3,4,5]:
                    trainx_list.append(self.trainx[np.squeeze(self.trainy==digit)][:b_list[1]])
                    trainy_list.append(self.trainy[np.squeeze(self.trainy==digit)][:b_list[1]])
                else:
                    trainx_list.append(self.trainx[np.squeeze(self.trainy==digit)][:b_list[2]])
                    trainy_list.append(self.trainy[np.squeeze(self.trainy==digit)][:b_list[2]])
            self.trainx = np.concatenate(trainx_list,axis=0)
            self.trainy = np.concatenate(trainy_list,axis=0)
    
    
        #==============================================================================
        count_overlap = collections.defaultdict(int) # hashmap to count overlaps
        count_occurence = collections.defaultdict(int) # hasmap to update where to select subset of current digit
        
        # constructing hashmap digit => occurence
        for set_ in self.dist_list: 
            for digit in set_:
                count_overlap[digit] += 1
                    
        distrib=[]   
        if self.dataset in ['celeba','stanford_kaggle']:
            for i in range(self.num_agent):
                distrib.append(glob(self.data_dir+ self.trainx[i] +'/*.jpg'))
                
        if self.dataset in ['celeba_ffhq']:
            assert self.num_agent==2
            distrib.append(glob(self.data_dir+ self.trainx[0] +'/*.jpg') + glob(self.data_dir+ self.trainx[1] +'/*.jpg'))
            distrib.append(glob(self.data_dir +'thumbnails64x64/*.png'))
                
        elif self.dataset in ['mnist', 'mnist_fashion','cifar10']:
            for i in range(self.num_agent):
                l = []        
                for digit in self.dist_list[i]:
                    n_digit = np.sum(self.trainy==digit) # count elements
        
                    start=count_occurence[digit]*n_digit//count_overlap[digit]
                    end =(count_occurence[digit]+1)*n_digit//count_overlap[digit]
        
                    count_occurence[digit] += 1 # 
        
                    l.append(self.trainx[np.squeeze(self.trainy==digit)][start:end])
                trainx1=np.vstack(l)
                distrib.append(trainx1[np.random.permutation(trainx1.shape[0])])
        
        else: # for the datasets which are not contrived
            for i in range(self.num_agent):
                l = []
                for digit in self.dist_list[i]:
                    l.append(self.trainx[np.squeeze(self.trainy==digit)])
                trainx1=np.vstack(l)
                distrib.append(trainx1[np.random.permutation(trainx1.shape[0])])

        return distrib
        
'''        
    elif dataset == 'cifar10and100':
        from keras.datasets import cifar10, cifar100
        if model == 'dcgan':
            from models.model_32_dcgan import generator,discriminator
        elif model == 'resnet':
            from models.model_32_resnet import generator,discriminator
        (trainx, trainy), (testx, testy) = cifar10.load_data()
        trainx_10 = trainx[:1000]/127.5 - 1
        trainy_10 = np.ones((trainx_10.shape[0],1),'int32')
        (trainx, trainy), (testx, testy) = cifar100.load_data()
        trainx_100 = trainx[:50000]/127.5 - 1
        trainy_100 = np.zeros((trainx_100.shape[0],1),'int32')
        trainx = np.concatenate([trainx_100,trainx_10],axis=0)
        trainy = np.concatenate([trainy_100,trainy_10],axis=0)
        del trainx_100, trainx_10, trainy_100, trainy_10
        basename_stats = ['fid_stats_cifar100_tf_train.npz', 'fid_stats_cifar10_tf_train.npz']
        dist_stats_path = []
        dist_stats_path.append(data_dir + basename_stats[0])
        dist_stats_path.append(data_dir + basename_stats[1])
        
        
    elif dataset == 'comnist' and resolution==28:
        from lib.utils import read_comnist
        #from models.model_32_dcgan import generator,discriminator
        from models.model_mnist import generator,discriminator
        if resolution == 28:
            convert = 'grayscale'
        else:
            convert = 'RGB'
        trainx, trainy = read_comnist(data_dir + 'CoMNIST-master/images',resolution,convert)
        if resolution == 28:
            trainx = np.expand_dims(trainx,axis=3)/127.5 - 1
        else:
            trainx = trainx/127.5 - 1 
            
    elif dataset == 'stanford_kaggle':# and args.resolution==64:
        if model == 'dcgan':
            from models.model_64_dcgan import generator,discriminator
        elif model == 'resnet':
            from models.model_64_resnet import generator,discriminator
        basename_data = ['stanford_dogs_64', 'dogs_cats_64_kaggle']
        basename_stats = ['fid_stats_stanford_dogs_64_tf.npz', 'fid_stats_dogs_cats_64_kaggle_tf.npz']
        dist_stats_path = []
        dist_stats_path.append(data_dir + basename_stats[0])
        dist_stats_path.append(data_dir + basename_stats[1])
        #inception_path = fid.check_or_download_inception(data_dir + 'imagenet_model')
        
        elif self.distribution =='small_no' and self.num_agent == 3 and self.connection_type == 'v1':
            dist_list = [[0,3,5],
                         [1,4,7],
                         [2,8,9]]
        
'''