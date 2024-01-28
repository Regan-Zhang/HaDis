import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

# device_ids = [0, 1, 2, 3]
device_ids = [0, ]
torch.cuda.set_device(device_ids[0])
torch.multiprocessing.set_sharing_strategy('file_system')

from models.basic_template import TrainTask

dataset_name = 'cifar10'  # cifar10/ cifar20 / ImageNet-dogs / ImageNet-10/train / tiny-imagenet-200/train
# dataset_name = 'imagenet'
test_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    TrainTask.normalize(dataset_name)
])
# test_transform = transforms.Compose([
#     transforms.Resize(96),
#     transforms.CenterCrop(96),
#     transforms.ToTensor(),
#     TrainTask.normalize(dataset_name)
# ])
train_dataset = TrainTask.create_dataset(
    data_root='/home/derek/datasets',
    dataset_name=dataset_name,
    train=True,
    transform=test_transform,
    memory=True,
)[0]
test_dataset = TrainTask.create_dataset(
    data_root='/home/derek/datasets',
    dataset_name=dataset_name,
    train=False,
    transform=test_transform,
    memory=True,
)[0]
num_classes = len(np.unique(train_dataset.targets))
print(num_classes)
train_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
# dataset = torch.utils.data.ConcatDataset([create_dataset(), create_dataset(False)])
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1000,
                                           shuffle=False,
                                           num_workers=8)

from models.propos.byol_wrapper import BYOLWrapper
from network import backbone_dict

# backbone = 'resnet50'
backbone = 'bigresnet18'
encoder_type, dim_in = backbone_dict[backbone]
encoder = encoder_type()
byol = BYOLWrapper(encoder,
                   num_cluster=num_classes,
                   in_dim=dim_in,
                   temperature=0.5,
                   hidden_size=4096,
                   fea_dim=256,
                   byol_momentum=0.999,
                   symmetric=True,
                   shuffling_bn=True,
                   latent_std=0.001)

# for epoch in [1, 200, 400, 600, 800, 1000]:
for epoch in [1000]:
    checkpoint = f'ckpt/2023_08_04_21_51_48-cifar10_r18_propos/save_models/byol-{epoch}'
    msg = byol.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=False)
    print(msg)
    encoder = nn.Sequential(byol.encoder_k, byol.projector_k)
    encoder = nn.DataParallel(encoder, device_ids=device_ids).cuda().eval()

    from utils import extract_features

    mem_features, mem_labels = extract_features(encoder, train_loader)
    # test_features, test_labels = extract_features(encoder, test_loader)

    kwargs = {
        'metric': 'cosine',
        'distributed': True,
        'random_state': 0,
        'n_clusters': int(mem_labels.max()) + 1,
        'verbose': True
    }

    from torch_clustering import PyTorchKMeans, evaluate_clustering

    clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)

    psedo_labels = clustering_model.fit_predict(mem_features)
    cluster_centers = clustering_model.cluster_centers_

    results = evaluate_clustering(mem_labels.cpu().numpy(), psedo_labels.cpu().numpy(),
                                  eval_metric=['nmi', 'acc', 'ari'])
    print(results)


    def tsne(x, y, ind,p):  # cosine  euclidean   correlation
        # x = TSNE(learning_rate=1000, metric='cosine').fit_transform(x)
        # x = TSNE(n_components=2, perplexity=20).fit_transform(x)
        metric = 'cosine'
        x = TSNE(n_components=2, perplexity=p, n_iter=3000, metric=metric).fit_transform(x)
        # x = TSNE().fit_transform(x)
        print("X's shape:", x.shape)
        plt.figure()
        # ImageNet-10
        colours = ListedColormap(
            ['#ECAAD7', '#FF4B4B', '#966E5A', '#969696', '#C3C33C', '#FFB050', '#64C8DC', '#3E8CBE', '#41AA41',
             '#A078C8'])
        # ImageNet-dogs
        # colours = ListedColormap(
        #     ['#ECAAD7', '#FF4B4B', '#966E5A', '#969696', '#C3C33C', '#FFB050', '#64C8DC', '#3E8CBE', '#41AA41',
        #      '#A078C8', '#F8F819','#636304','#66FF66','#333333','#FF9999'])

        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=colours, s=1, marker='.')

        # plt.savefig(f'figures/tsne/{ind}-p{p}-metric{metric}.eps')
        plt.show()
        # plt.close()


    # print(mem_features.shape, mem_features.type)
    # print(mem_labels.shape, mem_labels.type)
    for p in [60]:
        tsne(mem_features.cpu().numpy(), psedo_labels.cpu().numpy(), f'CIFAR10-{epoch}',p)
