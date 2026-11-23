import torch
from torch.utils.data import DataLoader,TensorDataset


def build_loaders(config):
    generator = torch.Generator().manual_seed(config['seed'])
    if config['synthetic']:
        loaders=[]
        for split,n in enumerate([24,6,18]):
            g=torch.Generator().manual_seed(config['seed']+split)
            y=torch.eye(3)[torch.arange(n)%3]
            x=torch.randn(n,3,16,16,generator=g)*.1+y[:,:,None,None]
            ds=TensorDataset(x,y,torch.arange(n))
            loaders.append(DataLoader(ds,batch_size=config['batch_size'],shuffle=split==0,
                generator=generator if split==0 else torch.Generator()))
        return (*loaders,3)
    # Reuse Hash-Zoo public dataset definitions, splits and transforms.
    from common.hashzoo_data import init_dataset,ImageDataset,build_trans,get_class_num
    data=init_dataset(config['dataset'],config['data_dir'])
    loaders=[]
    for i,split in enumerate(['train','query','dbase']):
        ds=ImageDataset(getattr(data,split),build_trans('train' if i==0 else 'other'))
        loaders.append(DataLoader(ds,batch_size=config['batch_size'],shuffle=i==0,
            num_workers=config['n_workers'],drop_last=False,
            generator=generator if i==0 else torch.Generator()))
    if set(loaders[1].dataset.data[0]) & set(loaders[2].dataset.data[0]):
        raise ValueError('Query/database identity overlap: explicit exclusion is required')
    return (*loaders,get_class_num(config['dataset']))
