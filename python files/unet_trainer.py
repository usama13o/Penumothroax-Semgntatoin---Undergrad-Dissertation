# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# Student Name: Usama zidan
# Id: 18025713
# ######
# This file handles loading the data, training and exporting the segmentaiton models
# Most of functions used are well documented in the fastai liberary site

# %%
import sys
get_ipython().run_line_magic('matplotlib', 'inline')
import fastai
from fastai.vision import *
from mask_functions import *
from sklearn.model_selection import KFold
from unet_loss import *


# %%
fastai.__version__


# %%
#Loading path to data 
SZ = 256
# path = Path(f'/cs/home/khfy6uat/data/data1024/train') # path to full data 
path = Path(f'/cs/home/khfy6uat/data/one_class_data_1024') # path to one case data


# %%
# Setting div=True in open_mask
class SegmentationLabelList(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)
    
class SegmentationItemList(SegmentationItemList):
    _label_cls = SegmentationLabelList

# Setting transformations on masks to False on test set
def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
    if not tfms: tfms=(None,None)
    assert is_listy(tfms) and len(tfms) == 2
    self.train.transform(tfms[0], **kwargs)
    self.valid.transform(tfms[1], **kwargs)
    kwargs['tfm_y'] = False # Test data has no labels
    if self.test: self.test.transform(tfms[1], **kwargs)
    return self
fastai.data_block.ItemLists.transform = transform


# %%
#metric definitions
def dice2(input:Tensor, targs:Tensor, iou:bool=False, eps:float=1e-8)->Rank0Tensor:
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(-1)
    targs = targs.view(-1)
    intersect = (input * targs).sum(dim=-1).float()# A (and) B 
    union = (input+targs).sum(dim=-1).float()# A + B [Total not actuall union]
    if not iou: l = 2. * (intersect+eps) / (union+eps)
    return l
def iou(input:Tensor, targs:Tensor, iou:bool=True, eps:float=1e-8)->Rank0Tensor:
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(-1)
    targs = targs.view(-1)
    intersect = (input * targs).sum(dim=-1).float()
    union = (input+targs).sum(dim=-1).float()
    l = (intersect+eps) / (union-intersect+eps)
    return l


# %%
# Create databunch

data = (SegmentationItemList.from_folder(path=path/'train')
        .split_by_rand_pct(.2)
        .label_from_func(lambda x : str(x).replace('train', 'masks'), classes=[0, 1])
        .transform(get_transforms(), size=256, tfm_y=True)
        .databunch(path=Path('.'), bs=12)
        .normalize(imagenet_stats))
data


# %%
#initializing logging framework
from wandb.fastai import WandbCallback
import wandb

run = wandb.init(project='unet_testing_loss funtion',name="One_class_unet_JACC_256_adam_generated",reinit =True)

wandbclc=partial(WandbCallback,log="all",input_type='images')


# %%
# Create U-Net with a pretrained resnet34 as encoder
learn = unet_learner(data,models.resnet34,metrics=[dice,dice2,iou],callback_fns=[wandbclc])


# %%
learn.summary()


# %%
wrap_BN(learn.model.layers)


# %%
learn.model.layers[1]=bn2group(learn.model.layers[1])
for i in [4,5,6,7]:
    learn.model.layers[i].bn=bn2group(learn.model.layers[i].bn)
  


# %%
### comment in the below line to load a model
# learn.load('/cs/home/khfy6uat/bin/oneclass_normalUnet_Combo_1024_2')


# %%
# Fit one cycle of 6 epochs with max lr of 1e-3
lr = 1e-3
#loading loss function
learn.loss_func = JaccardLoss()


# %%
# Fit one cycle of 12 epochs
lr = 1e-3
learn.fit_one_cycle(12, slice(lr/30, lr),callbacks=ShowGraph(learn))


# %%
#saving model
learn.save('/cs/home/khfy6uat/bin/oneclass_normalUnet_JACC_adam_BN_generated')


# %%



# %%
# for k-fold corss validation, the remainag lines cross validate the model by runing the smae cycle 5 times on the sliced data
kf = KFold(n_splits=5, shuffle=True, random_state=2020)
#Loading path to data 
SZ = 256
# path = Path(f'/cs/home/khfy6uat/data/data1024/train') # path to full data 
path = Path(f'/cs/home/khfy6uat/data/one_class_data_1024') # path to one case data

def get_fold(fold_number):
    valid_idx = list(kf.split(list(range(len((path/'train').ls())))))[fold_number][1]
    print(valid_idx)
    # Create databunch
    data = (SegmentationItemList.from_folder(path=path/'train')
        .split_by_rand_pct(.2)
        .label_from_func(lambda x : str(x).replace('train', 'masks'), classes=[0, 1])
        .transform(get_transforms(), size=256, tfm_y=True)
        .databunch(path=Path('.'), bs=12)
        .normalize(imagenet_stats))
    return data


# %%
from wandb.fastai import WandbCallback
import wandb
for fold_number in range(5):
    from wandb.fastai import WandbCallback
    import wandb
    print("Fold:"+str(fold_number))
    # simply change the the string for the varaive 'name' to change the output fold model name depending on wich model to train

    run = wandb.init(project='U-Net cross validation',name="Baseline_fold_"+str(fold_number),reinit =True)

    wandbclc=partial(WandbCallback,log="all",input_type='images')

    data=get_fold(fold_number)
    
    learn = unet_learner(data,models.resnet34,metrics=[dice,dice2,iou],callback_fns=[wandbclc])
    # Fit one cycle of 6 epochs with max lr of 1e-3
    lr = 1e-3
    learn.loss_fn = JaccardLoss()
    print(learn.loss_fn,learn.loss_func)
    learn.fit_one_cycle(6,lr)
    # Unfreeze the encoder (resnet34)
    learn.unfreeze()
    # Fit one cycle of 12 epochs
    lr = 1e-3
    learn.fit_one_cycle(12, slice(lr/30, lr),callbacks=ShowGraph(learn))
    learn.save("Baseline_fold_"+str(fold_number))


# %%


