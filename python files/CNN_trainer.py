# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# Student Name: Usama zidan
# Id: 18025713
# ######
# This file handles loading the data, training and exporting the classifier model
# Most of functions used are well documented in the fastai liberary site
# %% [markdown]
# # Loading Libraries

# %%
import torchvision
from fastai.vision import *
from fastai.metrics import error_rate
from fastai import *
# import cv2 as cv
import numpy as np
import pandas as pd
import fastai
from sklearn.model_selection import StratifiedKFold
from class_loss import * # classifier loss functions


# %%
from fastai.metrics import *


# %%
from wandb.fastai import WandbCallback
import wandb


# %%
from fastai.callbacks import OverSamplingCallback


# %%
# initialising logging framwork
run = wandb.init(project='Classifier',name="224oversampled",reinit =True)

wandb.config.batch_size = 32
wandb.config.img_size = (256, 256)
wandb.config.learning_rate =1e-3
wandb.config.weight_decay = 1e-2
wandb.config.num_epochs = 6+12
wandbclc=partial(WandbCallback,log="all",input_type='images')


# %%
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# # Making the training set and dataloader

# %%
tfms = get_transforms(do_flip=True, flip_vert=False, max_lighting=0.1, max_zoom=1.05,
                      max_warp=0.,
                      xtra_tfms=[rand_crop(), rand_zoom(1, 1.5),
                                 symmetric_warp(magnitude=(-0.2, 0.2))])


path = '/cs/home/khfy6uat/data/classification_1024/classifier_data' #PATH TO DATASET
seed = 10
data = (ImageList.from_folder(path)
        .split_by_rand_pct(seed=10)
        .label_from_folder()
        .transform(tfms, size=224)
        .databunch(bs=13).normalize(imagenet_stats))


# %%
data.show_batch(rows=3, figsize=(12,9)) # displays a batch of the training set


# %%
data # displays the data details


# %%
# class names and number of classes
print(data.classes)
len(data.classes),data.c


# %%
f_score = partial(fbeta, thresh=0.2, beta = 0.5)
per = Precision()
rec= Recall()


# %%
import pretrainedmodels # model library


# %%
import pretrainedmodels
def resnext50_32x4d(pretrained=True):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    return nn.Sequential(*list(model.children()))


# %%



# %%
#metrics 
def fbeta_0(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True)->Rank0Tensor:
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred[:,0]>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=-1)
    prec = TP/(y_pred.sum(dim=-1)+eps)
    rec = TP/(y_true.sum(dim=-1)+eps)
    res = (prec*rec)/(prec+rec+eps)*2
    return res.mean()
def fbeta_1(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True)->Rank0Tensor:
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred[:,1]>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=-1)
    prec = TP/(y_pred.sum(dim=-1)+eps)
    rec = TP/(y_true.sum(dim=-1)+eps)
    res = (prec*rec)/(prec+rec+eps)*2
    return res.mean()
def fbeta_thr(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True)->Rank0Tensor:
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    y_pred.float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=-1)
    prec = TP/(y_pred.sum(dim=-1)+eps)
    rec = TP/(y_true.sum(dim=-1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean()


# %%
path = '/cs/home/khfy6uat/data/classification_1024/classifier_data'
seed = 10
data = (ImageList.from_folder(path)
        .split_by_rand_pct(0)
        .label_from_folder()
        .transform(tfms, size=512)
        .databunch(bs=6).normalize(imagenet_stats))
SEED=2020
# train_path = 
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

df=data.to_df()
print(df)
for train_index,val_index in kf.split(df.index,df['y']):
    print(train_index.shape,val_index)
    data_fold=(ImageList.from_df(df,path)
        .split_by_idxs(train_index,val_index)
        .label_from_df()
        .transform(tfms, size=224)
        .databunch(num_workers=0,bs=64).normalize(imagenet_stats))


# %%
learn = cnn_learner(data, resnext50_32x4d, pretrained=True, cut=-2,
                    split_on=lambda m: (m[0][3], m[1]), 
                    metrics=[per,rec,fbeta_0,fbeta_1,AUROC()],callback_fns=[wandbclc])
learn.callbacks=[OverSamplingCallback(learn)]
# learn.loss_fn = fbeta_loss()


# %%
learn = cnn_learner(data,models.densenet201, metrics=[per,rec,fbeta_0,fbeta_1,AUROC()])
learn.loss_fn = fbeta_loss()


# %%
### cooment in, to load an already trained model, specfy full path below without the 'pth' extension
# learn.load('f1model')
# learn.data=data


# %%
learn.opt_func=RAdam # specifying which optimiser to use


# %%
# RUNNING LEARNING RATE FINDER TO FIND BEST LEARNING RATE 
learn.lr_find()
learn.recorder.plot(suggestion=True)

# %% [markdown]
# # Stage 1 training with size 128

# %%
learn.loss_func=F1_loss_sig_all() # loading loss function
# first training cycle, 24 epochs with lr 0.02 and wieght decay 0.000005
learn.fit_one_cycle(24, max_lr=slice(2e-2), wd=1e-5)  


# %%
#unfreeezing early layers for training
learn.unfreeze();
#gradient clipping
learn = learn.clip_grad();


# %%
# Second training cycle, 32 epochs with cyclic lr 0.0003 and wieght decay 0.00000005
lr = [3e-3/100, 3e-3/20, 3e-3/10]
learn.fit_one_cycle(32, lr, wd=1e-7)

# %% [markdown]
# 
# # Adding cutout augmentation

# %%
SZ = 224
cutout_frac = 0.20
p_cutout = 0.75
cutout_sz = round(SZ*cutout_frac)
cutout_tfm = cutout(n_holes=(1,1), length=(cutout_sz, cutout_sz), p=p_cutout)

tfms = get_transforms(do_flip=True, max_rotate=15, flip_vert=False, max_lighting=0.1,
                      max_zoom=1.05, max_warp=0.,
                      xtra_tfms=[rand_crop(), rand_zoom(1, 1.5),
                                 symmetric_warp(magnitude=(-0.2, 0.2)), cutout_tfm])


# %%
data = (ImageList.from_folder(path)
        .split_by_rand_pct(seed=10)
        .label_from_folder()
        .transform(tfms, size=224)
        .databunch(bs=32).normalize(imagenet_stats))

learn.data = data
learn.freeze()


# %%
# 3rd training cycle, 24 epochs with lr 0.02 and wieght decay 0.00006
learn.fit_one_cycle(24, slice(2e-2), wd=5e-6)


# %%
# unfreezing and gradient clipping
learn.unfreeze();
learn = learn.clip_grad();


# %%
# Final training cycle, 32 epochs with lr 0.0001 and wieght decay 0.000005
lr = [1e-3/200, 1e-3/20, 1e-3/10]
learn.fit_one_cycle(32, lr)


# %%
#saving model
learn.save('/cs/home/khfy6uat/bin/classifier')

# %% [markdown]
# # Training on 512

# %%
gc.collect()
torch.cuda.empty_cache()
tfms = get_transforms(do_flip=True, flip_vert=False, max_lighting=0.1, max_zoom=1.05,
                      max_warp=0.,
                      xtra_tfms=[rand_crop(), rand_zoom(1, 1.5),
                                 symmetric_warp(magnitude=(-0.2, 0.2))])

data = (ImageList.from_folder(path)
        .split_by_rand_pct(seed=10)
        .label_from_folder()
        .transform(tfms, size=512)
        .databunch(bs=5).normalize(imagenet_stats))

learn.data = data
# learn.freeze()


# %%
# learn.callback_fns=[wandbclc]
learn.loss_func=fbeta_loss_sig_all()
learn.fit_one_cycle(12, max_lr=slice(2e-2), wd=1e-5)


# %%
learn.unfreeze();
learn = learn.clip_grad();


# %%
lr = [1e-3/200, 1e-3/20, 1e-3/10]
learn.fit_one_cycle(24, lr)


# %%
learn.save('/cs/home/khfy6uat/bin/Focal_f1score224256')


# %%
learn.recorder.plot_lr(),learn.recorder.plot_metrics(),learn.recorder.plot_losses()


# %%
interp = ClassificationInterpretation.from_learner(learn)


# %%
interp.plot_confusion_matrix()


# %%
wandb.history.torch
# learn.export('cnnn.pkl')


# %%
import pretrainedmodels
from fastai.metrics import error_rate
from fastai.metrics import Precision
from fastai.metrics import Recall
per = Precision()
rec= Recall()
def resnext50_32x4d(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    return nn.Sequential(*list(model.children()))
wandbclc=partial(WandbCallback,log="all",input_type='images',monitor='recall',mode='max')
learn = cnn_learner(data, resnext50_32x4d, pretrained=True, cut=-2,
                split_on=lambda m: (m[0][3], m[1]), 
                    metrics=[per,rec,error_rate],callback_fns=[wandbclc])


# %%
import fastprogress
fastprogress.fastprogress.NO_BAR = True
master_bar, progress_bar = fastprogress.fastprogress.force_console_behavior()
fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar
fastai.basic_data.master_bar, fastai.basic_data.progress_bar = master_bar, progress_bar
dataclass.master_bar, dataclass.progress_bar = master_bar, progress_bar

fastai.core.master_bar, fastai.core.progress_bar = master_bar, progress_bar


# %%



# %%
def get_learner(data):
   
    return learn 


# %%
get_learner(data)


# %%
get_ipython().run_line_magic('matplotlib', 'inline')
import torchvision
from fastai.vision import *
from fastai.metrics import error_rate
from fastai import *
def train(learn):
    import fastprogress

    # import cv2 as cv
    import numpy as np
    import pandas as pd
    import fastai
    from sklearn.model_selection import KFold
    from wandb.fastai import WandbCallback
    import wandb
    from torch import nn
    import torch.nn.functional as F
    ALPHA = 2.0
    BETA = 10000.5
    GAMMA = 10
    import pretrainedmodels
    def resnext50_32x4d(pretrained=False):
        pretrained = 'imagenet' if pretrained else None
        model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
        return nn.Sequential(*list(model.children()))
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1., gamma=2.):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets, **kwargs):
            CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
            pt = torch.exp(-CE_loss)
            F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
            return F_loss.mean()
    print(1)
    path = '/cs/home/khfy6uat/data/classification_1024/classifier_data'

    fastprogress.fastprogress.NO_BAR = True
    master_bar, progress_bar = fastprogress.fastprogress.force_console_behavior()
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar
    fastai.basic_data.master_bar, fastai.basic_data.progress_bar = master_bar, progress_bar
    dataclass.master_bar, dataclass.progress_bar = master_bar, progress_bar

    fastai.core.master_bar, fastai.core.progress_bar = master_bar, progress_bar
#     def resnext50_32x4d(pretrained=False):
#         pretrained = 'imagenet' if pretrained else None
#         model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
#         return nn.Sequential(*list(model.children()))
#     class FocalLoss(nn.Module):
#         def __init__(self, alpha=1., gamma=2.):
#             super().__init__()
#             self.alpha = alpha
#             self.gamma = gamma

#         def forward(self, inputs, targets, **kwargs):
#             CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
#             pt = torch.exp(-CE_loss)
#             F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
#             return F_loss.mean()
    sz=0
    sz1=0
    # Default values for hyper-parameters we're going to sweep over
    config_defaults = {
        'epochs': 2,
        'batch_size': 12,
        'weight_decay': 0.0005,
        'learning_rate': 1e-3,
        'seed': 42,
        'encoder_size':128,
        'decoder_size':224
    }
      # Initialize a new wandb run
    wandb.init(config=config_defaults)
    
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    
    sz=(config.encoder_size)

    sz1=(config.decoder_size)


    tfms = get_transforms(do_flip=True, flip_vert=False, max_lighting=0.1, max_zoom=1.05,
                      max_warp=0.,
                      xtra_tfms=[rand_crop(), rand_zoom(1, 1.5),
                                 symmetric_warp(magnitude=(-0.2, 0.2))])
    print('right before data')
    data = (ImageList.from_folder(path)
        .split_by_rand_pct(seed=10)
        .label_from_folder()
        .transform(tfms, size=sz)
        .databunch(bs=12).normalize(imagenet_stats))
    print('right before learner')
    wandbclc=partial(WandbCallback,log="all",input_type='images',monitor='recall',mode='max')
    per = Precision()
    rec= Recall()

#     learn = cnn_learner(data, resnext50_32x4d, pretrained=True, cut=-2,
#                     split_on=lambda m: (m[0][3], m[1]), 
#                         metrics=[per,rec,error_rate],callback_fns=[wandbclc])   
    print(learn.data.train_dl.batch_size)  
    lr=config.learning_rate
    print(config.epochs)
    learn.fit_one_cycle(config.epochs, max_lr=slice(lr), wd=1e-5)
    
    learn.unfreeze();
    learn = learn.clip_grad();
    lr = [lr/200, lr/20, lr/10]

    learn.fit_one_cycle(config.epochs, max_lr=slice(lr), wd=1e-5)

    if (sz1 > 0):
        SZ = sz1
        cutout_frac = 0.20
        p_cutout = 0.75
        cutout_sz = round(SZ*cutout_frac)
        cutout_tfm = cutout(n_holes=(1,1), length=(cutout_sz, cutout_sz), p=p_cutout)

        tfms = get_transforms(do_flip=True, max_rotate=15, flip_vert=False, max_lighting=0.1,
                              max_zoom=1.05, max_warp=0.,
                              xtra_tfms=[rand_crop(), rand_zoom(1, 1.5),
                                         symmetric_warp(magnitude=(-0.2, 0.2)), cutout_tfm])
        data = (ImageList.from_folder(path)
            .split_by_rand_pct(seed=10)
            .label_from_folder()
            .transform(tfms, size=sz1)
            .databunch(bs=12).normalize(imagenet_stats))

        learn.data=data
        learn.fit_one_cycle(config.epochs, max_lr=slice(lr), wd=1e-5)

        learn.unfreeze();
        learn = learn.clip_grad();
        lr = [lr/200, lr/20, lr/10]

        learn.fit_one_cycle(config.epochs, max_lr=slice(lr), wd=1e-5)


# %%
# Configure the sweep – specify the parameters to search through, the search strategy, the optimization metric et all.
sweep_config = {
    'controller':{'type':'local'
    },
    'method': 'grid', #grid, random
    'metric': {
      'name': 'recall',
      'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {
            'values': [24, 32, 39,20]
        },
        'encoder_size': {
            'values': [128,224,256]
        },
        'decoder_size':{
            'values':[128,224,256,0]
        },
        'learning_rate': {
            'values': [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
        }
    }
}


# %%
sweep_id = wandb.sweep(sweep_config, entity="usama_ml", project="testing_sweeps_classifier")


# %%
sweep = wandb.controller(sweep_id)
sweep.run()

while not sweep.done():
    sweep.print_status()
    sweep.step()
    train(learn)
    time.sleep(5)


# %%
get_ipython().system('wandb sweep sweep_grid_f1loss_trials.yaml')


# %%
get_ipython().system('wandb agent usama_ml/OneClass_segmentation_group_normalization_testing_2/aewda6a5')


# %%
get_ipython().system('pip install wandb --upgrade')


# %%
get_ipython().system('nvidia-smi')


# %%
learn.export('fastai_resnet.pkl');

# %% [markdown]
# # Predicting on the test set

# %%
learn = load_learner('/cs/home/khfy6uat/data/classfication_128/classifier_data/','fastai_resnet.pkl', ImageList.from_folder('/cs/home/khfy6uat/data/data1024/test/'))
preds,_ = learn.get_preds(ds_type=DatasetType.Test)
cls_pred = (F.softmax(preds,1)[:,0]<0.99).cpu().numpy()


# %%
cls_pred = (F.softmax(preds,1)).argmax(1).cpu().numpy()


# %%
paths = list(map(str,list(learn.data.test_ds.x.items)))
all_test_paths = [p.split('/')[-1][:-4] for p in paths]
ids = [o.stem for o in learn.data.test_ds.x.items]

df_preds = pd.DataFrame()
df_preds['test_paths'] = ids
df_preds['class_pred'] = cls_pred

df_preds.set_index('test_paths',inplace=True)


# %%
df_preds.head()


# %%
no_dis_idx = df_preds[df_preds.class_pred==1].index
len(no_dis_idx),3205-(len(no_dis_idx))


# %%
sub = pd.read_csv('/cs/home/khfy6uat/bin/submission_128_244_256_tfms.csv'
                  ,index_col='ImageId')
sub.head()


# %%
sub.loc[no_dis_idx] = '-1'


# %%
sub.to_csv('sub_classifier_correction_thresh_new_model_unshnaged.csv')


# %%
get_ipython().system('rm -r */')


# %%
sub = pd.read_csv('/cs/home/khfy6uat/bin/sub_classifier_correction.csv'
                  ,index_col='ImageId')


# %%
learn.data=data


# %%
learn.data.classes


# %%
preds[11:20,0],preds[11:20,1],ys[11:20]


# %%
preds,ys = learn.get_preds(ds_type=DatasetType.Valid)


# %%
def fbeta(y_pred:Tensor, y_true:Tensor, thresh:float=0.9, sigmoid:bool=False):
    "Computes the f_beta between `preds` and `targets`"
  
    if sigmoid: y_pred = y_pred.sigmoid()
#     y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum()
    prec = TP/(y_pred.sum())
    rec = TP/(y_true.sum())
    res = ((prec*rec)/(prec+rec))*2
    return res.mean()


# %%
#finding a thresh to minimise the number of postives detected 
fs = []
scores, best_thrs = [],[]
thrs = np.arange(0.01, 1, 0.01)

for th in progress_bar(thrs):  
    cls_pred = (F.softmax(preds,1)[:,1]>th).cpu().numpy()
    paths = list(map(str,list(learn.data.test_ds.x.items)))
    all_test_paths = [p.split('/')[-1][:-4] for p in paths]
    ids = [o.stem for o in learn.data.test_ds.x.items]

    df_preds = pd.DataFrame()
    df_preds['test_paths'] = ids
    df_preds['class_pred'] = cls_pred

    df_preds.set_index('test_paths',inplace=True)
    no_dis_idx = df_preds[df_preds.class_pred==1].index
    fs.append(3205-(len(no_dis_idx)))
fs=np.array(fs)
scores.append(fs.min())
best_thrs.append(thrs[fs.argmin()])
best_thr = np.array(best_thrs).mean()

best_dice = fs.min()
plt.figure(figsize=(8,4))
plt.plot(thrs, fs)
plt.vlines(x=best_thrs[-1], ymin=fs.min(), ymax=fs.max())
plt.text(best_thrs[-1]+0.03, best_dice-0.01, f'F1 = {best_dice:.3f}', fontsize=14);
plt.show(),best_dice,best_thr


# %%
ys=ys
preds_=preds[:,1]
n = ys.shape[0]
preds_sig=preds_
# preds_sig=preds_.sigmoid()
# preds_sig=F.softmax(preds_,1)
thrs = np.arange(0.01, 1, 0.01)
targs = ys
fs = []
scores, best_thrs = [],[]
for th in progress_bar(thrs):
    pred = (preds_sig>th)
    fs.append(fbeta_thr(pred,targs))
fs=np.array(fs)
scores.append(fs.max())
best_thrs.append(thrs[fs.argmax()])
best_thr = np.array(best_thrs).mean()

best_dice = fs.max()
plt.figure(figsize=(8,4))
plt.plot(thrs, fs)
plt.vlines(x=best_thrs[-1], ymin=fs.min(), ymax=fs.max())
plt.text(best_thrs[-1]+0.03, best_dice-0.01, f'F1 = {best_dice:.3f}', fontsize=14);
plt.show()
    
# pre = preds.argmax(-1).view(-1).cpu()
# tar = ys.cpu()

# fbeta(pred,targs),preds_sig,pred


# %%
fbeta_0(preds,ys)


# %%
(preds[:,0]),(preds[:,0]).sigmoid(),preds[:,1],preds[:,1].sigmoid()


# %%
ys


# %%
F.softmax(preds)


# %%
len(learn.model)


# %%
learn. model


# %%
learn.model[0]


# %%
for i in range(1):
    print(i)


# %%
w= [2,1,2]
for i,v in w.items():
    print(i,v)


# %%
import wandb
import fastai
from fastai.callbacks import TrackerCallback
from pathlib import Path
import random
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend (avoid tkinter issues)
    import matplotlib.pyplot as plt
except:
    print('Warning: matplotlib required if logging sample image predictions')

class WandbCallback(TrackerCallback):
    """
    Automatically saves model topology, losses & metrics.
    Optionally logs weights, gradients, sample predictions and best trained model.
    Args:
        learn (fastai.basic_train.Learner): the fast.ai learner to hook.
        log (str): "gradients", "parameters", "all", or None. Losses & metrics are always logged.
        save_model (bool): save model at the end of each epoch. It will also load best model at the end of training.
        monitor (str): metric to monitor for saving best model. None uses default TrackerCallback monitor value.
        mode (str): "auto", "min" or "max" to compare "monitor" values and define best model.
        input_type (str): "images" or None. Used to display sample predictions.
        validation_data (list): data used for sample predictions if input_type is set.
        predictions (int): number of predictions to make if input_type is set and validation_data is None.
        seed (int): initialize random generator for sample predictions if input_type is set and validation_data is None.
    """

    # Record if watch has been called previously (even in another instance)
    _watch_called = False

    def __init__(self,
                 learn,
                 log="gradients",
                 save_model=True,
                 monitor=None,
                 mode='auto',
                 input_type=None,
                 validation_data=None,
                 predictions=36,
                 seed=12345):

        # Check if wandb.init has been called
        if wandb.run is None:
            raise ValueError(
                'You must call wandb.init() before WandbCallback()')

        # Adapted from fast.ai "SaveModelCallback"
        if monitor is None:
            # use default TrackerCallback monitor value
            super().__init__(learn, mode=mode)
        else:
            super().__init__(learn, monitor=monitor, mode=mode)
        self.save_model = save_model
        self.model_path = Path(wandb.run.dir) / 'bestmodel.pth'

        self.log = log
        self.input_type = input_type
        self.best = None

        # Select items for sample predictions to see evolution along training
        self.validation_data = validation_data
        if input_type and not self.validation_data:
            wandbRandom = random.Random(seed)  # For repeatability
            predictions = min(predictions, len(learn.data.valid_ds))
            indices = wandbRandom.sample(range(len(learn.data.valid_ds)),
                                         predictions)
            self.validation_data = [learn.data.valid_ds[i] for i in indices]

    def on_train_begin(self, **kwargs):
        "Call watch method to log model topology, gradients & weights"

        # Set self.best, method inherited from "TrackerCallback" by "SaveModelCallback"
        super().on_train_begin()

        # Ensure we don't call "watch" multiple times
        if not WandbCallback._watch_called:
            WandbCallback._watch_called = True

            # Logs model topology and optionally gradients and weights
            wandb.watch(self.learn.model, log=self.log)

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        "Logs training loss, validation loss and custom metrics & log prediction samples & save model"

        if self.save_model:
            # Adapted from fast.ai "SaveModelCallback"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                print(
                    'Better model found at epoch {} with {} value: {}.'.format(
                        epoch, self.monitor, current))
                self.best = current

                # Save within wandb folder
                with self.model_path.open('wb') as model_file:
                    self.learn.save(model_file)

        # Log sample predictions if learn.predict is available
        if self.validation_data:
            try:
                self._wandb_log_predictions()
#                 self._log_otherstuff()
            except FastaiError as e:
                wandb.termwarn(e.message)
                self.validation_data = None  # prevent from trying again on next loop
            except Exception as e:
                wandb.termwarn("Unable to log prediction samples.\n{}".format(e))
                self.validation_data=None  # prevent from trying again on next loop

        # Log losses & metrics
        # Adapted from fast.ai "CSVLogger"
        logs = {
            name: stat
            for name, stat in list(
                zip(self.learn.recorder.names, [epoch, smooth_loss] +
                    last_metrics))
        }
        wandb.log(logs)
       



    def on_train_end(self, **kwargs):
        "Load the best model."

        if self.save_model:
            # Adapted from fast.ai "SaveModelCallback"
            if self.model_path.is_file():
                with self.model_path.open('rb') as model_file:
                    self.learn.load(model_file, purge=False)
                    print('Loaded best saved model from {}'.format(
                        self.model_path))

    def _log_otherstuff(self):
        preds,y,losses = self.learn.get_preds(with_loss=True)

        interp = ClassificationInterpretation(self.learn, preds, y, losses)

        plt_cm=interp.plot_confusion_matrix(return_fig=True)
        wandb.log({'roc': wandb.plots.ROC(y, preds, self.learn.data.classes)})
        wandb.log({'pr': wandb.plots.precision_recall(y, preds, self.learn.data.classes)})
        wandb.log({"CM": plt_cm})
        
    def _wandb_log_predictions(self):
        "Log prediction samples"

        pred_log = []
        pred_log_2=[]
        y_log=[]
        for x, y in self.validation_data:
            
            try:
                pred=self.learn.predict(x)
                pred_log_2.append(pred[2])
                y_log.append(y)
            except:
                raise FastaiError('Unable to run "predict" method from Learner to log prediction samples.')

            # scalar -> likely to be a category
            # tensor of dim 1 -> likely to be multicategory
            if not pred[1].shape or pred[1].dim() == 1:
                pred_log.append(
                    wandb.Image(
                        x.data,
                        caption='Ground Truth: {}\nPrediction: {}'.format(
                            y, pred[0])))

            # most vision datasets have a "show" function we can use
            elif hasattr(x, "show"):
                # log input data
                pred_log.append(
                    wandb.Image(x.data, caption='Input data', grouping=3))

                # log label and prediction
                for im, capt in ((pred[0], "Prediction"),
                                 (y, "Ground Truth")):
                    # Resize plot to image resolution
                    # from https://stackoverflow.com/a/13714915
                    my_dpi = 100
                    fig = plt.figure(frameon=False, dpi=my_dpi)
                    h, w = x.size
                    fig.set_size_inches(w / my_dpi, h / my_dpi)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)

                    # Superpose label or prediction to input image
                    x.show(ax=ax, y=im)
                    pred_log.append(wandb.Image(fig, caption=capt))
                    plt.close(fig)

            # likely to be an image
            elif hasattr(y, "shape") and (
                (len(y.shape) == 2) or
                    (len(y.shape) == 3 and y.shape[0] in [1, 3, 4])):

                pred_log.extend([
                    wandb.Image(x.data, caption='Input data', grouping=3),
                    wandb.Image(pred[0].data, caption='Prediction'),
                    wandb.Image(y.data, caption='Ground Truth')
                ])

            # we just log input data
            else:
                pred_log.append(wandb.Image(x.data, caption='Input data'))

            wandb.log({"Prediction Samples": pred_log}, commit=False)
            
#             interp = ClassificationInterpretation(self.learn, pred_log_2, y_log, losses)

#             plt_cm=interp.plot_confusion_matrix(return_fig=True)
        print(torch.stack(y_log),torch.stack(pred_log_2))
#         wandb.log({'roc': wandb.plots.ROC(torch.stack(y_log).numpy(), torch.stack(pred_log_2).numpy(), self.learn.data.classes)},commit=False)
#         wandb.log({'pr': wandb.plots.precision_recall( torch.stack(y_log).numpy(), torch.stack(pred_log_2).numpy(), self.learn.data.classes)},commit=False)
# #             wandb.log({"CM": plt_cm})
        print("LOGGED #####")

class FastaiError(wandb.Error):
    pass


# %%
p=learn.predict(learn.data.valid_dl.x[1])
p[2].shape


# %%
l=[tensor([0.8460, 0.1540]), tensor([0.6084, 0.3916]), tensor([0.5220, 0.4780]), tensor([0.0553, 0.9447]), tensor([0.9988, 0.0012]), tensor([0.9975, 0.0025]), tensor([0.7724, 0.2276]), tensor([0.9929, 0.0071]), tensor([0.6132, 0.3868]), tensor([0.9828, 0.0172]), tensor([0.0512, 0.9488]), tensor([0.7204, 0.2796]), tensor([0.8214, 0.1786]), tensor([0.9087, 0.0913]), tensor([0.1048, 0.8952]), tensor([0.6428, 0.3572]), tensor([0.7324, 0.2676]), tensor([9.9942e-01, 5.7936e-04]), tensor([0.0310, 0.9690]), tensor([0.9373, 0.0627]), tensor([0.1187, 0.8813]), tensor([0.3811, 0.6189]), tensor([0.8052, 0.1948]), tensor([0.6279, 0.3721]), tensor([0.4554, 0.5446]), tensor([0.7444, 0.2556]), tensor([0.9768, 0.0232]), tensor([0.1312, 0.8688]), tensor([0.7607, 0.2393]), tensor([0.4091, 0.5909]), tensor([0.7046, 0.2954]), tensor([0.6373, 0.3627]), tensor([0.4189, 0.5811]), tensor([0.8620, 0.1380]), tensor([0.1797, 0.8203]), tensor([0.9822, 0.0178])]


# %%
len(l)


# %%
b = torch.stack(l)
b.shape


# %%
torch.Tensor(36,2).numpy()


# %%
import math
import torch
from torch.optim.optimizer import Optimizer, required

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss

class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
                    
        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)


                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup = 0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup = warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
                
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss


# %%


