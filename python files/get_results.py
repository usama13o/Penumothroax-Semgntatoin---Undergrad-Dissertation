# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # This file is for assessing the models performances. It was used to get the results of the corss-validated models.
# %% [markdown]
# ## loading libraries

# %%
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import fastai
from fastai.vision import *
from mask_functions import *
from fastai.callbacks import *
import gc
from sklearn.model_selection import KFold
from PIL import Image
path = Path(f'/cs/home/khfy6uat/data/one_class_data_1024')

fastai.__version__

# %% [markdown]
# The original images, provided in this competition, have 1024x1024 resolution. To prevent additional overhead on image loading, the datasets composed of 256x256 scaled down images are prepared separately and used as an input. Check make-pneumothorax-oneCase-data for more details on image rescaling and mask generation.  

# %%

SEED = 2019


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #tf.set_random_seed(seed)
seed_everything(SEED)

# %% [markdown]
# ### Model

# %%
#metric used to assess the model performance
def dice_overall(preds, targs):
    n = preds.shape[0]
    preds = preds.view(n,-1)
    targs = targs.view(n,-1)
    intersect = (preds * targs).sum(1).float()
    total = (preds+targs).sum(1).float()
    u0 = total==0
    intersect[u0] = 1
    total[u0] = 2
    return (2 * intersect / total).mean()
def dice_overall_adjusted(preds, targs):
    n = preds.shape[0]
    preds = preds.view(-1)
    targs = targs.view(-1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds+targs).sum(-1).float()
#     u0 = union==0
    return (2 * intersect / union)
def iou_overall(preds, targs):
    preds = preds.view(-1)
    targs = targs.view(-1)
    intersect = (preds * targs).sum(-1).float()
    total = (preds+targs).sum(-1).float()
#     u0 = union==0
    return (intersect / (total-intersect))
def CM_overall(learn,preds,ys):
        mean_cm, _= generate_confusion(learn=learn,pred=preds,y_true=ys)
        return mean_cm[1][1]

# %% [markdown]
# The following function generates predictions with using flip TTA (average the result for the original image and a flipped one).

# %%
# Prediction with flip TTA
def pred_with_flip(learn:fastai.basic_train.Learner,
                   ds_type:fastai.basic_data.DatasetType=DatasetType.Valid):
    #get prediction
    preds, ys = learn.get_preds(ds_type)
    preds = preds[:,1,...]
    #add fiip to dataset and get prediction
    
    learn.data.dl(ds_type).dl.dataset.tfms.append(flip_lr())

    preds_lr, ys = learn.get_preds(ds_type)
    del learn.data.dl(ds_type).dl.dataset.tfms[-1]
    preds_lr = preds_lr[:,1,...]
    preds = 0.5*(preds + torch.flip(preds_lr,[-1]))
    del preds_lr
    gc.collect()
    torch.cuda.empty_cache()
    return preds, ys

# %% [markdown]
# ### Data

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
path = Path(f'/cs/home/khfy6uat/data/data1024')

def get_data(fold,tfms=get_transforms(),split=0.2,sz=sz,bs=bs):
    # Create databunch
    data = (SegmentationItemList.from_folder(path=path/'train')
            .split_by_rand_pct(split,seed=10)
            .label_from_func(lambda x : str(x).replace('train', '/masks'),classes=[0,1])
            .transform(None, size=sz, tfm_y=True)
            .databunch(path=Path('.'), bs=bs)
            .normalize(stats))
    return data

# Display some images with masks
# get_data(0).show_batch()

# %% [markdown]
# ### Training
# %% [markdown]
# # Postive Cases DATA 

# %%
path = Path(f'/cs/home/khfy6uat/data/one_class_data_1024')

data=get_data(0,sz=512,bs=12)
data

# %% [markdown]
# # Original DATA ALL CASES 

# %%
path = Path(f'/cs/home/khfy6uat/data/data1024')

data = (SegmentationItemList.from_folder(path=path/'train/train')
        .split_by_idx(list(range(1000)))
        .label_from_func(lambda x : str(x).replace('train/train', 'masks'), classes=[0, 1])
        .add_test((path/'test').ls(), label=None)
        .transform(get_transforms() ,size=256, tfm_y=True)
        .databunch(path=Path('.'), bs=12,num_workers=0)
        .normalize(imagenet_stats))


# %%
#loading a saved Unet mdoel (not learner!)
learn = unet_learner(data,models.resnet34,metrics=[dice,dice2,iou],callback_fns=[])

learn.load('/cs/home/khfy6uat/bin/models/baseline_unet_fold_4')


# %%
#get prediction on data, this was mainly used to check if everything was fine so far 
preds,ys = learn.get_preds()
preds = preds[:,1,...]
ys = ys.squeeze()
# dice_overall(preds,ys).mean()


# %%
#loading the classfier for the one case model results 
# a learner has to be first created to import a model into it 
path = '/cs/home/khfy6uat/data/classification_1024/classifier_data'
import pretrainedmodels
def resnext50_32x4d(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    return nn.Sequential(*list(model.children()))

tfms = get_transforms(do_flip=True, flip_vert=False, max_lighting=0.1, max_zoom=1.05,
                  max_warp=0.,
                  xtra_tfms=[rand_crop(), rand_zoom(1, 1.5),
                             symmetric_warp(magnitude=(-0.2, 0.2))])
data_cnn = (ImageList.from_folder(path)
    .split_by_rand_pct(seed=10)
    .label_from_folder()
    .transform(tfms, size=1024)
    .databunch(bs=2).normalize(imagenet_stats))
print('learner created!')
per = Precision()
rec= Recall()
learn_cnn = cnn_learner(data_cnn, resnext50_32x4d, pretrained=True, cut=-2,split_on=lambda m: (m[0][3], m[1]))
#loading calssifier
learn_cnn.model_dir='/cs/home/khfy6uat/bin/'
learn_cnn.load('f1loss_CNN')
learn_cnn.data=data
#getting indices of images that the classifier predicts as negative samples 
p,ys_cnn=learn_cnn.get_preds()# getting predications 
#get predication of CNN and get class with higher confidence
pred_cnn=F.softmax(p,1).argmax(dim=1)
#find indicices that have negative class (here negative is labeled as (1))
idx_empty=pred_cnn==1
#creeate an empty mask templete 
empty_temp=torch.zeros(256,256)
#replace all the negative cases to empty masks to reduce the flase psoitve rate 
# preds[idx_empty]=empty_temp


# %%
# function gets predictions and applies three different post-processing techniques 
def get_multi_preds(learn,correct=False,idx=None):
    empty_temp=torch.zeros(256,256)

    print('Getting Raw preds')
    preds,ys = learn.get_preds()#need a threshold 
    print('Getting TTA preds')
    preds_tta,_=pred_with_flip(learn)#need a threshold
    print('ARgmaxing ...')
    preds_argmax=preds.argmax(dim=1)
    preds = preds[:,1,...]
    ys = ys.squeeze()
    gc.collect()
    torch.cuda.empty_cache()
    # if its a one case model the use the CNN to filter out the results
    if correct is True and idx is not None:
        preds[idx]=empty_temp
        preds_tta[idx]=empty_temp
        preds_argmax[idx]=empty_temp.long()
    print('Done !')

    return preds,preds_tta,preds_argmax,ys


# %%
# helper functions that produce the confusion matrix results and is used to get the Mean ratio of intersection scores seen in the report 
def generate_confusion(pred:Tensor, y_true:Tensor,learn:Learner):
        "Average and Per Image Confusion: intersection of pixels given a true label, true label sums to 1"
       
        pred_class= pred
        single_img_confusion = []
        mean_confusion = []
        n =  pred_class.shape[0]
        for c_j in range(learn.data.c):
            true_binary = y_true.squeeze(1) == c_j# a inary array representing where if each pixel has the class or not for all the iamges n*128*128 array 
            total_true = true_binary.view(n,-1).sum(dim=1).float()#total number of pixels belonging to a class for each image , its the size of (n)
            for c_i in range(learn.data.c):
                pred_binary = pred_class == c_i
                total_intersect = (true_binary*pred_binary).view(n,-1).sum(dim=1).float()
                p_given_t = (total_intersect / (total_true)) #intersection in each image for each class 
                p_given_t_mean = p_given_t[~torch.isnan(p_given_t)].mean()
                single_img_confusion.append(p_given_t)
                mean_confusion.append(p_given_t_mean)
        single_img_cm = to_np(torch.stack(single_img_confusion).permute(1,0).view(-1, learn.data.c, learn.data.c))
        mean_cm = to_np(torch.tensor(mean_confusion).view(learn.data.c, learn.data.c))
        return mean_cm, single_img_cm
def plot_intersect_cm(self, cm, title="Intersection with Predict given True"):
        "Plot confusion matrices: self.mean_cm or self.single_img_cm generated by `_generate_confusion`"
        from IPython.display import display, HTML
        fig,ax=plt.subplots(1,1,figsize=(10,10))
        im=ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title}")
        ax.set_xticks(range(self.data.c))
        ax.set_yticks(range(self.data.c))
        ax.set_xticklabels(self.data.classes, rotation='vertical')
        ax.set_yticklabels(self.data.classes)
        fig.colorbar(im)
        
        df = (pd.DataFrame([self.data.classes, cm.diagonal()], index=['label', 'score'])
            .T.sort_values('score', ascending=False))
        with pd.option_context('display.max_colwidth', -1):
            display(HTML(df.to_html(index=False)))
        return df


# %%
# fil;e names need to be changed for the model file names 
for i in [1,2,3,4,5]:
    if i == 1:
        learn.load('/cs/home/khfy6uat/bin/models/baseline_unet_fold_0')
    elif i == 2:
        learn.load('/cs/home/khfy6uat/bin/models/baseline_unet_fold_1')
    elif i == 3:
        learn.load('/cs/home/khfy6uat/bin/models/baseline_unet_fold_2')
    elif i == 4:
        learn.load('/cs/home/khfy6uat/bin/models/baseline_unet_fold_3')
    elif i == 5:
        learn.load('/cs/home/khfy6uat/bin/models/baseline_unet_fold_4')
    
    CMS=[]
    dices = []
    dices_ad=[]
    ious=[]
    scores, best_thrs = [],[]
    cm_score=[]
    preds,preds_tta,preds_arg,ys=get_multi_preds(learn,correct=False)
    all_preds = [preds,preds_tta,preds_arg]
    thrs = np.arange(0.01, 1, 0.01)

########################## GETTING RESULTS WITH CONSTANT THRESH
    for preds in all_preds:
        dices = []
        dices_ad=[]
        ious=[]
        CMS=[]

        th=0.3
        preds_m=(preds>th).long()
        CMS.append(CM_overall(learn,preds_m,ys))
        dices.append(dice_overall(preds_m, ys))
        dices_ad.append(dice_overall_adjusted(preds_m, ys))
        ious.append(iou_overall(preds_m, ys))
        print(CMS,dices,dices_ad,ious)


# %%
#for one case models
for i in [1,2,3,4,5]:
    if i == 1:
        learn.load('/cs/home/khfy6uat/bin/models/One_class_unet_fold_0')
    elif i == 2:
        learn.load('/cs/home/khfy6uat/bin/models/One_class_unet_fold_1')
    elif i == 3:
        learn.load('/cs/home/khfy6uat/bin/models/One_class_unet_fold_2')
    elif i == 4:
        learn.load('/cs/home/khfy6uat/bin/models/One_class_unet_fold_3')
    elif i == 5:
        learn.load('/cs/home/khfy6uat/bin/models/One_class_unet_fold_4')
    dices = []
    dices_ad=[]
    ious=[]
    scores, best_thrs = [],[]
    cm_score=[]
    preds,preds_tta,preds_arg,ys=get_multi_preds(learn,correct=True,idx=idx_empty)
    all_preds = [preds,preds_tta,preds_arg]
    thrs = np.arange(0.01, 1, 0.01)

    for preds in all_preds:
        dices = []
        dices_ad=[]
        ious=[]
        CMS=[]

        th=0.3
        preds_m=(preds>th).long()
        CMS.append(CM_overall(learn,preds_m,ys))
        dices.append(dice_overall(preds_m, ys))
        dices_ad.append(dice_overall_adjusted(preds_m, ys))
        ious.append(iou_overall(preds_m, ys))
        print(CMS,dices,dices_ad,ious)


# %%
# Predictions for test set
# preds, _ = learn.get_preds(ds_type=DatasetType.Test)
preds,_=pred_with_flip(learn,ds_type=DatasetType.Test)
preds = (preds>.3).long().numpy()
print(preds.sum())
p,_=learn_cnn.get_preds(ds_type=DatasetType.Test)# getting predications 
#get predication of CNN and get class with higher confidence
pred_cnn=F.softmax(p,1).argmax(dim=1)
#find indicices that have negative class (here negative is labeled as (1)) ONLY FOR ONE CASE MODEL
idx_empty=pred_cnn==1
print(idx_empty.sum(),3250-idx_empty.sum())
preds[idx_empty]=empty_temp
# Generate rle encodings (images are first converted to the original size)
rles = []
for p in progress_bar(preds):
    im = PIL.Image.fromarray((p.T*255).astype(np.uint8)).resize((1024,1024))
    im = np.asarray(im)
    rles.append(mask2rle(im, 1024, 1024))
ids = [o.stem for o in data.test_ds.items]
sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
sub_df.to_csv('Normal_unet_JACC_BN_corrected_f1256.csv', index=False)
sub_df.head()
get_ipython().system('kaggle competitions submit -c siim-acr-pneumothorax-segmentation -f Normal_unet_JACC_BN_corrected_f1256.csv -m "Unet: JACC loss BN , CNN: f1 score 256 "')


# %%


