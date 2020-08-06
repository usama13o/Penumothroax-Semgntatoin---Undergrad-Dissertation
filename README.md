
# Penumothroax-Semgntatoin---Undergrad-Dissertation
Neural Networks have revolutionised the healthcare industry by introducing Deep Learning models that reach human-level accuracy. 
In this work, a pipeline implementation is explored that tries to enhance the accuracy of Deep Learning models in the detection and segmentation of pneumothorax from a set of chest 
radiographs. This is done in the hopes of facilitating future applications in the medical field. 
By cascading a CNN and a U-Net, we ensure that the CNN will filter out all the cases
with no pathology, i.e.’Normal’ cases, leaving the U-Net (trained on positive cases only) to focus on arranging a set of filters for the separation of pathology from the infected 
lungs. The models are cross validated on the same data to ensure consistency. The proposed method demonstrates higher accuracy in both segmenting and detecting the pathology. 
By implementing it with more complex architectures and integrating in the domain knowledge of radiologists, this methods can be applied in conjunction with other applications to rapidly triage and prioritise cases for the presence of anomalies.

## Getting Started

I have included the notebook files that i used for this project. They contian all the code used for training and viewing the results.


### Prerequisites
The dataset has to be downloaded to view any results, it can downloaded at the followingn link:

https://www.kaggle.com/iafoss/siimacr-pneumothorax-segmentation-data-256

## files included are:

'Getting_results_kernel' notebook can be used to load and run the models

'make-pneumothorax-classifer-data' , 'make-pneumothorax-oneCase-data' these two are used to create 2 more datasets for classifier and one case model

'pneumothorax fastai 5-fold U-Net 128x128 (1)' runs training for segmentation models

'pneumothorax-classifier' runs training for classifier model

**Folder 'python files' contains the raw python files in case there isn't access to juptyer environment.

*** Folder 'some results' contains some graphs that were exracted from the notebooks to show results of training.

## Authors

* **Usama Zidan** - *Initial work* - [usama Zidan](https://github.com/usama13o/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


