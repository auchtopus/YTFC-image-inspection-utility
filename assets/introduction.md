
This is a lightweight utility for analyzing the results of the YTFC Herbarium Phenology project. This classifies Herbarium Samples pulled from NEVP (New England Vascular Plant) Specimen Databases and the CNH (Consortium of Northeast Herbaria) Databases. 

The objective of this research was to train Convolutional Neural Networks (CNNs) to automatically classify the phenological status of herbarium samples. Four binary classification models were trained that can discern whether a sample is budding, flowering, fruiting, or generally reproductive. This work prioritizes accuracy over recall, so a confidence threshold is used to reject predictions the model has lower confidence in, and is thus less likely to be correct. 

Each CNN was trained on roughly 80 percent of a NEVP database with ~45,000 images, with ~8,400 set aside for use as validation data. 

## Dataset provenance
- *Dataset 1* constitutes the 8,000 images reserved as validation data. It is from the same dataset as the training data, with the same species represented in approximarily similar frequencies. 
- *Dataset 2* represents ~100 images pulled from the CNH database, but with the same taxa as those in Dataset 1 and scored by Patrick Sweeney by hand. 
- *Dataset 3* constitutes ~100 images pulled fro mthe CNH database, but with taxa not included in Dataset 1 or 2. 

## Model Performance Summary

The model performance is summarized here
| Model | Dataset 1 | Dataset 2 | Dataset 3 |
| - | - | - | - |
| Reproductive | ?|? | ?| 
| Flowering | ? | ? | ? |
| Fruiting | ? | ? | ? |
| Budding | ? | ? | ? |