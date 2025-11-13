# Image augmentation by a vision-language foundation model for durian leaf disease recognition
The overall methodology of the proposed Foreground-Aware Contrastive Learning for Durian Leaf Disease Detection (FACLDD) follows a two-stage framework, as illustrated below. The first stage involves unsupervised pre-training using the proposed Foreground-Aware Contrastive Learning (FACL) framework to learn foreground-sensitive leaf representations from unlabeled orchard images under complex environmental conditions. The second stage performs supervised fine-tuning for durian leaf disease detection, where the pre-trained backbone is transferred to a YOLO-based detector to learn disease-specific lesion patterns. This two-stage design enables the model to extract robust, leaf-oriented features while mitigating the interference of background variability.
<img width="708" height="566" alt="image" src="https://github.com/user-attachments/assets/a05f9758-fefd-41aa-9162-70b482c2bc2d" />

Fig.1 Overall framework of FACLDD
# Experimental Results
<img width="792" height="440" alt="image" src="https://github.com/user-attachments/assets/f6103d75-af70-4836-98d1-40bd6311d51d" />

Fig.2 Multi-scale visualization of the proposed FACLDD framework

<img width="782" height="422" alt="image" src="https://github.com/user-attachments/assets/e67a3eac-12ef-47fe-b6a2-a17b73e843bb" />
<img width="832" height="206" alt="image" src="https://github.com/user-attachments/assets/7065682c-6dd7-4b10-a4dd-0371379dea47" />
<img width="830" height="192" alt="image" src="https://github.com/user-attachments/assets/47873462-7eec-418e-93d1-f9a4dcc84ec0" />

Fig.3 FACLDD Detection performance of a visualization ,b on DLDD vs CDDD, and c bad case

# Datasets
Table 1 Overview of the datasets used in this study

<img width="520" height="89" alt="image" src="https://github.com/user-attachments/assets/d4e16cf1-e68c-4d72-969d-314710b87c6e" />

<img width="830" height="342" alt="image" src="https://github.com/user-attachments/assets/2fc5fdc3-7506-48e0-b15b-1a5b79e4b892" />

Fig.4 Overview of a orchard locations in Penang, and b Datasets samples

# Developing Environments
Pytyon

PyTorch

huggingface-hub

transformers

OpenCV

numpy,sklearn,scipy,six,PIL,matplotlib,seaborn,mkl

albumentations, grad-cam,plantcv,umap-learn, ultralytics


# Technologies Used
Computer Vision
SimCLRv2
