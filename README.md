# Breast Cancer Classification
Our task is to classify tumors into malignant or benign tumors using features of pain from several cell images.  
The dataset, which is from Scikit - Learn, contains real images indicating that the cancer is malignant or benign, and we extract 30 features which are indicating for instance:
* Radius
* Texture
* Perimeter
* Area
* Smoothness


These features are feeded into our classifier. This will give us two target classes: malignant (0) and benign (1), respectively. 
Datasets are linearly separable using all 30 input features:
* Number of Instances: 569
* Class Distribution: 212 Malignant, 357 Benign
More info on the dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

# Result
As we can see from the confusion matrix in the 'Plots' folder, SVM was able to classify tumors into Malignant/benign with 98% accuracy. Since we have zero Type-II errors, we can therefore conclude that this technique can rapidly evaluate breast masses and classify them in an automated fashion. However, we can further improve this model by combining Computer Vision to directly classify cancer using tissue images. 

