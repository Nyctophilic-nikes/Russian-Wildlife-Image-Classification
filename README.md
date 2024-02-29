## Image Classification Project Description

### 1. Dataset Preparation and Visualization
#### (a) Data Collection and Splitting
https://drive.google.com/drive/folders/1f0rnpZNPfoZGwTlZroaxBVJTf9EtdNNc

In this image classification project, we utilize the Russian Wildlife Dataset. The dataset is categorized into ten classes: 'amur leopard', 'amur tiger', 'birds', 'black bear', 'brown bear', 'dog', 'roe deer', 'sika deer', 'wild boar', and 'people'. The class labels are mapped as follows: {'amur leopard': 0, 'amur tiger': 1, 'birds': 2, 'black bear': 3, 'brown bear': 4, 'dog': 5, 'roe deer': 6, 'sika deer': 7, 'wild boar': 8, 'people': 9}. To ensure balanced representation, a stratified random split is performed with ratios of 0.7 for training, 0.1 for validation, and 0.2 for testing.

#### (b) Custom Dataset Class and WandB Initialization
A custom PyTorch Dataset class is created to facilitate efficient loading of the dataset. Additionally, Weights & Biases (WandB) is initialized for logging and tracking the training process. This enables seamless monitoring of key metrics during model training.

#### (c) Data Distribution Visualization
To gain insights into the dataset, visualizations of the data distribution across class labels for both training and validation sets are provided. Understanding the distribution aids in assessing potential biases and challenges during the training process.

![image](https://github.com/Nyctophilic-nikes/Russian-Wildlife-Image-Classification/assets/107383684/ce9bf935-76c2-49fe-8f2a-d1cf5412f2b1)

![image](https://github.com/Nyctophilic-nikes/Russian-Wildlife-Image-Classification/assets/107383684/b33e0154-bdb2-498e-89b5-58de85b6f129)

### 2. CNN Model Training
#### (a) CNN Architecture Design
The Convolutional Neural Network (CNN) architecture is designed with three convolution layers. The kernel size is set to 3x3 with padding and stride of 1. The number of feature maps increases progressively to 32, 64, and 128 for the respective layers. Max pooling layers are applied with varying kernel sizes and strides, followed by flattening the output for the classification head. ReLU activation functions are incorporated where applicable.

#### (b) Model Training
The CNN model is trained using the Cross-Entropy Loss and the Adam optimizer for 10 epochs. WandB is employed to log training and validation losses, as well as accuracies, providing a comprehensive view of the model's learning progress.

<img width="871" alt="image" src="https://github.com/Nyctophilic-nikes/Russian-Wildlife-Image-Classification/assets/107383684/4d3800c5-e22f-4583-b4ec-27a17c5910aa">


#### (c) Overfitting Analysis
The training and validation loss plots are examined to assess potential overfitting. Insights into the model's generalization capabilities are derived from the trends observed in these plots.

#### (d) Performance Evaluation
The model's performance is evaluated on the test set, reporting accuracy, F1-score, and generating a confusion matrix using WandB. These metrics offer a detailed understanding of the model's classification capabilities.

<img width="712" alt="image" src="https://github.com/Nyctophilic-nikes/Russian-Wildlife-Image-Classification/assets/107383684/c1fe096e-04a2-47da-9bac-79268c700355">


#### (e) Misclassification Analysis
As a post-evaluation step, the project delves into misclassifications on the test set. For each misclassified class, three example images are visualized along with the predicted labels. An analysis is provided, exploring potential reasons for misclassification and proposing workarounds to enhance model robustness.

This comprehensive image classification project encompasses dataset preparation, model training, evaluation, and post-analysis, offering a thorough exploration of the model's performance and insights into potential improvements.

## Fine-tuning Pretrained Model and Data Augmentation

### 3. Fine-tuning a Pretrained Model
#### (a) Fine-tuning with ResNet-18
This section focuses on training another classifier using a fine-tuned ResNet-18, a pre-trained model on ImageNet. The training strategy follows the methodology outlined in Question 1.2.(b). WandB is once again employed for comprehensive logging of loss and accuracy during the fine-tuning process.

<img width="874" alt="image" src="https://github.com/Nyctophilic-nikes/Russian-Wildlife-Image-Classification/assets/107383684/4c13b236-2eef-48b6-b98e-e9359100b7c1">


#### (b) Overfitting Analysis
Similar to the analysis in Question 2.(c), the training and validation loss plots are examined to determine whether the fine-tuned model is prone to overfitting. This insight helps in understanding the generalization capabilities of the model.

#### (c) Performance Evaluation
Accuracy and F1-Score on the test set are reported, accompanied by a confusion matrix logged using WandB. These metrics provide a holistic view of the fine-tuned model's performance and its ability to accurately classify the images in the dataset.

<img width="779" alt="image" src="https://github.com/Nyctophilic-nikes/Russian-Wildlife-Image-Classification/assets/107383684/f7543f39-d21b-44cd-b2ed-1e6b00dd6a44">


#### (d) Feature Space Visualization
The backbone of deep neural networks plays a crucial role in extracting feature representations. In this case, the ResNet-18 backbone is used to obtain feature vectors from training and validation sets. These feature vectors are visualized in a 2-D space using t-Distributed Stochastic Neighbor Embedding (tSNE) plots. Additionally, the tSNE plot of the validation set is extended to a 3D space for a more intricate view of the feature distribution.

### 4. Data Augmentation Techniques
#### (a) Introduction to Data Augmentation
In this section, three or more data augmentation techniques suitable for the image classification problem are introduced. Data augmentation is a pivotal step in enhancing model robustness by synthetically expanding the training dataset, allowing the model to train on a more diverse set of samples.

![image](https://github.com/Nyctophilic-nikes/Russian-Wildlife-Image-Classification/assets/107383684/768f2d11-cdea-4cd4-938a-8387166cd973)



#### (b) Model Training with Augmented Data
Following the methodology from Question 1.3.(a), the model is trained using the augmented dataset. The same metrics as in previous sections - training and validation loss plots, accuracy, F1-Score, and confusion matrix - are logged using WandB.

#### (c) Overfitting Resolution
An analysis of the training and validation loss plots is conducted to determine if the implemented data augmentation techniques effectively mitigate the problem of overfitting. This step provides insights into the impact of data augmentation on the model's generalization capabilities.

#### (d) Performance Evaluation with Augmentation
Finally, accuracy, F1-Score, and the confusion matrix on the test set are reported for the model trained with augmented data. These metrics offer a comparison to the performance of the models trained without augmentation, demonstrating the effectiveness of the augmentation techniques in improving classification accuracy.

This section of the project encapsulates the exploration of fine-tuning a pretrained model, visualizing feature spaces, and implementing data augmentation techniques to enhance the robustness of the image classification models. The thorough evaluation and analysis provide valuable insights for model selection and improvement strategies.
