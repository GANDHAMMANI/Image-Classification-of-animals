 Implementation of ML model for image classification
Of Dogs, Cats & Snakes
A Project Report
submitted in partial fulfillment of the requirements
of 
AICTE Internship on AI: Transformative Learning 
with 
TechSaksham – A joint CSR initiative of Microsoft & SAP


by

Gandham Mani Saketh
gandhammani2421@gmail.com

Under the Guidance of 
Master Trainer Edunet Foundation
 Abdul Aziz Md


 
 
ACKNOWLEDGEMENT

	 I, Gandham Mani Saketh, would like to take this opportunity to express my heartfelt gratitude to all those who have supported me during the course of this project.
Firstly, I extend my deepest appreciation to my mentor, Abdul Aziz Md, for his invaluable guidance, encouragement, and constructive criticism throughout this journey. His unwavering support and insightful feedback have been instrumental in shaping this project. It has been an honor to work under his mentorship, and his confidence in me has been a constant source of inspiration.
I am also grateful to TechSaksham – A joint CSR initiative of Microsoft & SAP, under which I had the privilege to undertake the AICTE Internship on AI: Transformative Learning. This program provided me with a platform to enhance my skills, gain industry-relevant knowledge, and execute this project successfully.
Finally, I wish to thank my family, friends, and all individuals who contributed directly or indirectly to this project. Your encouragement and belief in my abilities have been invaluable in bringing this work to fruition.
 

ABSTRACT
This project focuses on the development of an intelligent image classification system to identify three specific species: cats, dogs, and snakes. The system leverages state-of-the-art deep learning techniques and models, including CNN, ResNet152, and MobileNetV3Large, to classify input images with high accuracy. Designed for ease of use, the solution is deployed as a Streamlit application that allows users to upload images and obtain predictions along with confidence scores for each class.
The system preprocesses input images, normalizing them to align with the requirements of the trained models. A robust evaluation process ensures that the models deliver reliable predictions. Additionally, measures have been incorporated to alert users if the uploaded image is outside the intended scope, enhancing usability and preventing misclassification.
This project also emphasizes user experience through an intuitive interface and visually appealing outputs, such as dynamically styled tables to display confidence scores and clear messages for prediction results. The solution is ideal for scenarios where identifying and distinguishing between these species is critical, such as pet care, wildlife monitoring, or educational purposes.
Through this project, we aim to demonstrate the practical applications of deep learning in real-world problems while providing a foundation for further enhancements, such as expanding the classification to other species or integrating additional AI functionalities.









TABLE OF CONTENT
Abstract			I	

Chapter 1. 	Introduction	1
1.1		Problem Statement 	1
1.2		Motivation	1
1.3		Objectives	1
1.4.       Scope of the Project	2
Chapter 2. 	Literature Survey	 2-4
Chapter 3. 	Proposed Methodology	4-6
Chapter 4. 	Implementation and Results 	7-9
Chapter 5. 	Discussion and Conclusion 	9-10
References		11



















LIST OF FIGURES
Figure No.	Figure Caption	Page No.
Figure 1		Workflow	5
Figure 2		Main page of the application	7
Figure 3		Uploading an image into the application	8
Figure 4		Results	8





LIST OF TABLES
Table. No.	Table Caption	Page No.
1	 Hardware requirements	6
2	Software requirements	6

 
CHAPTER 1
Introduction

1.1	Problem Statement: 

In recent years, identifying and classifying images accurately has become a crucial requirement in numerous applications, from wildlife monitoring to pet care. Traditional methods struggle with high variability in images, and a dedicated system for classifying specific categories like cats, dogs, and snakes is absent. This project addresses the challenge of creating a robust and scalable image classification system for these three categories.

1.2	Motivation: 

The ability to identify and classify animals accurately is essential for a variety of applications, including pet management, educational purposes, and wildlife conservation. The availability of advanced deep learning frameworks and pre-trained models motivated the development of this project. By combining state-of-the-art algorithms and an intuitive user interface, the project aims to bridge the gap between technology and practical usability in image classification tasks.

1.3	 Collection of the Dataset:
 
It has been collected from the Kaggle platform. The Animal Image Classification Dataset is a comprehensive collection of images tailored for the development and evaluation of machine learning models in the field of computer vision. It contains 3,000 JPG images, carefully segmented into three classes representing common pets and wildlife: cats, dogs, and snakes.

1.4	Objective: 

•  Develop a reliable image classification system using deep learning models.
•  Integrate multiple architectures (CNN, ResNet152, MobileNetV3Large) for comparison and accuracy.
•  Create a user-friendly web application to facilitate predictions.





1.5	 Scope of the project:

The project focuses on classifying three specific species: cats, dogs, and snakes. The scope includes training and evaluating multiple deep learning models, deploying a Streamlit-based application, and ensuring robust handling of edge cases. Future scalability is also considered for incorporating additional species or tasks.


CHAPTER 2
Literature Survey

2.1 Image Classification Overview
Image classification is a core problem in computer vision, involving the categorization of images into predefined labels. Traditional approaches relied on handcrafted features like Scale-Invariant Feature Transform (SIFT) and Histogram of Oriented Gradients (HOG). However, these methods were limited by their inability to generalize across complex datasets with significant variability in object shapes, textures, and lighting conditions.
The advent of deep learning, particularly Convolutional Neural Networks (CNNs), revolutionized image classification. Models such as AlexNet, VGGNet, and ResNet demonstrated remarkable accuracy on benchmark datasets like ImageNet, setting new standards in the field.

2.2 Review of Deep Learning Models
1.	Convolutional Neural Networks (CNNs):

CNNs are the backbone of most image classification tasks due to their ability to capture spatial hierarchies through convolutional layers. Early CNN models like LeNet and AlexNet laid the foundation for modern architectures by demonstrating the efficacy of deep networks in learning features from raw pixel data.
2.	ResNet (Residual Networks):

Introduced by He et al., ResNet uses skip connections to address the problem of vanishing gradients in deep networks. ResNet152, a deeper variant, has shown exceptional performance on large-scale image datasets while maintaining training stability. It is well-suited for complex classification tasks like this project.

3.	MobileNetV3Large:

MobileNet is a family of lightweight neural networks designed for mobile and embedded devices. MobileNetV3 incorporates advanced techniques like Squeeze-and-Excitation (SE) blocks and neural architecture search (NAS) to optimize performance while keeping computational costs low. It provides a good trade-off between accuracy and efficiency for real-time applications.

2.3 Applications of Image Classification

•	Wildlife Conservation: Classifying animals for monitoring and conservation efforts.
•	Pet Management: Assisting in identifying pets for registration and healthcare purposes.
•	Medical Imaging: Detecting anomalies in radiology images.
•	E-commerce: Categorizing products in online retail platforms.

2.4 Challenges in Image Classification
•	Class Imbalance: Many datasets have skewed distributions, which can lead to biased models.
•	Edge Cases: Out-of-scope images (e.g., human photos in this project) can lead to incorrect predictions.
•	Dataset Quality: High-quality datasets are essential for training robust models, but collecting diverse and representative samples is challenging.
•	Overfitting: Deep models risk overfitting on training data if not regularized properly.
•	Computational Resources: Training large models like ResNet152 demands significant computational power.

CHAPTER 3
Proposed Methodology
3.1 System Design
System Diagram:
The proposed system is designed to classify images of animals (cats, dogs, and snakes) using a Convolutional Neural Network (CNN). Below is the flow of the system:
1.	Input Layer: Users upload an image of an animal through the interface.
2.	Preprocessing Layer: The uploaded image undergoes preprocessing techniques like resizing, normalization, and data augmentation (shearing, zooming, flipping) to improve the model's robustness.
3.	CNN Model: The image is passed through a CNN, consisting of convolutional layers for feature extraction, max-pooling layers for dimensionality reduction, fully connected layers for learning non-linear combinations, and an output layer with softmax activation for multi-class classification.
4.	Output Layer: The predicted class label (cat, dog, or snake) and confidence scores are displayed to the user.
5.	Error Handling: If an out-of-scope image (not a cat, dog, or snake) is uploaded, a warning message is shown.




System Diagram Explanation:

                                                                                                                                                                                  



The diagram represents the step-by-step pipeline: 
1.	Data Input: The system takes an image as input through a user-friendly interface (Streamlit or a similar platform).
2.	Preprocessing Block:
o	Resizing: Ensures all images are uniform (224x224).
o	Normalization: Scales pixel values to the range [0, 1].
o	Data Augmentation: Enhances the dataset diversity by applying transformations like rotation, flipping, and zooming.
3.	Classification Model:
o	Convolutional Layers: Extract features such as edges, textures, and shapes.
o	Max-Pooling Layers: Down-sample feature maps, reducing spatial dimensions while retaining significant information.
o	Dense Layers: Learn patterns and associations from extracted features.
4.	Prediction Block: The model assigns a probability to each class and outputs the predicted label with confidence scores.






3.2 Requirement Specification
3.2.1 Hardware Requirements:
T

Component	Specifications
Processor	Intel i5 or higher
RAM	Minimum 8 GB
Storage	Minimum 10 GB free space
GPU (Optional)	NVIDIA GTX 1050 or higher (for faster training)
________________________________________
3.2.2 Software Requirements:

Software/Tool	Description
Operating System	Windows, macOS, or Linux
Programming Language	Python 
Frameworks	TensorFlow, Keras
Data Handling	NumPy, Pandas
Visualization	Matplotlib
IDE/Editor	Jupyter Notebook, VS Code
Deployment Platform	Streamlit (for creating the user interface)
Dataset Management	ImageDataGenerator (for preprocessing)






CHAPTER 4
Implementation and Result

4.1	 Snap Shots of Result:
The following is the main page of the streamlit application with different models in the dropdown of the sidebar






The below figure is the representing the working of the application by uploading an image as an input while using the CNN model.
 
Figure 3: Uploading an image into the application
Finally, the results are shown in the below:
 
Figure-4: Results


4.2	 GitHub Link for Code:

CHAPTER 5
Discussion and Conclusion

5.1	Future Work: 
The project demonstrates a functional and effective approach to classifying images of animals into three categories: cats, dogs, and snakes. However, there are several opportunities for improvement and expansion:
1.	Increase Dataset Diversity:
Collect more diverse and high-quality images to enhance model performance and generalization across various lighting conditions, angles, and backgrounds.
2.	Expand Class Categories:
Include more animal categories or generalize the model to recognize a wider variety of objects.
3.	Enhance Model Architecture:
Experiment with more advanced architectures like EfficientNet or Vision Transformers to further improve accuracy and efficiency.
4.	Incorporate Explainability:
Integrate explainable AI tools like Grad-CAM to visualize which parts of the image the model uses for predictions, improving transparency and user trust.
5.	Edge Deployment:
Optimize the model for deployment on edge devices, such as smartphones or IoT devices, using techniques like model quantization or pruning for better real-time performance.
6.	Error Detection and Recovery:
Develop mechanisms to handle out-of-domain inputs more effectively, such as training the model to recognize and flag non-animal images explicitly.
7.	Integrate Advanced Preprocessing:
Use preprocessing techniques like histogram equalization or CLAHE for better contrast and feature extraction, especially in challenging images.

5.2	Conclusion: 
This project successfully implements a Convolutional Neural Network (CNN) for classifying images of cats, dogs, and snakes with high accuracy. The system combines advanced preprocessing techniques, robust data augmentation, and a carefully designed model architecture to deliver reliable predictions.
The project contributes to understanding practical applications of deep learning in image classification. It demonstrates how preprocessing and model architecture choices can significantly affect performance. The deployment of the system through a user-friendly interface (e.g., Streamlit) makes it accessible to users without technical expertise.
While the current implementation is limited to three animal classes, the methodology can be extended to larger and more complex datasets, proving its scalability and adaptability. The project underscores the importance of combining innovation, technical knowledge, and practical considerations to create solutions that address real-world challenges effectively.
By bridging theoretical concepts with hands-on implementation, this project paves the way for future research and development in image recognition and machine learning systems.





REFERENCES

[1]	Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25, 1097–1105.
[2]	Chollet, F. (2017). Deep Learning with Python. Manning Publications. A comprehensive guide to understanding and implementing deep learning concepts using Python and Keras.
[3]	He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.

[4]	Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
A detailed textbook covering the foundations and applications of deep learning.




