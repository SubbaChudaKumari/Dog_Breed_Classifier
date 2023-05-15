# Dog_Breed_Classifier

1. Context

This deep learning project puts deep learning concept into practice in particular to image classfication - I am using the classic case of stanford dog breed dataset to build a deep learning model that accurately identifies the breed of the dog, the key challenge for this model is to correctly identify the breed as some breeds have near to identical features of colors and age.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

2. Content and "file name"

- Image classfication model: "1. Dog breed classification.py"
- Requirements file: "2. Requirements.in"
- Model accuracy & loss scores: "3. Dog breed classification - Accuracy & Loss scores of model.xlsx "
- Read me: This is the readme texts
- Dataset (images/ train data/ test data): No. of categories (120) & No. of Images (20,580)


#Posted the link as the file itself is quite large approx 800MB - http://vision.stanford.edu/aditya86/ImageNetDogs/main.html 

#Note to user: If you would like to skip to the accruacy and loss, kindly see the .xlsx file in this repository which shows - The model's accuracy increases while the loss decreases as the epoch unit increases as as the shape shows the model is just right hence underfitting & overfitting is avoided. 


------------------------------------------------------------------------------------------------------------------------------------------------------------------

3. Explanation of the script/ code for myself and any user reading this - Just a quick rundown of the steps on how the code works and what functions/ models are used.

Step 1: Imported all the relevant libraries
Libraries such as numpy, os, interools, random, matplotlib, sklearn, keras (preprocess/ Adam, Global Avg Pool, Dense, Dropout, Inception), tensorflow are imported. 


Step 2: Load the stanford data set 


Step 3: Separate to obtain the dog names from the file name
As the file names are quite long so used 'split' to obtain the later end of the names using 'split' and 'os' - the purpose is to ensure the right no. of breeds & images are captured in value as per the dataset specified from its source.


Step 4: Validate and sense check the no. of breeds & images
As per the source, the split of breeds & images are correctly identified as the source which are 120 breeds & 20580 respectively.


Step 5: Randomly shuffle the data for machine learning
This part helps combine list x and y into tuples. I learned that doing shuffling in the beginning helps "randomly" provide the train and test orders randomly to avoid biasness in results.


Step 6: Display random images of the dogs to see if step 5 worked and they are correctly labelled
This part helps check if the images are really randomly loaded and also to see if they are being correctly labelled as per its original source.


Step 7: Train, test then validate the model
Resized the images so that all the images in the dataset are consistent. Then the images are spilt into test, train and validation set which are further spilt into 20/64/14.

#Note to self: 
1. Calculation of the train & validation set are (Train = 80% of 80% is 64% / Validation = 20% of 80% is 14%)
2. Validation set is important as it helps to avoid overfitting situation and tune the hyperparameters to give us the highest performance rate.


Step 8: Create, train then validate the generators


Step 9: Built the model with the pretained data set
The base model is built using CNN called InceptionV3 - After skimming through various CNN models, Inception seems to be more well suited for image recognition task as the intended projects as per forumns and google.
To further improve performance, a mix combination of techniques such as 1)GlobalAverage Pooling, 2)Dense, and 3)Dropout are used.


Step 10: Train the model and check accuracy
Then the model uses a compilation of Keras model to give a generator-based approach. I learned that the purpose of using a generator based approach is to efficiently train the dqataset so that it can be more memory efficient - also is commonly used for computer vision and NLP projects. And Adam is used for performance optimizer.

