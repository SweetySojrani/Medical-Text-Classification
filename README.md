# Medical-Text-Classification

## Introduction:
The objective of this assignment is to implement min-epsilon k-nearest neighbor
classifier on medical text classification data provided as train and test file.

## Implement text processing:
Below steps of text processing was performed on both training data and test data
1. Read the train file and test file using pandas. 
2. Create lists of description and class of the file. Now we can remove the unimportant text from both the files using steps from 3 to 6.
3. Convert the data description in training file and test file to lowercase
4. Remove stop words and stemmers using nltk library which will remove words like pronouns, conjunctions, interjections which are unimportant for document comparison.
5. Clean punctuations from both the lists.
6. Remove short words less than length 3.
7. Once the data is clean, I transformed the cleaned train list to sparse matrix and normalized it using TfidfTransformer library functions.
8. This normalized matrix is then split into Training Data(80%) and Test(20%). Let us call the test which is 20% of the data file as validation data.
9. The split of training data will be used to train the KNN algorithm for optimizing the F1 score using required parameter values for the KNN Classifier algorithm. The result of which will be used to identify the F1 score for the validation data since we
have the class labels of it. The process is implemented in the KNN Classifier algorithm.

## Implement KNN Classifier:
1. Calculate Cosine Similarity for entire dataset: The cosine similarity will be calculated on the normalized matrix.
2. Get Nearest Neighbours for Validation: The cosine similarity values will vary from 0 to 1. 1 being the most similar and 0 being the least similar. We would like to predict the class of the descriptions in the validation set as close to 1 as possible. For which the top k values of cosine similarity will be considered for K nearest calculation.
3. KNN Predictions for test validation: To predict the class of the validation description the majority vote of class will be calculated from the chosen k cosine similarities using most common method. However, it is possible that the values of the majority class voted is very low for Eg. 0.02. In such cases the prediction will not be accurate which is why we are considering min-epsilon parameter to skip
the cosine similarities in the top k which are very below the threshold. Eliminating the low similarity values apart from precision also improves the speed of the algorithm. Also, I have considered a case if a training document exactly matches the validation document in which case the similarity will be >0.99. In such case the corresponding class of this train document will be the result of the validation document and the other cosine values will be ignored. Another case would be if none of the K nearest cosine similarity are greater than the min-epsilon. In this case the maximum cosine similarity for that description will be considered. Once
the class prediction is obtained, the output will be used to calculate f1 score using the actual values of validation data. The f1 score will determine if the min-epsilon and k value needs to be changed.
4. KNN prediction for Test file: After the K and min epsilon parameters are optimized. The same KNN classifier algorithm will be executed for test file data. The output obtained from this step will be the final prediction output of the algorithm for the test file.

## Methodology of choosing parameters:
The min-epsilon value and K parameter values were calculated using the test validation data which was driven by splitting in the ratio 80:20 from the train file. This method provides a feedback on precision so as to optimize the values of K and min-epsilon. For my implementation I have considered min-epsilon = 0.1 and K = 34 as my optimized value after rounds of comparing f1 score on parameters selected.

## Run the code:
The libraries required for running the code are numpy, pandas, re for cleaning the punctuations, ‘nltk’ for removing stop words and stemming, Counter for majority vote calculation in KNN, classification_report for calculating the F1 score , CountVectorizer for creating sparse matrix, TfidfTransformer for normalizing the sparse matrix.

## Conclusion: The pre-processing and KNN algorithm is implemented to predict the class
of the medical condition descriptions in the test file which will be used to correctly predict
the diseases of patients. The classifier can be further improved using more methods of
data preprocessing for higher precision.
