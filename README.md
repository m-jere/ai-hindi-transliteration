## Hindi to English Transliteration Model
This model is based off of the project done during the school year, where we improved upon the HathiTrust system by organizing the metadata. However, we failed to address the shortcomings in translation issues, since many works on the platform deal with non-romanized languages. If a work has been transliterated differently or incorrectly, then further duplicate works will remain in the library, which will continue to impede the user search process. Thus, by improving the transliteration methodology, HathiTrust can be improved upon and used as a resource amongst students and academics alike.

## Dataset
We wanted to introduce a way to transliterate works in different languages to hopefully identify more duplicate works that we had missed originally. We will be transliterating Hindi language words, using these pairs to train the AI model, and see how effective it is at romanizing Hindi. Our dataset is built from a list of 10000 common hindi words, which is then passed through an algorithm that transliterates them into English letters. 

## Model Architecture
The transliteration model is based on a sequence-to-sequence architecture utilizing an encoder-decoder framework. The encoder processes the input Hindi sequence, and the decoder produces the corresponding English transliteration. The model incorporates recurrent neural network (RNN) layers, like LSTM or GRU, which are well-suited for handling sequence modeling tasks. This project uses one-hot-encoding, where each character in the input sequence is represented as a vector of binary numbers. The network then processes and learns from the sequences of characters. 

## Training
The model is trained using the training dataset, where the input sequences are the Hindi words and the target sequences are the English transliterations. The training process involves feeding the input sequences into the model, comparing the generated output with the target sequences, and adjusting the model's parameters to minimize the loss. Loss here refers to a measure of how far the model's predictions are from the actual target values. It quantifies the error between the predicted output and the true output.

## Evaluation
The model's performance is evaluated using a separate validation dataset. In this project, 20% of the input data is set aside for this dataset. The evaluation metrics include loss and accuracy, which measure the model's ability to generate accurate transliterations. The validation dataset provides an estimate of the model’s ability to generalize to new, unseen data. This helps in understanding whether the model is just memorizing the training data (overfitting) or if it's learning patterns that apply broadly. 
# Validation Dataset
The validation dataset is used to assess the model's generalization and to tune hyperparameters based on the accuracy, such as the learning rate or the size of the hidden layers. The learning rate controls how quickly or slowly a model adjusts its parameters in response to the computed loss. A learning rate that's too high might cause the model to miss the optimal solution, while one that's too low might result in excessively slow training. The number of units in the hidden layers affects the model’s capacity to learn complex patterns. Too few units might make the model too simple (underfitting), while too many might lead to overfitting. Tuning both the learning rate and hidden layers improves the model and the learning process of the algorithm.

## Inference
Once the model is trained, it can be used to generate English transliterations for new Hindi words. The input Hindi word is passed through the trained model, and the decoder generates the corresponding English transliteration character by character. The generated output is the predicted English transliteration.


# Steps of Model 
To use the transliteration model, follow these steps:
1. Prepare the dataset: Use the list of common hindi words and transliterate them to get romanizations. These romanizations use the Library of Congress Hindi Romanization Table character-to-character conversions
2. Train the model: Use the training dataset to train the transliteration model. Adjust the hyperparameters and experiment with different architectures, if necessary.
3. Evaluate the model: Measure the performance of the trained model using the validation dataset. Monitor the loss and accuracy to assess the model's quality.
4. Save the trained model: Save the trained model's parameters and architecture to a file (e.g., model.keras in this case). Examine to evaluate the current accuracy
5. Predict transliterations: Use the loaded model to predict English transliterations for new Hindi words. Pass the Hindi word through the model and obtain the generated transliteration. Compare this with the transliteration function. 

# Tools
* Pandas
* TensorFlow
* Keras API
* [Library of Congress](https://github.com/eymenefealtun/all-words-in-all-languages/blob/main/Hindi/Hindi.txt)
* [Reference Project](https://github.com/roshancharlie/Hindi-To-English-Transliteration-Model/tree/main)
* [List of Hindi Words](https://www.loc.gov/catdir/cpso/romanization/hindi.pdf)
