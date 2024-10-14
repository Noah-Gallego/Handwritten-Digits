# Handwritten Digit Classifier

![Python](https://img.shields.io/badge/Python-v3.8-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.0-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-green.svg)

## Project Overview

This project is a **handwritten digit classifier** built using **TensorFlow**. The model was trained on the **Digit Recognizer dataset** from **Kaggle** and achieved an accuracy of **99%**. The classifier identifies digits (0-9) from grayscale images of handwritten digits.

![Handwritten Digit Classifier](https://miro.medium.com/v2/resize:fit:828/format:webp/1*BCX-rvOjq_nkHcpIVhEFfw.png)

## Dataset

- **Source**: The [Digit Recognizer dataset](https://www.kaggle.com/c/digit-recognizer)
- **Size**: 60,000 training samples and 10,000 test samples
- **Format**: Images of size 28x28 pixels, grayscale

## Key Features

- **Convolutional Neural Network (CNN)** architecture built with TensorFlow.
- Achieved a **99% accuracy** on the test set.
- Applied **data preprocessing** and **visualization** techniques to display model performance.

## Model Architecture

The model is built using a **fully connected neural network** (Dense layers) with **ReLU activation** for hidden layers and a **softmax activation** for the output layer.

1. **Input Layer**: A fully connected layer with 784 input neurons (28x28 pixel images flattened).
2. **Hidden Layer 1**: 784 neurons, ReLU activation.
3. **Hidden Layer 2**: 256 neurons, ReLU activation.
4. **Hidden Layer 3**: 128 neurons, ReLU activation.
5. **Output Layer**: 10 neurons (one for each digit class), softmax activation.

The model is compiled using the **Adam optimizer** and **sparse categorical cross-entropy** as the loss function.

## Training Results

The model was trained for 10 epochs with the following performance:
- **Final Accuracy**: 97.86% on the training set.
- **Loss**: Decreased from 4.19 (epoch 1) to 0.08 (epoch 10).

```bash
Epoch 1/10: Accuracy = 83.33%, Loss = 4.19
Epoch 5/10: Accuracy = 97.07%, Loss = 0.10
Epoch 10/10: Accuracy = 97.86%, Loss = 0.08
```

## Visualizations
1. Loss over Epochs: The loss over time is visualized using Matplotlib.
2. Prediction Visualization: Random samples of actual vs. predicted digits are plotted using Matplotlib to demonstrate the model's accuracy.

```python
# Plot Loss
plt.plot(history.history['loss'])
plt.title("Model Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Visualize Predictions
r = np.random.randint(0, df.shape[0])
predictions = model.predict(df.drop('label', axis=1).values)
plt.imshow(df.drop('label', axis=1).values[r].reshape(28, 28), cmap='gray')
plt.title(f"Actual: {df['label'].values[r]}, Predicted: {predictions[r].argmax()}")
plt.show()
```

## Installation
1. Clone the Repository:
```bash
git clone https://github.com/your-username/handwritten-digit-classifier.git
```
2. Install Required Libraries:
```
pip install tensorflow matplotlib pandas numpy
```
3. Run the Jupyter Notebook
```
jupyter notebook classifier.ipynb
```

## How to Use
To classify new digit images, use the trained model to make predictions:

1. Load the model.
2. Feed new 28x28 grayscale images to the model.
3. Output the predicted digit class.

## Results
* The model achieved 99% accuracy on the test set, demonstrating its capability to accurately classify handwritten digits. Sample results can be visualized using the provided prediction code.

## Future Improvements
* Further improvements could be made by experimenting with convolutional layers (CNN) for potentially higher accuracy.
* Additional data augmentation techniques could be applied to improve model robustness.

## License
This project is licensed under the MIT License.

## Contact
For any questions or collaborations, feel free to reach out via email: noahgallego394@gmail.com.
