# Recipe Suggestion
Detects ingredients in food images using deep learning. Suggests recipes based on detected ingredients matched with a database of 100K+ recipes. 

## Project goal/Motivation
Food waste is a significant global issue, with an estimated one-third of all food produced worldwide being wasted. This waste occurs at various stages, from production and transportation to retail and consumer levels. At the household level, a significant portion of food waste arises due to poor planning, over-purchasing, and a lack of knowledge on how to utilize available ingredients effectively. Additionally, many people enjoy cooking but often struggle with finding inspiration or new recipes to try, leading them to rely on takeout or prepared foods, which can contribute to unhealthy eating habits and further waste. By leveraging machine learning and computer vision, the project aims to create a tool that can reduce food faste, enhance the cooking experience and promote healthy eating.

## Data Collection
For my project, I gathered data from Roboflow, which included TXT annotations and a YAML configuration file for use with YOLOv8. I reviewed the data to ensure it was accurate and cleaned it up where needed. This involved checking the annotations and making any necessary corrections to improve the dataset's quality for both training and validation.

## Modeling
For the modeling component, I experimented with various versions of the YOLOv8 CNN, including nano, small, medium, and large models. I also tested different batch sizes and early stopping methods, using optimizers such as Adam, AdamW, and SGD. The best results were achieved with the YOLOv8 large model, a batch size of 16, and the AdamW optimizer with initial learning rate of 0.00008 and momentum of 0.9. The training process took approximately 6 hours, utilizing Python 3.10.13 and YOLO from Ultralytics.

Original Dataset: https://universe.roboflow.com/food-recipe-ingredient-images-0gnku/food-ingredients-dataset/dataset/4

Dataset on Kaggle: https://www.kaggle.com/datasets/gitspenv/food-images/data

Ultralytics: https://github.com/ultralytics/ultralytics

The notebook that was used to train the model is in this repository. Import it to Kaggle to try it yourself or simply follow this link: https://www.kaggle.com/code/gitspenv/recipe-suggestion (Note: To run a notebook on Kaggle and utilize the available accelerators, you need to have an account.)

## Model Evaluation

- **Losses**:
  - Box Loss: 0.37159
  - Class Loss: 0.23461
  - DFL Loss: 0.96656

- **Learning Rate**:
  - pg0: 0.0000024138
  - pg1: 0.0000024138
  - pg2: 0.0000024138

- **Metrics**:
  - Precision (B): 0.6745
  - Recall (B): 0.5495
  - mAP50 (B): 0.6121
  - mAP50-95 (B): 0.3750

- **Model Efficiency**:
  - Parameters: 43,722,360
  - GFLOPs: 165.915
  - Speed (PyTorch): 17.063 ms

The train box loss is 0.3716, train classification loss is 0.2346, and train DFL loss is 0.9666, indicating a reasonably well-performing model during training. The precision score of 0.6745 and recall score of 0.5495 suggest that the model balances true positives and false negatives effectively. The mAP50 of 0.6121 and mAP50-95 of 0.3750 illustrate the model's effectiveness at various intersection-over-union thresholds. The AdamW optimizer presumably performed better than SGD due to its adaptive learning rate mechanism, better handling of non-stationary objectives, faster convergence and robustness to hyperparameters.

However, higher validation losses (box loss of 1.4717, classification loss of 1.5041, and DFL loss of 2.3609) indicate possible overfitting. The model was tested with several out-of-sample images, showing overall good performance but occasionally misclassifying round white objects as eggs or garlic. This suggests potential bias in the training data or the need for more diverse samples.

The overall impression from the human evaluators was that the model provided good classifications and suggestions. However, due to the high recall, many objects were not detected in the images, leaving some out. The model performs well but could benefit from fine-tuning and addressing validation losses and misclassification issues for improved robustness and accuracy.

## How To Run The App Locally

To run the app locally, first clone the project repository to your local machine. Once cloned, navigate to the project directory and create a virtual environment. For this project, version 3.12.1 of Python was used, but other versions may also work. After activating the virtual environment, execute the command streamlit run app.py in the terminal. This will launch the application, and the page will automatically open in your default web browser. To test the model, input test images into the input\test directory. There are already some out-of-sample images in this directory that were used to test the model.
