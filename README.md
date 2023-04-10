# Dcard Article Like Prediction

## Reproducing my Prediction

### Environment Setup & Installation

Python version :```3.10.8```

All the packages that should be installed are in ```requirements.txt```

Or you can build the virtual environment via   
```$ virtualenv -p <path to python version> myenv```  
```$ pip install -r requirements.txt```

## Download the code and some files
Download the files below:  
```models.py```  
```evaluate.py```  
```MLP_train.ipynb```  
```CNN_train.ipynb```  
```best_model.pt```   
```intern_homework_train_dataset.csv```   
```intern_homework_public_test_dataset.csv```    
```intern_homework_private_test_dataset.csv```    
```intern_homework_example_result.csv```    
After downloading the files above, put them in the same folder.  
Or you can just download the whole folder in the repositories.   

## Training Code
There are two files of training code: ```MLP_train.ipynb``` and ```CNN_train.ipynb```.  
They are used to train by MLP and CNN respectively.  
You can adjust the hyperparameters to train and store the model.  

## Pre-trained Model
In ```evaluate.py```, you can produce the prediction through the pre-trained model ```best_model.pt``` or you can also train your    
own model and replace it.

