# Churn Prediction
One of the challenges faced by companies that offers contract services, such as banks and telecom, is to prevent their customers from stopping using their service, which is called churn. In this scenario, being able to predict who is likely to cancel their contract can be a valuable tool, as the company can offer discounts and new service conditions in order to keep these users.   

Machine Learning can help solving this problem. Using past data from customers and with the information if they churned or not, it is possible to create a model that can predict if present customers are about to leave the service.   

The goal of this project is to create a churn prediction model for customers of a bank. The [dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling) used contains details of the bank's customers, such as Credit Score, Balance, Age and Location, and whether or not they've closed their account or continued to use the service. The project goes through a detailed analysis that tries to identify trends and other useful information from the data, followed by a study of the machine learning algorithms that can offer the best solutions for this problem.  

## Instructions
The churn_model.ipynb file contains: 
- Data preparation and data cleaning;
- EDA, feature importance analysis;
- Model selection process and parameter tuning.

### Installing dependencies
To reproduce this project on your system, you need to download the code and install the dependencies using pipenv. 

If you don't have pipenv installed, you can just install it using the following command:
```
pip install pipenv
```
With pipenv, you can create a virtual environment with the needed packages installed. All you need to do is use the following command while in the project folder:
```
pipenv install
```
When all the dependencies are installed, the virtual environment can be activated with:
```
pipenv shell
```
Then, you can run the train.py to create a file contaning the saved model:
```
python train.py
```
To host the service locally, just run the predict.py file:
```
python predict.py
```
### Making a prediction
Finally, you can use the model to make a prediction by creating a script/notebook with the example code below. Pay attention for the type of each sample. The churn_request.ipynb contains this code.
```python
import requests

customer = {
    'CreditScore': int, 
    'Geography': str, # 'Spain', 'Germany' of 'France'
    'Gender': str, 
    'Age': int, 
    'Tenure': int, 
    'NumOfProducts': int, 
    'IsActiveMember': int, 
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=vehicle)
result = response.json()
print(result)
```

The churn_prediction.ipynb contains information about the possible entries for each feature. You can also find this same info in the canadian government [website](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64). 

### Deployment using Docker
You can deploy this file using docker. First, you need to create a docker image using the Dockerfile. This can be done by running this command in the project folder:
```
docker build -t churn_prediction .
```
Now, just use the command below and the model will be served in your local host, and you can make a prediction in the same way as explained above. 
```
docker run -it -p 9696:9696 churn_prediction:latest
```
