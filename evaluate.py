import numpy as np
import pandas as pd
from pandas import read_csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim
from models import MLP

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

df_private_test = pd.read_csv('intern_homework_private_test_dataset.csv')

has_update = df_private_test['title'].str.contains('#更新|\(更新\)|\(更\)|#更|2更|更2|更\)|\(更').astype(int)
df_private_test['update'] = has_update

df_private_test['created_at'] = pd.to_datetime(df_private_test['created_at'], format='%Y-%m-%d %H:%M:%S %Z')
df_private_test['hour'] = df_private_test['created_at'].apply(lambda x: x.hour)
df_private_test['weekday'] = df_private_test['created_at'].apply(lambda x: x.weekday)
df_private_test = df_private_test.drop(['author_id', 'forum_id', 'created_at', 'title'], axis=1)
private_test = df_private_test.values

private_test = torch.tensor(private_test, dtype=torch.float32).to(device)
model=torch.load('best_model.pt')
model.to(device)

model.eval()
with torch.no_grad():
    outputs = model(private_test)

predictions = outputs.cpu().numpy()
print(predictions)
print(predictions.shape)

df = pd.read_csv('example_result.csv')
df['like_count_24h'] = predictions
df.to_csv('example_result.csv', index=False)
