# 1) Design mode (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#  - forward pass: compute prediction
#  - backward pass: gradients
#  - update wtights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

### When you set random_state to an integer (e.g., random_state=0), you fix the seed for the random number generator. This means that 
# the same sequence of random numbers will be generated each time you run the code with the same random_state value. As a result, 
# the same train-test split will be produced, allowing for reproducible results. In the examples provided, random_state=0 is 
# used to ensure that the same split is generated consistently. If you were to change random_state to a different value, 
# say 42, a different split would be produced.

# scale

sc = StandardScaler() # This will make our features to have zero mean and unit variance. This is always recoommended to do when we 
# want to deal with logistic regression. 

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) model
# f = wx + b, sigmoid at the end

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_precicted = model(X_train)
    loss = criterion(y_precicted, y_train)

    # backward pass
    loss.backward()

    # updates 
    optimizer.step()

    # Zero gradients
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch + 1}, loss = {loss.item():.4f} ")

with torch.no_grad():
    y_precicted = model(X_test)
    y_precicted_cls = y_precicted.round()
    acc = y_precicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy = {acc:.4f}")