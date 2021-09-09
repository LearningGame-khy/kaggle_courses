import pandas as pd

melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# print(melbourne_data.columns)

melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
# print(X.describe())
# print(X.head())

from sklearn.tree import DecisionTreeRegressor

# 모델 정의. random_state를 지정해서 항상 같은 결과가 나올 수 있도록 설정
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)

print('예측을 진행할 집')
print(X.head())
print('예측결과는')
print(melbourne_model.predict(X.head()))
