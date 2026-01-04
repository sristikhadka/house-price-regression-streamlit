import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv(r"C:\Users\Admin\Downloads\archive (4)\house_price_regression_dataset.csv")
print(df.head())



X = df[['Square_Footage','Num_Bedrooms','Num_Bathrooms','Lot_Size','Garage_Size','Neighborhood_Quality']]
y = df['House_Price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state= 42)
              
models = {
    'knr':KNeighborsRegressor(),
    'svm':Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())

    ]),
    'LR':LinearRegression()
    
}
results = {}
for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    results[name] = r2
    print(f"{name} R2_score:{r2:.4f}")

plt.figure(figsize=(6,4))
plt.plot(list(results.keys()),list(results.values()),marker = 'o')
# plt.axhline(0,color = 'black')
plt.ylim(0.8,1.0)
plt.xlabel('models')
plt.ylabel('R2 square')
plt.title('model comparison: models vs root r2 square')

plt.grid(True)

for i,v in enumerate(results.values()):
    plt.text(i,v,f"{v:.3f}",ha = 'center',va = 'bottom')
    
plt.savefig(r'c:\Users\Admin\Desktop\r2square.png',dpi = 150)
# plt.show()
    



joblib.dump(models['LR'],'House_price_model.pkl')
print('Model saved successfully')