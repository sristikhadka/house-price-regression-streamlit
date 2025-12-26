import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


df = pd.read_csv("house_price_regression_dataset.csv")
print(df.head())



X = df[['Square_Footage','Num_Bedrooms','Num_Bathrooms','Lot_Size','Garage_Size','Neighborhood_Quality']]
y = df['House_Price']

# print(df.describe())
st.header('House Price Regression')

st.sidebar.header('Housing Option')
Year_filter = st.sidebar.multiselect('select year:',options= df['Year_Built'].unique(),default = df['Year_Built'].unique())


df_data = df[df['Year_Built'].isin (Year_filter)]
st.subheader('house data')
st.dataframe(df_data)

st.subheader('Square_Footage vs House_Price')
fig,ax = plt.subplots()

ax.scatter(df_data['Square_Footage'],df_data['House_Price'],alpha = 0.6,label = 'Houses')

ax.set_xlabel('Square_Footage')
ax.set_ylabel('House_Price')
ax.set_title('Scatter Plot')
ax.legend()
st.pyplot(fig)



st.subheader('Average Price by Neighborhood')
avg_price = df_data.groupby('Neighborhood_Quality')['House_Price'].mean()
st.bar_chart(avg_price,x_label= 'Neighborhood Quality',y_label= 'House Price')



plt.figure(figsize= (8,5))
plt.boxplot([df_data['Square_Footage'],df_data['Lot_Size'],df_data['Garage_Size']],
         tick_labels = ['Square_Footage','Lot_Size','Garage_Size'])
plt.title('Boxplot')
st.pyplot(plt)

st.subheader('Correlation Heatmap')
corr_data = df_data[['Square_Footage',
                    'Num_Bedrooms',
                    'Num_Bathrooms',
                    'Lot_Size',
                    'Garage_Size',
                    'Neighborhood_Quality']
                    ]
corr_matrix = corr_data.corr()
fig3,ax3 = plt.subplots(figsize = (8,6))
sns.heatmap(corr_matrix,annot = True,cmap = 'coolwarm',fmt = '.2f',linewidths = 0.5,ax = ax3)
ax3.set_title('Feature Corelation Heatmap')
st.pyplot(fig3)





X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state= 42)
              
models = {
    'knr':KNeighborsRegressor(),
    'svm':SVR(),
    'LR':LinearRegression()
    
}
results = {}
for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    results[name] = r2
    print(name,'r2_score:',r2)

    

plt.figure(figsize=(6,4))
plt.bar(results.keys(),results.values())
plt.xlabel('models')
plt.ylabel('r2 square')
plt.title('model comparison: models vs root r2 square')
st.pyplot(plt)

param_grid = {
    'n_neighbors' : [3,5,7,9]
}

grid = GridSearchCV(KNeighborsRegressor(),
                    param_grid,
                    cv= 5,
                    scoring= 'r2')

grid.fit(X_train,y_train)
best_model = grid.best_estimator_
st.write("Best Model:",grid.best_params_)

st.sidebar.header("Enter House Details")
sqft = st.sidebar.number_input("Square Footage",500,5000)
bed = st.sidebar.number_input('Bedrooms',1,5)

bath = st.sidebar.number_input('Bathroom',1,5)
ls = st.sidebar.number_input('lot size',min_value=0.1,max_value=5.0,format='%.4f')
gs = st.sidebar.number_input('Garage Size',0,2)
neigh = st.sidebar.number_input('neighboor',1,10)

input_data = [[sqft,bed,bath,ls,gs,neigh]]

if st.sidebar.button('Predict Price'):
    prediction = best_model.predict(input_data)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")


