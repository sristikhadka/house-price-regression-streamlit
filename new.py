import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


df = pd.read_csv("house_price_regression_dataset.csv")
# print(df.head())
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


st.subheader('Boxplot:,Square Footage')
fig_sq,ax_sq = plt.subplots()
ax_sq.boxplot(df_data['Square_Footage'])
ax_sq.set_ylabel('Square Footage')
ax_sq.set_xticklabels(['Square_Footage'])
st.pyplot(fig_sq)

st.subheader('Boxplot:,Lot_Size')
fig_ls,ax_ls = plt.subplots()
ax_ls.boxplot(df_data['Lot_Size'])
ax_ls.set_ylabel('Lot_Size')
ax_ls.set_xticklabels(['Lot_Size'])
st.pyplot(fig_ls)

st.subheader('Boxplot:,Garage_Size')
fig_gs,ax_gs = plt.subplots()
ax_gs.boxplot(df_data['Garage_Size'])
ax_gs.set_ylabel('Garage_Size')
ax_gs.set_xticklabels(['Garage_Size'])
st.pyplot(fig_gs)
            

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

st.subheader('Model R2 score comparison')
st.image('r2square.png',use_container_width= True)


st.sidebar.header("Enter House Details")
sqft = st.sidebar.number_input("Square Footage",500,5000)
bed = st.sidebar.number_input('Bedrooms',1,5)

bath = st.sidebar.number_input('Bathroom',1,5)
ls = st.sidebar.number_input('lot size',min_value=0.1,max_value=5.0,format='%.4f')
gs = st.sidebar.number_input('Garage Size',0,2)
neigh = st.sidebar.number_input('neighboor',1,10)

model = joblib.load('House_price_model.pkl')

input_data = [[sqft,bed,bath,ls,gs,neigh]]

if st.sidebar.button('Predict Price'):
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")


