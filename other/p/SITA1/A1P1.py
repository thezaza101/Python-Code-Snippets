import pandas as pd

'''The first task is to read the json file as a Pandas DataFrame and delete the rows
which contain invalid values in the attributes of “points” and “price”.'''
df = pd.read_json('datasets//wine.json')
df = df.dropna(subset=['points', 'price'])

'''what are the 10 varieties of wine which receives the highest number of reviews?'''
dfTop10MostReviews = df['variety'].value_counts()[:10]

'''which varieties of wine having the average price less than 20, with the average points at least 90?'''
averagePoints = df.groupby('variety', as_index=False)['points'].mean()
averagePoints = averagePoints.loc[averagePoints['points']>=90]
averagePrice = df.groupby('variety', as_index=False)['price'].mean()
averagePrice = averagePrice.loc[averagePrice['price']<20]
q2 = pd.merge(averagePrice, averagePoints, on='variety')

'''
In addition, you need to group all reviews by different countries and generate a statistic
table, and save as a csv file named “statisticByState.csv”. The table must have four
columns:
Country – listing the unique country name.
Variety – listing the varieties receiving the most reviews in that country.
AvgPoint – listing the average point (rounded to 2 decimal places) of wine in that
country
AvgPrice – listing the average price (rounded to 2 decimal places) of wine in that country
'''

countryList = df['country'].drop_duplicates().to_frame()
dfTopReviews = df.groupby('country')['variety'].value_counts()
dfTopReviews = dfTopReviews.to_frame()
dfTopReviews.columns = ['Var_count']
dfTopReviews = dfTopReviews.reset_index(inplace=False)  
dfTopReviews = dfTopReviews.set_index(['country', 'variety'],drop=False, inplace=False)
dfTopReviews = dfTopReviews.drop_duplicates(subset='country', keep='first', inplace=False)

averagePointsCt = df.groupby('country', as_index=False)['points'].mean().round(2)
averagePriceCt = df.groupby('country', as_index=False)['price'].mean().round(2)

ss = pd.merge(countryList,dfTopReviews,on='country')
ss = pd.merge(ss,averagePointsCt,on='country')
ss = pd.merge(ss,averagePriceCt,on='country')
ss = ss[['country','variety','points','price']]
ss.to_csv('datasets//StatisticByStateSP.csv')

