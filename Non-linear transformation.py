from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(df[['Hours']])

model = LinearRegression()
model.fit(X_poly,df['Marks'])

print(model.predict(poly.transform([[6]])))
