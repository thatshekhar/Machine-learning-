from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = df[['Hours','Sleep']]
y = df['Marks']

poly = PolynomialFeatures(degree=2,interaction_only=True)

X_interaction = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_interaction,y)

print(model.predict(poly.transform([[3,7]])))
