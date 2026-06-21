import pandas as pd

df = pd.read_csv("train.csv")

print("shape: ")
print(df.shape)

print("\ncolumns: ")
print(df.columns)

print("\ninfo: ")
print(df.info())

print("\nmissing values: ")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df = df.drop("Cabin", axis=1)

print("\nmissing values after filling: ")
print(df.isnull().sum())

print("\ntotal passengers: ")
print(len(df))

print("\nMale/female count: ")
print(df["Sex"].value_counts())

print("\naverage age: ")
print(df["Age"].mean())

print("\nhighest fare: ")
print(df["Fare"].max())

print("\ntotal survivors: ")
print(df["Survived"].sum())

print("\nsurvival rate: ")
print(df["Survived"].mean()*100)

print("\nSurvival by Gender:")
print(df.groupby("Sex")["Survived"].mean())

print("\nSurvival by Class:")
print(df.groupby("Pclass")["Survived"].mean())

df.groupby("Sex")["Survived"].mean()

df.groupby("Pclass")["Survived"].mean()

df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

X = df[[
    "Pclass",
    "Sex",
    "Age",
    "Fare"
]]

y = df["Survived"]

print("\nX:")
print(X.head())

print("\ny:")
print(y.head())

df.to_csv(
    "cleaned_titanic.csv",
    index=False
)