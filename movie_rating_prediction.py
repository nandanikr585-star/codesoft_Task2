
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# 2. Clean column names and ratings
df.columns = df.columns.str.strip()

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df.dropna(subset=["Rating"])

# 3. Select useful movie features
features = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]

# Check whether required columns exist
features = [col for col in features if col in df.columns]

X = df[features].copy()
y = df["Rating"]

# 4. Handle missing values and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            Pipeline([
                ("fill_missing", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encode", OneHotEncoder(handle_unknown="ignore"))
            ]),
            features
        )
    ]
)

# 5. Create prediction model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train model
print("\nTraining model...")
model.fit(X_train, y_train)

# 8. Evaluate model
predictions = model.predict(X_test)

print("\nModel Evaluation")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

# 9. Predict rating for a sample movie
sample_movie = pd.DataFrame([{
    col: "Unknown" for col in features
}])

if "Genre" in features:
    sample_movie.loc[0, "Genre"] = "Drama"
if "Director" in features:
    sample_movie.loc[0, "Director"] = "Unknown"
if "Actor 1" in features:
    sample_movie.loc[0, "Actor 1"] = "Unknown"
if "Actor 2" in features:
    sample_movie.loc[0, "Actor 2"] = "Unknown"
if "Actor 3" in features:
    sample_movie.loc[0, "Actor 3"] = "Unknown"

predicted_rating = model.predict(sample_movie)[0]

print("\nSample Movie Genre: Drama")
print("Predicted Movie Rating:", round(predicted_rating, 2), "/ 10")
