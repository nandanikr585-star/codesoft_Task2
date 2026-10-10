# 🎬 Movie Rating Prediction Using Python

## 📌 Project Overview
This project is developed as part of the **CODSOFT Data Science Internship**. It uses Machine Learning to predict movie ratings based on features such as genre, director, and actors.

## 🎯 Objective
The main objective is to build a Machine Learning model that predicts movie ratings using historical movie data.

## 📂 Dataset
- **Dataset:** IMDb Movies India
- **File:** `IMDb Movies India.csv`
- **Features:** Genre, Director, Actor 1, Actor 2, Actor 3
- **Target Variable:** Rating

Dataset source: [IMDb India Movies Dataset on Kaggle](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Machine Learning
- Random Forest Regression

## ⚙️ Project Workflow
1. Load the movie dataset.
2. Clean the data and handle missing ratings.
3. Select relevant movie features.
4. Encode categorical data.
5. Split the dataset into training and testing sets.
6. Train a Random Forest Regression model.
7. Evaluate model performance using MAE and R² score.
8. Predict a sample movie rating.

## 💻 Installation
Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn
```

## ▶️ How to Run
1. Download the dataset from Kaggle.
2. Place `IMDb Movies India.csv` in the project folder.
3. Run the following command:

```bash
python movie_rating_prediction.py
```

## 📊 Model Evaluation
The model evaluates its predictions using:
- **Mean Absolute Error (MAE)**
- **R² Score**

The actual evaluation results will be displayed in the terminal after running the program.

## 👩‍💻 Project Information
- **Internship:** CODSOFT Data Science Internship
- **Task:** Task 2 – Movie Rating Prediction
- **Language:** Python
- **Project Type:** Machine Learning / Regression

## 📜 Disclaimer
This project is created for educational purposes as part of a Data Science internship task.
