import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
st.set_page_config(page_title="Heart Disease Baseline", page_icon="<3", layout="centered")
st.title("<3 Heart Disease Risk - Baseline (LogReg)")

@st.cache_data
def load_data():
	df = pd.read_csv("data/heart.csv")
	return df

@st.cache_resource
def train_baseline(df):
	X = df.drop("target", axis=1)
	y = df["target"]
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)
	model = LogisticRegression(max_iter=1000, solver="liblinear")
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	metrics = {
		"accuracy": float(accuracy_score(y_test, y_pred)),
		"f1": float(f1_score(y_test, y_pred)),
	}

	return model, X.columns.tolist(), metrics

df = load_data()
model, feature_names, metrics = train_baseline(df)


st.subheader("Baseline Metrics")
st.write(f"Accuracy: **{metrics['accuracy']:.3f}** | F1: **{metrics['f1']:.3f}**")

st.subheader("Try a prediction")
defaults = df[feature_names].median(numeric_only=True).to_dict()

user_vals = {}
cols = st.columns(3)
for i, feat in enumerate(feature_names):
	with cols[i % 3]:
		v = float(st.number_input(feat, value=float(defaults.get(feat, 0.0))))
		user_vals[feat] = v

if st.button("Predict risk"):
	x = np.array([user_vals[f] for f in feature_names]).reshape(1, -1)
	proba = model.predict_proba(x)[0, 1]
	pred = int(proba >= 0.5)
	st.success(f"Risk probability: **{proba:.2%}** -> Prediction: **{pred}** (1=risk)")

