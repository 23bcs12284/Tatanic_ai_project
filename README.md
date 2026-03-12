# 🚢 Titanic Survival AI Predictor

An **AI-powered web application** that predicts whether a Titanic passenger would survive based on passenger attributes using **Machine Learning**.

This project demonstrates how **data preprocessing, machine learning models, and interactive dashboards** can be combined to build a real-world predictive system.

---

# 📌 Project Overview

The Titanic disaster is one of the most famous shipwrecks in history.
Using the Titanic dataset, this project builds a **machine learning model** that predicts passenger survival.

Users can enter passenger details and the system will predict:

* Survival outcome
* Survival probability
* Data insights through visualizations

The model is deployed using an **interactive Streamlit dashboard**.

---

# 🧠 Machine Learning Models

The system trains multiple machine learning models and selects the best one:

* Logistic Regression
* Random Forest Classifier

The best-performing model is automatically selected and used for predictions.

---

# ⚙️ Features

### 🤖 AI Prediction

Predict whether a passenger survived based on input features.

### 📊 Survival Probability

Displays the probability of survival using a **visual gauge meter**.

### 📈 Data Visualization

Interactive charts showing patterns in the Titanic dataset:

* Survival count
* Survival by gender
* Age distribution

### 🧪 Passenger Simulation

Simulate real passenger scenarios to demonstrate AI predictions.

### 🧠 Feature Importance

Visualizes which features influence survival prediction the most.

---

# 🖥️ Application Dashboard

The Streamlit web app allows users to:

1. Enter passenger details
2. Click **Predict Survival**
3. View AI prediction and survival probability
4. Explore dataset insights through charts

---

# 📂 Project Structure

```
titanic-ai-project
│
├── dataset
│     └── titanic.csv
│
├── model
│     └── train_model.py
│
├── utils
│     └── preprocess.py
│
├── app
│     └── app.py
│
├── outputs
│     └── model.pkl
│
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Clone the repository:

```
git clone https://github.com/your-username/titanic-survival-ai.git
```

Move into the project directory:

```
cd titanic-survival-ai
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit app:

```
streamlit run app/app.py
```

The application will open in your browser.

---

# 📊 Dataset

This project uses the **Titanic dataset**, which includes passenger information such as:

* Passenger Class
* Gender
* Age
* Fare
* Number of siblings/spouses
* Number of parents/children
* Embarked port

The dataset is widely used for **machine learning classification problems**.

---

# 📈 Model Performance

Typical model accuracy:

```
~80% – 85%
```

Performance may vary depending on training data splits.

---

# 💡 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Plotly

---

# 🏆 Hackathon Demonstration

This project was built as an **AI-powered predictive dashboard** demonstrating:

* Machine learning model training
* Data preprocessing pipelines
* Real-time predictions
* Interactive data visualization

---

# 👨‍💻 Author

**Prabhakar Kumar Jha**

Computer Science Student
Chandigarh University

---

# ⭐ Future Improvements

* Deep learning models
* Model deployment on cloud
* Advanced interactive dashboards
* Real-time API integration

---

# 📜 License

This project is for **educational and research purposes**.
