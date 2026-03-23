<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=200&section=header&text=No-Code%20ML%20Model%20Builder&fontSize=40&fontColor=fff&fontAlignY=42&desc=Upload%20CSV%20%C2%B7%20Select%20Model%20%C2%B7%20Train%20%C2%B7%20Predict%20%E2%80%94%20No%20Code%20Required&descAlignY=65&descSize=14&descColor=fff&animation=twinkling" width="100%"/>

<br/>

<img src="https://img.shields.io/badge/Author-Nevil%20Dhinoja-2196F3?style=for-the-badge&logo=github&logoColor=white&labelColor=0D1117"/>
<img src="https://img.shields.io/badge/Built%20At-Arocom%20Solutions-22C55E?style=for-the-badge&labelColor=0D1117"/>
<img src="https://img.shields.io/badge/Stack-Python%20%C2%B7%20Streamlit%20%C2%B7%20Scikit--learn-FF9800?style=for-the-badge&labelColor=0D1117"/>
<img src="https://img.shields.io/badge/Type-Internship%20Project-EF4444?style=for-the-badge&labelColor=0D1117"/>

</div>

---

## What It Does

A web application that lets anyone — with zero Python or ML knowledge — upload a dataset, choose a machine learning model, train it, evaluate it, and make predictions through a clean Streamlit interface.

No code. No installations. No ML background required.

---

## The Problem It Solves

Most ML tools require:
- Python knowledge
- Library installation
- Writing training scripts
- Understanding evaluation metrics manually

This tool eliminates all of that. Upload your CSV, click through the UI, get a trained model and predictions.

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     Streamlit UI                          │
│                                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Upload  │  │  Config  │  │  Train   │  │ Predict  │   │
│  │   CSV    │  │  Model   │  │ & Eval   │  │& Export  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼──────────────┼────────┘
        │             │             │              │
        ▼             ▼             ▼              ▼
┌───────────────────────────────────────────────────────────┐
│                    Helpers Layer                          │
│  data_helper.py · model_helper.py · eval_helper.py        │
└──────────────────────────┬────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Scikit-learn│  │    Pandas    │  │  Matplotlib  │
│  ML Models   │  │  DataFrame   │  │  Seaborn     │
│              │  │  Processing  │  │  Visuals     │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## Features

- **CSV upload** — drag and drop any dataset
- **Auto column detection** — identifies features and target automatically
- **Model selection** — choose from classification and regression algorithms
- **Training** — one-click model training with progress feedback
- **Evaluation** — accuracy, confusion matrix, classification report
- **Prediction** — input new values and get instant predictions
- **Visualisations** — feature importance, correlation heatmap, distribution plots

---

## Models Supported

| Task | Models Available |
|------|-----------------|
| Classification | Logistic Regression, Decision Tree, Random Forest, KNN, SVM |
| Regression | Linear Regression, Decision Tree Regressor, Random Forest Regressor |

---

## Project Structure

```
NO-CODE-ML-MODEL/
├── app.py                  # Main Streamlit entry point
├── pages/                  # Multi-page Streamlit app
│   ├── 1_upload.py         # Data upload + preview
│   ├── 2_configure.py      # Model + feature selection
│   ├── 3_train.py          # Training + evaluation
│   └── 4_predict.py        # Prediction interface
├── helpers/                # Core logic
│   ├── data_helper.py      # Data processing utilities
│   ├── model_helper.py     # Model training + saving
│   └── eval_helper.py      # Evaluation metrics
├── models/                 # Saved trained models
├── requirements.txt
└── .streamlit/config.toml  # UI theme config
```

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/Nevil-Dhinoja/NO-CODE-ML-MODEL.git
cd NO-CODE-ML-MODEL

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## How to Use

1. **Upload** — go to the Upload page, drag in your CSV file
2. **Configure** — select your target column and which features to use
3. **Train** — pick a model, click Train, see evaluation metrics instantly
4. **Predict** — enter new input values and get a prediction

---

## Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Streamlit | Web UI framework |
| Scikit-learn | ML models + evaluation |
| Pandas | Data processing |
| Matplotlib / Seaborn | Visualisations |
| Joblib | Model serialisation |

---

## Built At

This project was built during my internship at **Arocom Solutions**. It was my first real-world Python project — before I moved into AI engineering, LangChain, and agentic systems.

---

<div align="center">


<br/>

<table border="0" cellspacing="0" cellpadding="0">
<tr>
<td width="180" align="center" valign="top">

<img src="https://github.com/Nevil-Dhinoja.png" width="120" style="border-radius:50%"/>

</td>
<td width="30"></td>
<td valign="middle">

<h2 align="left">Nevil Dhinoja</h2>
<p align="left"><i>AI / ML Engineer &nbsp;·&nbsp; Full-Stack Developer &nbsp;·&nbsp; Gujarat, India</i></p>
<p align="left">
I build AI systems that are practical, deployable, and free to run.<br/>
This project is part of a larger series of open-source AI tools — each one<br/>
designed to teach a real concept through a working, shippable product.
</p>

</td>
</tr>
</table>

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Nevil%20Dhinoja-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/nevil-dhinoja)
[![GitHub](https://img.shields.io/badge/GitHub-Nevil--Dhinoja-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Nevil-Dhinoja)
[![Gmail](https://img.shields.io/badge/Email-nevil%40email.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:dhinoja.nevil@email.com)

<br/>

If this project helped you or saved you time, a star on the repo goes a long way. &nbsp;
![Views](https://komarev.com/ghpvc/?username=Nevil-Dhinoja&repo=data-analyst-agent&color=blue)

<br/>


<br/>
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=120&section=footer" width="100%"/>
