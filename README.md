<h1 align="center">ML Comparison Framework 🧪</h1>

<p align="center">
<a href="#">
    <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/Streamlit-1.54+-red.svg" alt="Streamlit">
    <img src="https://img.shields.io/badge/scikit--learn-1.8+-orange.svg" alt="scikit-learn">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</a>
</p>

<p align="center">
  <strong>Train, Compare, and Analyze Machine Learning Algorithms with an Interactive Interface</strong>
</p>

---

## 📌 Overview  
This is a **Machine Learning Comparison Framework** built with **Streamlit, scikit-learn, and Python** that allows users to train multiple ML algorithms, compare their performance, and analyze results with rich visualizations. Perfect for ML enthusiasts, students, and data scientists who want to quickly experiment with different algorithms.

---

## 🚀 Features  

✅ **Multiple Datasets** – 4 built-in datasets + Kaggle integration with popular dataset examples  
✅ **13 ML Algorithms** – Classification, Regression, and Semi-Supervised learning  
✅ **Preprocessing Pipeline** – Handle missing values, normalization, and outlier removal  
✅ **Hyperparameter Tuning** – GridSearchCV and K-Fold cross-validation  
✅ **Rich Visualizations** – Comparison charts, confusion matrices, ROC curves, feature importance, learning curves  
✅ **Custom Predictions** – Input custom values to get predictions from trained models  
✅ **AI Analysis** – Optional Ollama integration for intelligent insights  
✅ **Automated Reports** – Generate performance analysis reports  
✅ **Educational Content** – ML theory sections covering PAC learning, bias-variance, version space, and error bounds  
✅ **Model Persistence** – Save and load trained models  
✅ **Experiment History** – Track all your training runs  

---

## 🛠️ Tech Stack  

### **Core:**  
- 🐍 **Python 3.11+**  
- 🎨 **Streamlit (Web Framework)**  
- 🤖 **scikit-learn (ML Library)**  

### **Data & Visualization:**  
- 📊 **pandas, numpy (Data Processing)**  
- 📈 **matplotlib, seaborn (Visualizations)**  

### **Integrations:**  
- 🔗 **Kaggle API (Dataset Integration)**  
- 🤖 **Ollama (Optional AI Analysis)**  

---

## 📂 Project Structure  

```
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration and constants
├── datasets/                       # Built-in CSV datasets
├── kaggle_integration/             # Kaggle dataset downloader
├── preprocessing/                  # Data preprocessing modules
│   ├── missing_handler.py
│   ├── normalization.py
│   └── outlier_detection.py
├── models/                         # ML algorithm implementations
│   ├── classification_models.py
│   ├── regression_models.py
│   └── semi_supervised.py
├── evaluation/                     # Metrics and comparison tools
│   ├── metrics.py
│   └── comparison.py
├── visualization/                  # Plotting and charts
│   └── plots.py
├── tuning/                         # Hyperparameter tuning
│   └── hyperparameter_tuning.py
├── ollama_integration/             # AI report generation
│   └── report_generator.py
├── utils/                          # Helper utilities
│   └── helpers.py
├── saved_models/                   # Saved model files
└── reports/                        # Generated reports
```

---

## 🎯 Supported Algorithms  

### **Classification (9 Algorithms)**  
- KNN  
- Naive Bayes  
- Logistic Regression  
- Decision Tree (ID3 - Entropy)  
- CART  
- SVM (Linear)  
- SVM (Non-linear)  
- Random Forest  
- Multi-Layer Perceptron  

### **Regression (2 Algorithms)**  
- Linear Regression  
- Multiple Regression  

### **Semi-Supervised (2 Algorithms)**  
- Label Propagation  
- Self-Training  

---

## 📥 Installation  

### **1. Clone the Repository**  
```sh
git clone https://github.com/Manish-Let-It-Be/Algorithm-Comparer.git
cd Algorithm-Comparer/Algorithm-Comparer
```

### **2. Install Dependencies**  
```sh
pip install -r requirements.txt
```

Or using pyproject.toml:  
```sh
pip install -e .
```

### **3. Run the Application**  
```sh
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## ⚙️ Optional Setup  

### **Kaggle Integration**  
To use Kaggle datasets:

1. Get your API credentials from [Kaggle Account Settings](https://www.kaggle.com/account)
2. Create `kaggle.json` with your credentials:
```json
{"username":"your_username","key":"your_api_key"}
```
3. Place it in:
   - **Windows:** `C:\Users\<username>\.kaggle\`
   - **Linux/Mac:** `~/.kaggle/`

### **Ollama AI Analysis**  
To enable AI-powered analysis:

1. Install Ollama: [https://ollama.com/](https://ollama.com/)
2. Start Ollama:
```sh
ollama serve
```
3. Pull a model:
```sh
ollama pull mistral
```

---

## 🎮 Usage Guide  

1. **Select a Dataset** – Choose from built-in datasets or download from Kaggle  
2. **Choose Task Type** – Classification or Regression  
3. **Configure Preprocessing** – Handle missing values, normalize data, remove outliers  
4. **Select Algorithms** – Pick multiple algorithms to compare  
5. **Set Hyperparameters** – Adjust algorithm parameters or use GridSearchCV  
6. **Train Models** – Click "Train All Models" and watch the progress  
7. **Analyze Results** – View metrics, visualizations, and model comparisons  
8. **Make Predictions** – Input custom values to get predictions  
9. **Generate Reports** – Create AI-powered or automated analysis reports  

---

## 🔮 Future Plans  

✅ **Deep Learning Integration** – Add support for neural networks (TensorFlow/PyTorch)  
✅ **AutoML Features** – Automatic algorithm selection and hyperparameter optimization  
✅ **Time Series Support** – Add ARIMA, LSTM, and Prophet for time series forecasting  
✅ **Model Explainability** – Integrate SHAP and LIME for model interpretability  
✅ **Cloud Deployment** – Deploy models directly to cloud platforms  
✅ **Collaborative Features** – Share experiments and results with team members  

---

## 💡 Contributing  
If you'd like to contribute, feel free to **fork this repo**, create a new branch, and submit a **pull request**. All contributions are welcome!  

### **How to Contribute:**
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License  
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgement  

I would like to thank the open-source community for providing excellent libraries and tools that made this project possible:
- **Streamlit** for the amazing web framework
- **scikit-learn** for comprehensive ML algorithms
- **Kaggle** for providing access to diverse datasets

I appreciate everyone who provides feedback and suggestions to improve this framework.

---

## 📧 Contact  

**Manish** - [GitHub Profile](https://github.com/Manish-Let-It-Be)

Project Link: [https://github.com/Manish-Let-It-Be/Algorithm-Comparer](https://mlalgoct.streamlit.app/)

---

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=1F77B4&center=true&width=435&lines=Made+with+%E2%9D%A4%EF%B8%8F+for+ML+Enthusiasts;Thank+You+For+Checking+Out!">
</p>
