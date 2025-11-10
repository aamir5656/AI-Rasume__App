# 🤖 AI Resume Screening Project

### 📌 Overview
This project uses **Machine Learning and Natural Language Processing (NLP)** techniques to automatically analyze and classify resumes based on candidates’ skills, education, certifications, and job roles.  
It helps recruiters make faster and data-driven hiring decisions using AI.

The model is trained using both **numeric** and **textual** features and deployed as an **interactive Streamlit app**.

---

### 🧠 Key Features
- Preprocessed and analyzed resumes using **TF-IDF Vectorization** for text data.
- Standardized numerical features like experience, AI score, project count, and salary expectation.
- Combined **textual and numeric features** using SciPy’s `hstack()` method.
- Trained and evaluated a **Random Forest Classifier** with balanced class weights.
- Visualized multiple columns (Experience, AI Score, Skills Frequency, Salary Expectations, etc.) to understand data distribution and trends.
- Achieved a strong model accuracy with professional evaluation metrics.
- Exported trained model and preprocessing objects for app deployment.

---

### 📊 Technologies Used
- **Python 3.12+**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Joblib**
- **TF-IDF Vectorizer**
- **Matplotlib**, **Seaborn** (for visualizations)
- **Streamlit** (for deployment)

---

### ⚙️ Model Workflow
1. **Data Loading:**  
   Load dataset (`AI_Resume_Dataset.csv`).

2. **Label Encoding:**  
   Convert recruiter decision (Yes/No) into numeric labels.

3. **Numeric Feature Scaling:**  
   Standardize numeric columns using `StandardScaler`.

4. **Text Feature Engineering:**  
   Combine multiple text fields (`Skills`, `Education`, `Certifications`, `Job Role`) and vectorize using **TF-IDF**.

5. **Train-Test Split:**  
   Split data into training and testing sets (80/20 ratio).

6. **Model Training:**  
   Train a **Random Forest Classifier** with balanced class weights.

7. **Evaluation:**  
   Check accuracy and fine-tune if necessary.

8. **Model Saving:**  
   Save the trained model and all preprocessing objects:
   - `rf_model.pkl`
   - `vectorizer.pkl`
   - `label_encoder.pkl`
   - `scaler.pkl`

---

### 📈 Visualizations
Several insightful visualizations were created to understand the dataset:
- Experience vs AI Score comparison
- Top 10 most frequent skills
- Distribution of salary expectations
- Education level vs recruiter decision
- Job roles and their selection rates

These plots help in identifying the most influential factors affecting recruitment outcomes.

---

### 🌐 Streamlit App
An interactive web app has been created using **Streamlit**, allowing recruiters to:
- Upload or enter resume data.
- Instantly view AI-based predictions.
- Explore insights through dynamic visualizations.

🔗 **App Link:** [Click Here to Open the App](https://ai-rasumeapp-afncxh4fnqcgjmytbujfh2.streamlit.app/)  
*(Replace the above link with your deployed Streamlit app URL.)*

---

### 🚀 How to Run the Project Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/AI_Resume_App.git
   ```
2. Navigate to the project folder:
   ```bash
   cd AI_Resume_App
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

### 📂 Project Structure
```
AI_Resume_App/
│
├── AI_Resume_Dataset.csv
├── app.py
├── rf_model.pkl
├── vectorizer.pkl
├── label_encoder.pkl
├── scaler.pkl
├── README.md
└── requirements.txt
```

---

### 🧾 License
This project is open-source under the **MIT License**.  
You are free to use, modify, and distribute it with proper attribution.

---

### ✨ Author
**Aamir Shahzad**  
📧 [aamirjee44556@gmail.com]  
💼 [GitHub Profile](https://github.com/aamir5656)
