# 🔒 Privacy-Preserving Similarity Search System

<img width="1898" height="974" alt="privacy_pic" src="https://github.com/user-attachments/assets/f6070eaa-15bc-4c65-ac9b-31ba8323c9aa" />

📌 Overview

The Privacy-Preserving Similarity Search System is a secure machine learning application that allows users to upload datasets, encrypt them using Paillier Homomorphic Encryption, and perform similarity searches without exposing sensitive data.

It leverages:

Homomorphic Encryption (via phe) for secure computation.

VP-Tree (Vantage Point Tree) for efficient similarity search.

Streamlit for an interactive web interface.

PCA + Plotly for dataset visualization.

This project demonstrates how modern cryptographic techniques can be combined with machine learning to build privacy-preserving data analytics systems.

✨ Features

🔑 Homomorphic Encryption using Paillier cryptosystem.

🌲 Optimized VP-Tree implementation for fast similarity searches.

📊 Dataset Validation (numeric check, missing values handling, outlier detection).

🔍 Privacy Logger to track operations without compromising sensitive data.

📉 Dimensionality Reduction & Visualization with PCA + Plotly.

📥 Encrypted Dataset Download as CSV.

🌐 Streamlit UI for ease of use.

🛠️ Tech Stack

* Python 3.11+

* Libraries:

* streamlit (UI)

* numpy, scipy, scikit-learn (ML + math)

* phe (homomorphic encryption)

* plotly (data visualization)

* pandas (data handling)



https://github.com/user-attachments/assets/3122157f-5e8b-4271-80b6-a9e54c55323c


📂 Project Structure

privacy-preserving-similarity-search/

│── privacy_similarity_search.py   # Main Streamlit app

│── requirements.txt               # Project dependencies

│── privacy_search.log             # Auto-generated log file

│── README.md                      # Project description


🚀 Installation & Setup

1. Clone Repository
git clone https://github.com/your-username/privacy-preserving-similarity-search.git
cd privacy-preserving-similarity-search

2. Install Dependencies
python -m pip install -r requirements.txt

3. Run the Application
python -m streamlit run privacy_similarity_search.py

4. Open in Browser
Go to: http://localhost:8501


📊 Usage Guide

* Upload your dataset (CSV format).

* The system validates, cleans, and encrypts the dataset.

* VP-Tree is built on encrypted data.

* Enter a query vector to perform similarity search.

* View results, visualize embeddings, and download encrypted datasets.


📌 Future Enhancements

🔐 Support for additional homomorphic encryption schemes.

⚡ Performance optimization for large-scale datasets.

☁️ Integration with cloud storage for secure data sharing.

🤝 Multi-user authentication and access control.



📜 License

This project is licensed under the MIT License – feel free to use and modify it for your own research or applications.
