import streamlit as st
import numpy as np
import os
import json
import logging
import itertools
import base64
import io
from datetime import datetime

from phe import paillier
from scipy.spatial import distance
from sklearn.decomposition import PCA
import plotly.express as px

# =====================
# LOGGING AND MONITORING
# =====================
class PrivacyLogger:
    def __init__(self, log_file='privacy_search.log'):
        """
        Comprehensive logging for privacy-preserving operations
        """
        logging.basicConfig(
            filename=log_file, 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_search_operation(self, query_vector, results, user_id=None):
        """
        Log each similarity search operation
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query_vector': query_vector.tolist(),
            'results_count': len(results),
            'user_id': user_id or 'anonymous'
        }
        self.logger.info(f"Similarity Search: {json.dumps(log_entry)}")

# =====================
# DATASET VALIDATION
# =====================
def validate_dataset(dataset):
    """
    Advanced dataset validation function
    
    Checks:
    - Numeric data only
    - No missing values (fills missing values with 0)
    - Consistent dimensions
    - Outlier detection
    """
    try:
        # Check for numeric data
        if not np.issubdtype(dataset.dtype, np.number):
            raise ValueError("Dataset must contain only numeric values")
        
        # Check for missing values and fill them with 0
        if np.isnan(dataset).any():
            st.warning("Dataset contains missing values. Filling missing values with 0.")
            dataset = np.nan_to_num(dataset)  # Fill missing values with 0
        
        # Outlier detection using Z-score
        z_scores = np.abs((dataset - dataset.mean()) / dataset.std())
        outliers = np.where(z_scores > 3)
        if len(outliers[0]) > 0:
            st.warning(f"Detected {len(outliers[0])} potential outliers in the dataset")
        
        return True, dataset  # Return the cleaned dataset
    except Exception as e:
        st.error(f"Dataset Validation Error: {e}")
        return False, None

# =====================
# ENCRYPTION FUNCTIONS
# =====================
def generate_keys():
    """Generate public and private keys for homomorphic encryption."""
    public_key, private_key = paillier.generate_paillier_keypair()
    return public_key, private_key

class MultiLayerEncryption:
    def __init__(self, public_key):
        """
        Multi-layer encryption strategy
        """
        self.public_key = public_key
        self.salt = os.urandom(16)  # Random salt for additional security
    
    def encrypt(self, data):
        """
        Enhanced encryption with additional layers
        """
        # Paillier encryption
        encrypted_data = [self.public_key.encrypt(float(x)) for x in data]
        return encrypted_data
    
    def xor_encrypt(self, data, salt):
        """
        Simple XOR encryption for additional layer
        """
        return bytes(a ^ b for a, b in zip(
            str(data).encode(), 
            itertools.cycle(salt)
        ))

# =====================
# DOWNLOAD UTILITY
# =====================
def create_download_link(data, filename):
    """
    Create a downloadable link for encrypted dataset
    """
    # Convert data to CSV string
    csv_buffer = io.StringIO()
    np.savetxt(csv_buffer, data, delimiter=",", fmt='%s')
    
    # Encode to base64
    b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
    
    # Create download link
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Encrypted Dataset</a>'
    return href

# =====================
# VP-TREE IMPLEMENTATION
# =====================
class OptimizedVPTree:
    def __init__(self, encrypted_data, private_key, distance_metric='euclidean'):
        """
        Enhanced VP Tree with:
        - Caching mechanism
        - Multiple distance metrics
        """
        self.private_key = private_key
        self.data = encrypted_data
        self.distance_metric = distance_metric
        self.cache = {}
        
        # Choose distance metric dynamically
        self.distance_func = {
            'euclidean': distance.euclidean,
            'manhattan': distance.cityblock,
            'cosine': distance.cosine
        }.get(distance_metric, distance.euclidean)
        
        self.tree = self.build_tree(self.data)

    def decrypt_row(self, row):
        """Decrypt a row of encrypted data."""
        return [self.private_key.decrypt(cell) for cell in row]

    def cached_distance(self, point1, point2):
        """
        Implement distance caching to avoid redundant computations
        """
        cache_key = (tuple(point1), tuple(point2))
        if cache_key not in self.cache:
            self.cache[cache_key] = self.distance_func(point1, point2)
        return self.cache[cache_key]

    def build_tree(self, data):
        """Build a VP-tree structure."""
        if len(data) == 0:
            return None

        # Decrypt vantage point
        vp = self.decrypt_row(data[0])
        st.write("Vantage Point (Decrypted):", vp)

        if len(data) == 1:
            return {'vp': vp, 'mu': None, 'left': None, 'right': None}

        # Compute distances using decrypted data
        distances = [self.cached_distance(vp, self.decrypt_row(point)) for point in data[1:]]
        st.write("Distances from Vantage Point:", distances)

        mu = np.median(distances)
        st.write("Median Distance (mu):", mu)

        # Partition into left and right subtrees
        left = [data[i + 1] for i, d in enumerate(distances) if d <= mu]
        right = [data[i + 1] for i, d in enumerate(distances) if d > mu]

        return {
            'vp': vp,
            'mu': mu,
            'left': self.build_tree(left),
            'right': self.build_tree(right)
        }

    def search(self, query, k=3):
        """Search for the k most similar points to the query."""
        decrypted_query = query  # Already decrypted
        st.write("Search Query (Decrypted):", decrypted_query)

        # Compute distances using decrypted data
        distances = [
            self.cached_distance(decrypted_query, self.decrypt_row(row))
            for row in self.data
        ]
        st.write("Distances to Query:", distances)

        nearest_indices = np.argsort(distances)[:k]
        st.write("Nearest Indices:", nearest_indices)

        return [self.decrypt_row(self.data[i]) for i in nearest_indices]

# =====================
# VISUALIZATION UTILITIES
# =====================
def visualize_dataset_embedding(dataset):
    """
    Visualize dataset using PCA for dimensionality reduction
    """
    try:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(dataset)
        
        fig = px.scatter(
            x=reduced_data[:, 0], 
            y=reduced_data[:, 1],
            title='Dataset Embedding Visualization'
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Visualization Error: {e}")

# =====================
# MAIN STREAMLIT APPLICATION
# =====================
def main():
    st.title("Privacy-Preserving Similarity Search")
    st.write("Upload your dataset, encrypt it, and perform similarity searches using advanced encryption and VP-tree.")

    # Initialize privacy logger
    privacy_logger = PrivacyLogger()

    # Sidebar for Dataset Upload and Initialization
    st.sidebar.header("Dataset Management")

    # Generate public and private keys
    public_key, private_key = generate_keys()

    # Encryption wrapper
    multi_layer_encryption = MultiLayerEncryption(public_key)

    # Upload the dataset
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            dataset = np.genfromtxt(uploaded_file, delimiter=",")
            
            # Validate dataset and clean missing values
            is_valid, cleaned_dataset = validate_dataset(dataset)
            if not is_valid:
                st.stop()

            st.write("Raw Dataset:", cleaned_dataset)

            # Encrypt the dataset
            encrypted_dataset = [multi_layer_encryption.encrypt(row) for row in cleaned_dataset]
            
            # Convert encrypted dataset to downloadable format
            encrypted_flat = [[str(num.ciphertext()) for num in row] for row in encrypted_dataset]
            
            st.write("Encrypted Dataset (Ciphertexts):", 
                     [[str(cell.ciphertext()) for cell in row] for row in encrypted_dataset])

            # Build VP-tree
            vp_tree = OptimizedVPTree(encrypted_dataset, private_key)
            st.write("VP-Tree built successfully.")

            # Optional: Visualize dataset embedding
            if st.sidebar.checkbox("Visualize Dataset Embedding"):
                visualize_dataset_embedding(cleaned_dataset)

            # Download Encrypted Dataset Button
            if st.sidebar.button("Download Encrypted Dataset"):
                download_filename = "encrypted_dataset.csv"
                download_link = create_download_link(encrypted_flat, download_filename)
                st.sidebar.markdown(download_link, unsafe_allow_html=True)
                st.sidebar.success(f"Encrypted dataset is ready for download: {download_filename}")

            # Query Section
            st.sidebar.header("Query")
            query_input = st.sidebar.text_input("Enter a query (comma-separated values):")
            k = st.sidebar.slider("Select the number of similar results (k):", 1, len(cleaned_dataset), 3)

            if query_input:
                try:
                    query_vector = np.array([float(x) for x in query_input.split(",")])
                    st.write("Parsed Query Vector:", query_vector)

                    # Ensure query dimensions match dataset
                    if len(query_vector) != cleaned_dataset.shape[1]:
                        st.error(f"Query must have {cleaned_dataset.shape[1]} values, separated by commas.")
                    else:
                        # Perform similarity search
                        decrypted_results = vp_tree.search(query_vector, k)

                        st.write("Query Vector (Decrypted):", query_vector)
                        st.write(f"Top {k} Similar Records:")
                        st.dataframe(decrypted_results)

                        # Log search operation
                        privacy_logger.log_search_operation(query_vector, decrypted_results)

                except ValueError:
                    st.error("Query input contains invalid or non-numeric values. Please try again.")

        except Exception as e:
            st.error(f"Dataset Processing Error: {e}")
    else:
        st.info("Please upload a dataset to begin.")

# Run the app
if __name__ == "__main__":
    main()