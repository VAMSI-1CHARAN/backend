import numpy as np
import pandas as pd
import re
import tldextract
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import whois
import datetime
from scipy.stats import entropy
import math
import joblib
import time
import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutException("WHOIS lookup timed out")
    
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

def whois_lookup_with_retry(domain, max_retries=3, delay=2):
    """Perform WHOIS lookup with retry mechanism."""
    for attempt in range(max_retries):
        try:
            with timeout(10):  # Set 10 second timeout
                return whois.whois(domain)
        except TimeoutException:
            print(f"WHOIS lookup timed out for {domain}")
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"WHOIS lookup failed for {domain} after {max_retries} attempts: {str(e)}")
                return None
            time.sleep(delay)
    return None

def calculate_entropy(url):
    """Calculate Shannon entropy of URL string."""
    prob = [float(url.count(c)) / len(url) for c in set(url)]
    return entropy(prob)

def extract_advanced_features(url):
    """Extract advanced features from URL for improved phishing detection."""
    try:
        parsed_url = urlparse(url)
        domain_info = tldextract.extract(url)
        domain = domain_info.domain + '.' + domain_info.suffix
        
        features = []
        
        # Basic features
        features.append(len(url))  # URL length
        features.append(url.count('.'))  # Number of dots
        features.append(url.count('-'))  # Number of hyphens
        features.append(1 if parsed_url.scheme == 'https' else 0)  # HTTPS presence
        
        # Advanced domain features
        features.append(1 if '-' in domain_info.domain else 0)  # Prefix-Suffix
        features.append(1 if re.search(r'[^a-zA-Z0-9-.]', domain_info.domain) else 0)  # Special chars
        features.append(len(domain_info.domain))  # Domain length
        features.append(domain_info.domain.count('-'))  # Domain hyphen count
        
        # URL complexity features
        features.append(calculate_entropy(url))  # URL entropy
        features.append(len(url.split('/')))  # Directory depth
        features.append(1 if re.search(r'\d', domain_info.domain) else 0)  # Numbers in domain
        
        # Path analysis
        path = parsed_url.path
        features.append(len(path))  # Path length
        features.append(path.count('/'))  # Path depth
        features.append(1 if re.search(r'\.(php|html|asp|jsp|cgi)$', path.lower()) else 0)  # Suspicious extensions
        
        # Query parameters
        query = parsed_url.query
        features.append(len(query))  # Query length
        features.append(query.count('='))  # Number of parameters
        features.append(1 if re.search(r'(password|login|user|admin)', query.lower()) else 0)  # Sensitive terms
        
        # Additional security features
        features.append(1 if len(domain_info.domain) > 20 else 0)  # Long domain name
        features.append(1 if url.count('@') > 0 else 0)  # Contains @ symbol
        features.append(1 if re.search(r'(alert|confirm|prompt|eval|exec|system)', url.lower()) else 0)  # Suspicious JS terms
        features.append(1 if re.search(r'(redirect|forward|goto|redir)', url.lower()) else 0)  # Redirect terms
        features.append(1 if re.search(r'(bank|account|login|signin|verify)', url.lower()) else 0)  # Financial/login terms
        
        # IP-based features
        features.append(1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain_info.domain) else 0)  # IP as domain
        features.append(url.count('//'))  # Multiple forward slashes
        
        # TLD analysis
        features.append(1 if domain_info.suffix in ['tk', 'ml', 'ga', 'cf', 'gq'] else 0)  # Free TLD domains
        features.append(len(domain_info.suffix))  # TLD length
        
        # WHOIS features with retry mechanism and graceful failure
        whois_info = whois_lookup_with_retry(domain)
        if whois_info and whois_info.creation_date:
            if isinstance(whois_info.creation_date, list):
                domain_age = (datetime.datetime.now() - whois_info.creation_date[0]).days
            else:
                domain_age = (datetime.datetime.now() - whois_info.creation_date).days
            features.append(domain_age)
            features.append(1)  # DNS record exists
        else:
            features.append(0)  # Default domain age
            features.append(0)  # No DNS record
        
        return features
    except Exception as e:
        print(f"Feature extraction failed for URL {url}: {str(e)}")
        return [0] * 28  # Return default features on failure

def train_improved_random_forest(file_path):
    """Train an improved Random Forest model with advanced features and hyperparameter optimization."""
    print("Loading dataset...")
    df = pd.read_excel(file_path)
    
    print("\nExtracting advanced features...")
    total_urls = len(df['url'])
    X = []
    for i, url in enumerate(df['url'], 1):
        if i % 10 == 0:  # Show progress more frequently
            print(f"Processing URL {i}/{total_urls} ({(i/total_urls)*100:.1f}%)")
        features = extract_advanced_features(url)
        X.append(features)
    X = np.array(X)
    y = df['status'].map({'legitimate': 0, 'phishing': 1})
    
    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Scaling features...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nPerforming hyperparameter optimization...")
    param_dist = {
        'n_estimators': [300, 500, 700],
        'max_depth': [30, 50, 70, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample'],
        'criterion': ['gini', 'entropy']
    }
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    rf_random = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        random_state=42,
        n_jobs=-1,
        scoring='f1',
        verbose=2
    )
    
    print("\nTraining model with cross-validation...")
    rf_random.fit(X_train_scaled, y_train)
    
    print("\nBest parameters found:")
    print(rf_random.best_params_)
    
    best_rf = rf_random.best_estimator_
    y_pred = best_rf.predict(X_test_scaled)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    feature_names = [
        'URL Length', 'Dot Count', 'Hyphen Count', 'HTTPS Present',
        'Prefix-Suffix', 'Special Chars', 'Domain Length', 'Domain Hyphen Count',
        'URL Entropy', 'Directory Depth', 'Numbers in Domain',
        'Path Length', 'Path Depth', 'Suspicious Extensions',
        'Query Length', 'Parameter Count', 'Sensitive Terms',
        'Long Domain', '@Symbol Present', 'Suspicious JS Terms',
        'Redirect Terms', 'Financial Terms', 'IP as Domain',
        'Multiple Slashes', 'Free TLD', 'TLD Length',
        'Domain Age', 'DNS Record'
    ]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_rf.feature_importances_
    })
    print("\nTop 10 most important features:")
    print(feature_importance.sort_values('importance', ascending=False).head(10))
    
    print("\nSaving model and scaler...")
    joblib.dump(best_rf, 'improved_random_forest_model.joblib')
    joblib.dump(scaler, 'improved_feature_scaler.joblib')
    
    return best_rf, scaler

if __name__ == "__main__":
    file_path = "dataset/Dataset-2.xlsx"
    print("Training improved Random Forest model...")
    model, scaler = train_improved_random_forest(file_path)
    print("Training completed successfully!") 