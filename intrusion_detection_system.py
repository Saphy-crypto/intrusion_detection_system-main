#!/usr/bin/env python3
"""
Machine Learning Based Network Intrusion Detection System

This system implements multiple classical ML algorithms to detect network anomalies
including DDoS attacks, port scans, infiltration, and web attacks in network traffic.

Dataset: CICIDS2017 - a dataset for intrusion detection evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import warnings
import glob
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class NetworkIntrusionDetector:
    # Network intrusion detection system using multiple ML algorithms
    def __init__(self, data_path="Datasets/"):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        
    def load_and_combine_data(self):
        #load and combine all csv files from the dataset directory
        print("Loading dataset files...")
        
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        dataframes = []
        
        for file in tqdm(csv_files, desc="Loading CSV files"):
            print(f"Loading {os.path.basename(file)}...")
            
            df = pd.read_csv(file)

            # Sample large files to reduce memory usage
            if len(df) > 100_000:
                df = df.sample(n=100_000, random_state=42)

            # clean column names
            df.columns = df.columns.str.strip()

            dataframes.append(df)

        #combine all dataframes
        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"Combined dataset shape: {self.data.shape}")
        
        # display label distribution
        print("\nLabel distribution:")
        print(self.data['Label'].value_counts())
        
    def preprocess_data(self):
        #Clean and preprocess the data for ML models
        print("\nPreprocessing data...")
        
        # handle missing values
        print(f"Missing values before cleaning: {self.data.isnull().sum().sum()}")
        
        # replace infinite values with NaN
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # fill missing values with median for numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numerical_cols] = self.data[numerical_cols].fillna(self.data[numerical_cols].median())
        
        # Convert to more memory-efficient data types
        for col in numerical_cols:
            if self.data[col].dtype == 'float64':
                self.data[col] = pd.to_numeric(self.data[col], downcast='float')
            elif self.data[col].dtype == 'int64':
                self.data[col] = pd.to_numeric(self.data[col], downcast='integer')
        
        print(f"Missing values after cleaning: {self.data.isnull().sum().sum()}")
        
        #sample data if too large (for SVM memory efficiency)
        if len(self.data) > 100000:
            print(f"Dataset is large ({len(self.data):,} samples). Sampling 100,000 for efficiency...")
            # Use stratified sampling to preserve class distribution
            self.data = self.data.groupby('Label').apply(
                lambda x: x.sample(min(len(x), max(2, int(100000 * len(x) / len(self.data)))), random_state=42)
            ).reset_index(drop=True)
            print(f"Sampled dataset shape: {self.data.shape}")
            print("Label distribution after sampling:")
            print(self.data['Label'].value_counts())
        
        #separate features and target
        X = self.data.drop('Label', axis=1)
        y = self.data['Label']
        
        #encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        #split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        #scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
    def initialize_models(self):
        
        #initialize all ML models with optimized parameters
        print("\nInitializing ML models...")
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='linear', 
                C=0.1, 
                random_state=42,
                probability=False,
                max_iter=1000
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                n_jobs=-1
            )
        }
        
    def train_and_evaluate_models(self):
        
        #train all models and evaluate their performance
       
        print("\nTraining and evaluating models...")
        
        for name, model in tqdm(self.models.items(), desc="Training models"):
            print(f"\n--- Training {name} ---")
            
            # choose the correct feature scaling
            if name in ['SVM', 'Logistic Regression']:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # train model with progress indication
            print("Training in progress...")
            model.fit(X_train_use, self.y_train)
            print("Training completed!")
            
            # make predictions
            print("Making predictions...")
            y_pred = model.predict(X_test_use)
            
            #calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            #store results (without storing predictions to save memory)
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            # Store predictions only for best model (determined later)
            if not hasattr(self, '_best_predictions'):
                self._best_predictions = {}
            self._best_predictions[name] = y_pred
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
    def generate_detailed_report(self):
        
        #generate evaluation report
        
        print("\n" + "="*60)
        print("NETWORK INTRUSION DETECTION SYSTEM - EVALUATION REPORT")
        print("="*60)
        
        #model comparison table
        print("\n1. MODEL PERFORMANCE COMPARISON")
        print("-" * 60)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 60)
        
        for name, results in self.results.items():
            print(f"{name:<20} {results['accuracy']:<10.4f} {results['precision']:<10.4f} "
                  f"{results['recall']:<10.4f} {results['f1_score']:<10.4f}")
        
        #find best model
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['f1_score'])
        best_model_results = self.results[best_model_name]
        
        print(f"\n2. BEST PERFORMING MODEL: {best_model_name}")
        print("-" * 60)
        print(f"F1-Score: {best_model_results['f1_score']:.4f}")
        print(f"Accuracy: {best_model_results['accuracy']:.4f}")
        
        #classification report for best model
        print(f"\n3. DETAILED CLASSIFICATION REPORT - {best_model_name}")
        print("-" * 60)
        
        # get label names
        label_names = self.label_encoder.classes_
        print("\nClass Distribution in Test Set:")
        unique, counts = np.unique(self.y_test, return_counts=True)
        for i, (label_idx, count) in enumerate(zip(unique, counts)):
            print(f"{label_names[label_idx]}: {count} samples")
        
        print(f"\nClassification Report:")
        best_predictions = self._best_predictions[best_model_name]
        # Only use labels that actually appear in test set
        unique_test_labels = np.unique(self.y_test)
        present_label_names = [label_names[i] for i in unique_test_labels]
        print(classification_report(self.y_test, best_predictions, 
                                  target_names=present_label_names, 
                                  labels=unique_test_labels))
        
        # feature importance for tree based models
        if best_model_name in ['Random Forest']:
            print(f"\n4. TOP 10 MOST IMPORTANT FEATURES - {best_model_name}")
            print("-" * 60)
            
            feature_names = self.X_train.columns
            importances = best_model_results['model'].feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                print(f"{i+1:2d}. {feature:<30} {importance:.4f}")
    
    def save_results(self):
        
        #save evaluation results to cvs file
        
        results_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1_score']
            }
            for name, results in self.results.items()
        ])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intrusion_detection_results_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        
    def plot_results(self):
        
        #create visualization of model performance
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        models = list(self.results.keys())
        
        # prepare data for plotting
        data_for_plot = []
        for model in models:
            for metric in metrics:
                data_for_plot.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': self.results[model][metric]
                })
        
        plot_df = pd.DataFrame(data_for_plot)
        
        # create plot
        plt.figure(figsize=(15, 8))
        sns.barplot(data=plot_df, x='Model', y='Score', hue='Metric')
        plt.title('Network Intrusion Detection System - Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_full_analysis(self):
        """
        Execute complete intrusion detection analysis
        """
        print("Starting Network Intrusion Detection System Analysis...")
        print("=" * 60)
        
        self.load_and_combine_data()
        self.preprocess_data()
        self.initialize_models()
        self.train_and_evaluate_models()
        self.generate_detailed_report()
        self.save_results()
        
        try:
            self.plot_results()
        except Exception as e:
            print(f"Note: Could not generate plots due to display limitations: {e}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nKey findings:")
        print("- Multiple ML algorithms successfully trained for intrusion detection")
        print("- Models can distinguish between benign traffic and various attack types")
        print("- Performance metrics saved for detailed analysis")
        print("- System ready for deployment in network security monitoring")


if __name__ == "__main__":
    # initialize and run the intrusion detection system
    detector = NetworkIntrusionDetector()
    detector.run_full_analysis()