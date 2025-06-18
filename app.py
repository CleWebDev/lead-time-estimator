#!/usr/bin/env python3
"""
Lead Time Estimator Flask Application
Complete B2B Lead Time prediction system with web interface
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import pickle
import os
from typing import Dict, List, Tuple, Any

from flask import Flask, render_template_string, request, jsonify
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class LeadTimeDataGenerator:
    """Generates realistic training data for lead time estimation"""
    
    def __init__(self):
        # Factor definitions (8 factors, ≤6 options each)
        self.product_types = {
            'Electronics': {'base_days': 12, 'variance': 0.3},
            'Machinery': {'base_days': 35, 'variance': 0.4},
            'Auto Parts': {'base_days': 8, 'variance': 0.2},
            'Chemicals': {'base_days': 18, 'variance': 0.35},
            'Textiles': {'base_days': 15, 'variance': 0.25}
        }
        
        self.quantity_ranges = {
            'Small (1-25)': {'multiplier': 0.9, 'range': (1, 25)},
            'Medium (26-100)': {'multiplier': 1.0, 'range': (26, 100)},
            'Large (101-500)': {'multiplier': 1.3, 'range': (101, 500)},
            'Bulk (500+)': {'multiplier': 1.6, 'range': (500, 2000)}
        }
        
        self.regions = {
            'North America': {'shipping_days': 2, 'factor': 1.0},
            'Europe': {'shipping_days': 5, 'factor': 1.1},
            'Asia': {'shipping_days': 10, 'factor': 1.2},
            'South America': {'shipping_days': 8, 'factor': 1.15},
            'Middle East': {'shipping_days': 7, 'factor': 1.1},
            'Africa': {'shipping_days': 12, 'factor': 1.25}
        }
        
        self.seasons = {
            'Spring (Mar-May)': {'months': [3, 4, 5], 'factor': 1.0},
            'Summer (Jun-Aug)': {'months': [6, 7, 8], 'factor': 0.9},
            'Fall (Sep-Nov)': {'months': [9, 10, 11], 'factor': 1.2},
            'Winter (Dec-Feb)': {'months': [12, 1, 2], 'factor': 1.4}
        }
        
        self.complexity_levels = {
            'Simple': {'factor': 0.8},
            'Standard': {'factor': 1.0},
            'Complex': {'factor': 1.4},
            'Custom': {'factor': 1.8}
        }
        
        self.supply_chain_status = {
            'Normal': {'factor': 1.0},
            'Minor Delays': {'factor': 1.2},
            'Moderate Delays': {'factor': 1.5},
            'Major Delays': {'factor': 2.0},
            'Critical Issues': {'factor': 2.5}
        }
        
        self.factory_loads = {
            'Low (<60%)': {'factor': 0.8},
            'Normal (60-80%)': {'factor': 1.0},
            'High (80-95%)': {'factor': 1.3},
            'Critical (95%+)': {'factor': 1.7}
        }
        
        self.priority_levels = {
            'Standard': {'factor': 1.0},
            'High Priority': {'factor': 0.7},
            'Expedited': {'factor': 0.5}
        }

    def calculate_lead_time(self, product_type: str, quantity_range: str, 
                          region: str, season: str, complexity: str,
                          supply_status: str, factory_load: str, 
                          priority: str) -> float:
        """Calculate lead time based on the 8 factors"""
        
        base_days = self.product_types[product_type]['base_days']
        
        total_factor = (
            self.quantity_ranges[quantity_range]['multiplier'] *
            self.regions[region]['factor'] *
            self.seasons[season]['factor'] *
            self.complexity_levels[complexity]['factor'] *
            self.supply_chain_status[supply_status]['factor'] *
            self.factory_loads[factory_load]['factor'] *
            self.priority_levels[priority]['factor']
        )
        
        manufacturing_days = base_days * total_factor
        shipping_days = self.regions[region]['shipping_days']
        total_days = manufacturing_days + shipping_days
        
        # Add realistic variance
        variance = self.product_types[product_type]['variance']
        noise = np.random.normal(1.0, variance * 0.1)
        
        return max(1.0, total_days * noise)

    def generate_sample_quantity(self, quantity_range: str) -> int:
        """Generate a sample quantity within the range"""
        min_qty, max_qty = self.quantity_ranges[quantity_range]['range']
        return random.randint(min_qty, max_qty)

    def generate_training_data(self, n_orders: int = 5000) -> pd.DataFrame:
        """Generate training data"""
        
        orders = []
        
        for i in range(n_orders):
            product_type = random.choice(list(self.product_types.keys()))
            quantity_range = random.choice(list(self.quantity_ranges.keys()))
            region = random.choice(list(self.regions.keys()))
            season = random.choice(list(self.seasons.keys()))
            complexity = random.choice(list(self.complexity_levels.keys()))
            supply_status = random.choice(list(self.supply_chain_status.keys()))
            factory_load = random.choice(list(self.factory_loads.keys()))
            priority = random.choice(list(self.priority_levels.keys()))
            
            quantity = self.generate_sample_quantity(quantity_range)
            
            lead_time = self.calculate_lead_time(
                product_type, quantity_range, region, season, 
                complexity, supply_status, factory_load, priority
            )
            
            orders.append({
                'product_type': product_type,
                'quantity_range': quantity_range,
                'customer_region': region,
                'season': season,
                'product_complexity': complexity,
                'supply_chain_status': supply_status,
                'factory_load': factory_load,
                'priority_level': priority,
                'actual_quantity': quantity,
                'estimated_lead_time_days': round(lead_time, 1)
            })
        
        return pd.DataFrame(orders)

    def get_factor_options(self) -> Dict:
        """Return all factor options for UI generation"""
        return {
            'product_type': list(self.product_types.keys()),
            'quantity_range': list(self.quantity_ranges.keys()),
            'customer_region': list(self.regions.keys()),
            'season': list(self.seasons.keys()),
            'product_complexity': list(self.complexity_levels.keys()),
            'supply_chain_status': list(self.supply_chain_status.keys()),
            'factory_load': list(self.factory_loads.keys()),
            'priority_level': list(self.priority_levels.keys())
        }


class LeadTimeModel:
    """XGBoost model for lead time prediction"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = [
            'product_type', 'quantity_range', 'customer_region', 'season',
            'product_complexity', 'supply_chain_status', 'factory_load', 'priority_level'
        ]
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        df_encoded = df.copy()
        
        for feature in self.feature_names:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df_encoded[feature] = self.label_encoders[feature].fit_transform(df[feature])
            else:
                # Handle unseen categories during prediction
                seen_categories = set(self.label_encoders[feature].classes_)
                df_encoded[feature] = df[feature].apply(
                    lambda x: x if x in seen_categories else self.label_encoders[feature].classes_[0]
                )
                df_encoded[feature] = self.label_encoders[feature].transform(df_encoded[feature])
                
        return df_encoded[self.feature_names]
    
    def train(self, df: pd.DataFrame):
        """Train the XGBoost model"""
        X = self.prepare_features(df)
        y = df['estimated_lead_time_days']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"Training MAE: {train_mae:.2f} days")
        print(f"Testing MAE: {test_mae:.2f} days")
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
    
    def predict(self, input_data: Dict) -> Dict:
        """Make prediction and provide explanation"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        X = self.prepare_features(df)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Convert to Python native float for JSON serialization
        prediction = float(prediction)
        
        # Get feature importance for explanation
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Generate explanation
        explanation = self._generate_explanation(input_data, prediction, feature_importance)
        
        # Determine urgency level
        urgency_info = self._get_urgency_info(prediction)
        
        return {
            'predicted_lead_time': round(prediction, 1),
            'explanation': explanation,
            **urgency_info
        }
    
    def _generate_explanation(self, input_data: Dict, prediction: float, 
                            feature_importance: Dict) -> List[str]:
        """Generate human-readable explanation"""
        explanation = []
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Key factors explanation
        top_factors = sorted_features[:3]
        
        for feature, importance in top_factors:
            value = input_data[feature]
            impact = self._get_factor_impact(feature, value)
            explanation.append(f"{feature.replace('_', ' ').title()}: {value} {impact}")
        
        # Overall assessment
        if prediction > 30:
            explanation.append("Extended lead time due to multiple complexity factors")
        elif prediction < 10:
            explanation.append("Short lead time - favorable conditions across factors")
        else:
            explanation.append("Standard lead time within normal business parameters")
            
        return explanation
    
    def _get_factor_impact(self, feature: str, value: str) -> str:
        """Get impact description for a factor value"""
        impact_map = {
            'product_type': {
                'Electronics': '(moderate complexity)',
                'Machinery': '(high complexity)',
                'Auto Parts': '(low complexity)',
                'Chemicals': '(regulatory requirements)',
                'Textiles': '(seasonal variations)'
            },
            'quantity_range': {
                'Small (1-25)': '(quick turnaround)',
                'Medium (26-100)': '(standard processing)',
                'Large (101-500)': '(extended manufacturing)',
                'Bulk (500+)': '(significant setup time)'
            },
            'customer_region': {
                'North America': '(fast shipping)',
                'Europe': '(moderate shipping)',
                'Asia': '(longer shipping)',
                'South America': '(customs delays)',
                'Middle East': '(moderate delays)',
                'Africa': '(extended shipping)'
            },
            'priority_level': {
                'Standard': '(normal queue)',
                'High Priority': '(accelerated)',
                'Expedited': '(rush processing)'
            }
        }
        
        return impact_map.get(feature, {}).get(value, '')
    
    def _get_urgency_info(self, prediction: float) -> Dict:
        """Determine urgency level and messaging"""
        if prediction <= 10:
            return {
                'urgency_level': 'low',
                'urgency_color': '#27ae60',
                'urgency_message': 'Fast delivery expected'
            }
        elif prediction <= 20:
            return {
                'urgency_level': 'normal',
                'urgency_color': '#f39c12',
                'urgency_message': 'Standard delivery timeframe'
            }
        elif prediction <= 40:
            return {
                'urgency_level': 'extended',
                'urgency_color': '#e67e22',
                'urgency_message': 'Extended delivery timeframe'
            }
        else:
            return {
                'urgency_level': 'critical',
                'urgency_color': '#e74c3c',
                'urgency_message': 'Significant lead time - plan accordingly'
            }
    
    def save_model(self, filepath: str):
        """Save the trained model and encoders"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model and encoders"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']


# Flask Application
app = Flask(__name__)

# Global variables
lead_time_model = LeadTimeModel()
factor_options = {}

def initialize_model():
    """Initialize or load the model"""
    model_path = 'lead_time_model.pkl'
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        lead_time_model.load_model(model_path)
    else:
        print("Training new model...")
        generator = LeadTimeDataGenerator()
        
        # Generate training data
        training_data = generator.generate_training_data(n_orders=5000)
        print(f"Generated {len(training_data)} training samples")
        
        # Train model
        metrics = lead_time_model.train(training_data)
        print(f"Model trained with MAE: {metrics['test_mae']:.2f} days")
        
        # Save model
        lead_time_model.save_model(model_path)
        print("Model saved")
    
    # Set factor options for UI
    global factor_options
    generator = LeadTimeDataGenerator()
    factor_options = generator.get_factor_options()

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lead Time Estimator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            text-align: center;
        }

        .header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }

        .header p {
            color: #7f8c8d;
        }

        .main-content {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }

        .input-panel {
            flex: 1;
            min-width: 400px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .results-panel {
            flex: 1;
            min-width: 400px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .form-group {
            margin-bottom: 15px;
        }

        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #2c3e50;
        }

        select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }

        select:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }

        .predict-btn {
            width: 100%;
            padding: 12px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        .predict-btn:hover {
            background-color: #2980b9;
        }

        .predict-btn:disabled {
            background-color: #bdc3c7;
            cursor: not-allowed;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
        }

        .results {
            display: none;
        }

        .urgency-indicator {
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: 500;
        }

        .prediction-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .explanation {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }

        .explanation h4 {
            margin-bottom: 10px;
            color: #2c3e50;
        }

        .explanation ul {
            list-style-type: none;
            padding-left: 0;
        }

        .explanation li {
            padding: 5px 0;
            border-bottom: 1px solid #ecf0f1;
        }

        .explanation li:last-child {
            border-bottom: none;
        }

        .error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }

        .demo-scenarios {
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f4fd;
            border-radius: 6px;
        }

        .demo-btn {
            padding: 8px 12px;
            margin: 5px;
            background-color: #17a2b8;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }

        .demo-btn:hover {
            background-color: #138496;
        }

        @media (max-width: 768px) {
            .main-content {
                flex-direction: column;
            }
            
            .input-panel, .results-panel {
                min-width: auto;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>B2B Lead Time Estimator</h1>
            <p>AI-powered delivery estimation considering product complexity, supply chain, and operational factors</p>
        </div>

        <div class="main-content">
            <div class="input-panel">
                <h3>Order Parameters</h3>
                <form id="predictionForm">
                    <div class="form-group">
                        <label for="product_type">Product Type:</label>
                        <select id="product_type" name="product_type" required>
                            <option value="">Select product type...</option>
                            {% for option in factor_options.product_type %}
                            <option value="{{ option }}">{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="quantity_range">Order Quantity:</label>
                        <select id="quantity_range" name="quantity_range" required>
                            <option value="">Select quantity range...</option>
                            {% for option in factor_options.quantity_range %}
                            <option value="{{ option }}">{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="customer_region">Customer Region:</label>
                        <select id="customer_region" name="customer_region" required>
                            <option value="">Select region...</option>
                            {% for option in factor_options.customer_region %}
                            <option value="{{ option }}">{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="season">Season:</label>
                        <select id="season" name="season" required>
                            <option value="">Select season...</option>
                            {% for option in factor_options.season %}
                            <option value="{{ option }}">{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="product_complexity">Product Complexity:</label>
                        <select id="product_complexity" name="product_complexity" required>
                            <option value="">Select complexity...</option>
                            {% for option in factor_options.product_complexity %}
                            <option value="{{ option }}">{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="supply_chain_status">Supply Chain Status:</label>
                        <select id="supply_chain_status" name="supply_chain_status" required>
                            <option value="">Select status...</option>
                            {% for option in factor_options.supply_chain_status %}
                            <option value="{{ option }}">{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="factory_load">Factory Load:</label>
                        <select id="factory_load" name="factory_load" required>
                            <option value="">Select factory load...</option>
                            {% for option in factor_options.factory_load %}
                            <option value="{{ option }}">{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="priority_level">Priority Level:</label>
                        <select id="priority_level" name="priority_level" required>
                            <option value="">Select priority...</option>
                            {% for option in factor_options.priority_level %}
                            <option value="{{ option }}">{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <button type="submit" class="predict-btn" id="predictBtn">
                        Calculate Lead Time
                    </button>
                </form>

                <div class="demo-scenarios">
                    <h4>Quick Demo Scenarios:</h4>
                    <button class="demo-btn" onclick="loadScenario('best')">Best Case</button>
                    <button class="demo-btn" onclick="loadScenario('worst')">Worst Case</button>
                    <button class="demo-btn" onclick="loadScenario('typical')">Typical Order</button>
                    <button class="demo-btn" onclick="loadScenario('rush')">Rush Order</button>
                    <br><small>Debug: {{ factor_options|length }} factor groups loaded</small>
                </div>
            </div>

            <div class="results-panel">
                <h3>Lead Time Estimate</h3>
                
                <div id="loading" class="loading">
                    <p>Calculating lead time...</p>
                </div>

                <div id="error" class="error" style="display: none;"></div>

                <div id="results" class="results">
                    <div id="urgencyIndicator" class="urgency-indicator">
                        <div class="prediction-value" id="predictionValue"></div>
                        <div id="urgencyMessage"></div>
                    </div>

                    <div class="explanation">
                        <h4>Estimation Factors</h4>
                        <ul id="explanationList"></ul>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const demoScenarios = {
            best: {
                product_type: 'Auto Parts',
                quantity_range: 'Small (1-25)',
                customer_region: 'North America',
                season: 'Summer (Jun-Aug)',
                product_complexity: 'Simple',
                supply_chain_status: 'Normal',
                factory_load: 'Low (<60%)',
                priority_level: 'Expedited'
            },
            worst: {
                product_type: 'Machinery',
                quantity_range: 'Bulk (500+)',
                customer_region: 'Africa',
                season: 'Winter (Dec-Feb)',
                product_complexity: 'Custom',
                supply_chain_status: 'Critical Issues',
                factory_load: 'Critical (95%+)',
                priority_level: 'Standard'
            },
            typical: {
                product_type: 'Electronics',
                quantity_range: 'Medium (26-100)',
                customer_region: 'Europe',
                season: 'Spring (Mar-May)',
                product_complexity: 'Standard',
                supply_chain_status: 'Normal',
                factory_load: 'Normal (60-80%)',
                priority_level: 'Standard'
            },
            rush: {
                product_type: 'Chemicals',
                quantity_range: 'Small (1-25)',
                customer_region: 'Asia',
                season: 'Fall (Sep-Nov)',
                product_complexity: 'Complex',
                supply_chain_status: 'Minor Delays',
                factory_load: 'High (80-95%)',
                priority_level: 'High Priority'
            }
        };

        function loadScenario(scenario) {
            const data = demoScenarios[scenario];
            for (const [key, value] of Object.entries(data)) {
                document.getElementById(key).value = value;
            }
        }

        class LeadTimeApp {
            constructor() {
                this.form = document.getElementById('predictionForm');
                this.predictBtn = document.getElementById('predictBtn');
                this.loading = document.getElementById('loading');
                this.results = document.getElementById('results');
                this.error = document.getElementById('error');
                
                this.initEventListeners();
            }

            initEventListeners() {
                this.form.addEventListener('submit', (e) => this.handlePrediction(e));
            }

            async handlePrediction(event) {
                event.preventDefault();
                
                this.showLoading();
                this.hideError();
                
                try {
                    const formData = new FormData(this.form);
                    const data = {};
                    for (let [key, value] of formData.entries()) {
                        data[key] = value;
                    }

                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const result = await response.json();
                    this.displayResults(result);
                    
                } catch (error) {
                    this.showError(`Error calculating lead time: ${error.message}`);
                } finally {
                    this.hideLoading();
                }
            }

            displayResults(result) {
                const urgencyIndicator = document.getElementById('urgencyIndicator');
                const predictionValue = document.getElementById('predictionValue');
                const urgencyMessage = document.getElementById('urgencyMessage');
                
                urgencyIndicator.style.backgroundColor = result.urgency_color + '20';
                urgencyIndicator.style.borderLeft = `4px solid ${result.urgency_color}`;
                
                predictionValue.innerHTML = `${result.predicted_lead_time} days`;
                predictionValue.style.color = result.urgency_color;
                urgencyMessage.textContent = result.urgency_message;

                // Update explanation
                const explanationList = document.getElementById('explanationList');
                explanationList.innerHTML = '';
                
                result.explanation.forEach(explanation => {
                    const li = document.createElement('li');
                    li.textContent = explanation;
                    explanationList.appendChild(li);
                });

                this.results.style.display = 'block';
            }

            showLoading() {
                this.loading.style.display = 'block';
                this.results.style.display = 'none';
                this.predictBtn.disabled = true;
                this.predictBtn.textContent = 'Calculating...';
            }

            hideLoading() {
                this.loading.style.display = 'none';
                this.predictBtn.disabled = false;
                this.predictBtn.textContent = 'Calculate Lead Time';
            }

            showError(message) {
                this.error.textContent = message;
                this.error.style.display = 'block';
                this.results.style.display = 'none';
            }

            hideError() {
                this.error.style.display = 'none';
            }
        }

        // Initialize the app when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            new LeadTimeApp();
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Main page"""
    global factor_options
    
    # Ensure factor_options is populated
    if not factor_options:
        print("Factor options not loaded, initializing...")  # Use print as fallback
        try:
            generator = LeadTimeDataGenerator()
            factor_options = generator.get_factor_options()
            print("Factor options loaded successfully")
        except Exception as e:
            print(f"Failed to load factor options: {e}")
            # Provide minimal fallback options
            factor_options = {
                'product_type': ['Electronics', 'Machinery', 'Auto Parts'],
                'quantity_range': ['Small (1-25)', 'Medium (26-100)', 'Large (101-500)'],
                'customer_region': ['North America', 'Europe', 'Asia'],
                'season': ['Spring (Mar-May)', 'Summer (Jun-Aug)', 'Fall (Sep-Nov)', 'Winter (Dec-Feb)'],
                'product_complexity': ['Simple', 'Standard', 'Complex'],
                'supply_chain_status': ['Normal', 'Minor Delays', 'Major Delays'],
                'factory_load': ['Low (<60%)', 'Normal (60-80%)', 'High (80-95%)'],
                'priority_level': ['Standard', 'High Priority', 'Expedited']
            }
    
    print(f"Rendering page with {len(factor_options)} factor groups")
    return render_template_string(HTML_TEMPLATE, factor_options=factor_options)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for lead time prediction"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = [
            'product_type', 'quantity_range', 'customer_region', 'season',
            'product_complexity', 'supply_chain_status', 'factory_load', 'priority_level'
        ]
        
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Make prediction
        result = lead_time_model.predict(data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': lead_time_model.model is not None,
        'factor_options_loaded': bool(factor_options),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/debug')
def debug():
    """Debug endpoint to check factor options"""
    return jsonify({
        'factor_options_loaded': bool(factor_options),
        'factor_options': factor_options,
        'model_loaded': lead_time_model.model is not None,
        'available_factors': list(factor_options.keys()) if factor_options else []
    })

def main():
    """Main function to run the application"""
    print("Initializing Lead Time Estimator...")
    
    # Initialize model
    initialize_model()
    
    print("Model initialized successfully!")
    print("\nDemo scenarios available:")
    print("- Best Case: Auto Parts, Small quantity, North America, Expedited")
    print("- Worst Case: Machinery, Bulk quantity, Africa, Critical issues")
    print("- Typical Order: Electronics, Medium quantity, Europe, Standard")
    print("- Rush Order: Chemicals, Small quantity, Asia, High Priority")
    
    print(f"\nFactor options loaded:")
    for factor, options in factor_options.items():
        print(f"  {factor}: {len(options)} options")
    
    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()