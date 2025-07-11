"""
Lead Time Estimator - Simplified Version
No logging dependencies, guaranteed to work
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime
from typing import Dict, List

from flask import Flask, render_template_string, request, jsonify
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class LeadTimeDataGenerator:
    def __init__(self):
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

    def calculate_lead_time(self, product_type, quantity_range, region, season, 
                          complexity, supply_status, factory_load, priority):
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
        
        variance = self.product_types[product_type]['variance']
        noise = np.random.normal(1.0, variance * 0.1)
        
        return max(1.0, total_days * noise)

    def generate_sample_quantity(self, quantity_range):
        min_qty, max_qty = self.quantity_ranges[quantity_range]['range']
        return random.randint(min_qty, max_qty)

    def generate_training_data(self, n_orders=2000):
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

    def get_factor_options(self):
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
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = [
            'product_type', 'quantity_range', 'customer_region', 'season',
            'product_complexity', 'supply_chain_status', 'factory_load', 'priority_level'
        ]
        
    def prepare_features(self, df):
        df_encoded = df.copy()
        
        for feature in self.feature_names:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df_encoded[feature] = self.label_encoders[feature].fit_transform(df[feature])
            else:
                seen_categories = set(self.label_encoders[feature].classes_)
                df_encoded[feature] = df[feature].apply(
                    lambda x: x if x in seen_categories else self.label_encoders[feature].classes_[0]
                )
                df_encoded[feature] = self.label_encoders[feature].transform(df_encoded[feature])
                
        return df_encoded[self.feature_names]
    
    def train(self, df):
        X = self.prepare_features(df)
        y = df['estimated_lead_time_days']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        
        self.model.fit(X_train, y_train)
        
        test_pred = self.model.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"Model trained with MAE: {test_mae:.2f} days")
        
        return {'test_mae': float(test_mae)}
    
    def predict(self, input_data):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        df = pd.DataFrame([input_data])
        X = self.prepare_features(df)
        
        prediction = float(self.model.predict(X)[0])
        
        explanation = self._generate_explanation(input_data, prediction)
        urgency_info = self._get_urgency_info(prediction)
        
        return {
            'predicted_lead_time': round(prediction, 1),
            'explanation': explanation,
            **urgency_info
        }
    
    def _generate_explanation(self, input_data, prediction):
        explanation = []
        
        # Key factors
        explanation.append(f"Product Type: {input_data['product_type']} {self._get_factor_impact('product_type', input_data['product_type'])}")
        explanation.append(f"Quantity: {input_data['quantity_range']} {self._get_factor_impact('quantity_range', input_data['quantity_range'])}")
        explanation.append(f"Region: {input_data['customer_region']} {self._get_factor_impact('customer_region', input_data['customer_region'])}")
        
        # Overall assessment
        if prediction > 30:
            explanation.append("Extended lead time due to multiple complexity factors")
        elif prediction < 10:
            explanation.append("Short lead time - favorable conditions across factors")
        else:
            explanation.append("Standard lead time within normal business parameters")
            
        return explanation
    
    def _get_factor_impact(self, feature, value):
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
            }
        }
        
        return impact_map.get(feature, {}).get(value, '')
    
    def _get_urgency_info(self, prediction):
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


# Flask Application
app = Flask(__name__)

# Global variables
lead_time_model = LeadTimeModel()
factor_options = {}

def initialize_model():
    global factor_options
    
    try:
        print("Training new model for demo...")
        
        generator = LeadTimeDataGenerator()
        factor_options = generator.get_factor_options()
        print("Factor options loaded")
        
        training_data = generator.generate_training_data(n_orders=2000)
        print(f"Generated {len(training_data)} training samples")
        
        lead_time_model.train(training_data)
        print("Model training completed")
        
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        if not factor_options:
            generator = LeadTimeDataGenerator()
            factor_options = generator.get_factor_options()
            print("Factor options loaded as fallback")

# Wizard-style HTML Template optimized for 1024x768 iframe
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lead Time Estimator</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">
</head>
<body>
    <div class="page-container">
        <div class="page-header">
            <div class="brand-section">
                <div class="company-logo">A</div>
                <div class="company-name">
                    <span class="primary">ATLAS</span> <span class="secondary">Industrial Supply</span>
                </div>
            </div>
            <div class="app-title-section">
                <h1 class="page-title">Lead Time Estimator</h1>
                <p class="page-subtitle">AI-powered delivery estimation considering product complexity, supply chain, and operational factors</p>
            </div>
        </div>
        
        <!-- Progress Bar -->
        <div class="progress-bar">
            <div class="progress-step active" id="step1-progress">
                <div class="progress-dot"></div>
                <span>Choose Scenario</span>
            </div>
            <div class="progress-step" id="step2-progress">
                <div class="progress-dot"></div>
                <span>Customize</span>
            </div>
            <div class="progress-step" id="step3-progress">
                <div class="progress-dot"></div>
                <span>Results</span>
            </div>
        </div>
        
        <div class="main-panel">
            <h2 class="panel-title" id="panelTitle">Choose Your Scenario</h2>
            
            <!-- Step 1: Demo Scenarios -->
            <div class="wizard-step active" id="step1">
                <div class="demo-scenarios-step">
                    <div class="demo-intro">
                        <h3>Quick Start with Pre-configured Scenarios</h3>
                        <p>Select a scenario to get started quickly, or click "Next" to build your own</p>
                    </div>
                    <div class="demo-grid">
                        <div class="demo-card" data-scenario="best">
                            <h4>Best Case</h4>
                            <p>Fast delivery with optimal conditions</p>
                        </div>
                        <div class="demo-card" data-scenario="worst">
                            <h4>Worst Case</h4>
                            <p>Extended timeline with challenging factors</p>
                        </div>
                        <div class="demo-card" data-scenario="typical">
                            <h4>Typical Order</h4>
                            <p>Standard conditions and average timeline</p>
                        </div>
                        <div class="demo-card" data-scenario="rush">
                            <h4>Rush Order</h4>
                            <p>Expedited processing with priority handling</p>
                        </div>
                    </div>
                </div>
                <div class="wizard-navigation">
                    <button class="nav-button" disabled>Previous</button>
                    <button class="nav-button primary" id="nextToStep2">Next</button>
                </div>
            </div>
            
            <!-- Step 2: Custom Form -->
            <div class="wizard-step" id="step2">
                <div class="form-container">
                <form id="predictionForm">
                        <div class="form-grid">
                    {% for factor, options in factor_options.items() %}
                    <div class="form-group">
                                <label class="form-label" for="{{ factor }}">{{ factor.replace('_', ' ').title() }}</label>
                                <select class="form-select" id="{{ factor }}" name="{{ factor }}" required>
                            <option value="">Select {{ factor.replace('_', ' ') }}...</option>
                            {% for option in options %}
                            <option value="{{ option }}">{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    {% endfor %}
                        </div>
                </form>
                </div>
                <div class="wizard-navigation">
                    <button class="nav-button" id="backToStep1">Previous</button>
                    <button class="nav-button primary" id="calculateBtn">Calculate Lead Time</button>
                </div>
            </div>
            
            <!-- Step 3: Results -->
            <div class="wizard-step" id="step3">
                <div id="loading" class="loading-state">
                    <div class="loading-spinner"></div>
                    <p>Calculating lead time...</p>
                </div>
                <div id="error" class="error-message" style="display: none;"></div>
                <div id="results" class="results-container">
                    <div id="urgencyIndicator" class="urgency-card">
                        <div class="prediction-value" id="predictionValue"></div>
                        <div class="urgency-message" id="urgencyMessage"></div>
                    </div>
                    <div class="explanation-section">
                        <h3 class="explanation-title">Estimation Factors</h3>
                        <ul class="explanation-list" id="explanationList"></ul>
                    </div>
                </div>
                <div class="wizard-navigation">
                    <button class="nav-button" id="backToStep2">Previous</button>
                    <button class="nav-button primary" id="newEstimate">New Estimate</button>
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

        class LeadTimeWizard {
            constructor() {
                this.currentStep = 1;
                this.selectedScenario = null;
                this.form = document.getElementById('predictionForm');
                this.loading = document.getElementById('loading');
                this.results = document.getElementById('results');
                this.error = document.getElementById('error');
                this.initEventListeners();
            }

            initEventListeners() {
                // Demo card selection
                document.querySelectorAll('.demo-card').forEach(card => {
                    card.addEventListener('click', (e) => this.selectScenario(e.currentTarget.dataset.scenario));
                });

                // Navigation buttons
                document.getElementById('nextToStep2').addEventListener('click', () => this.nextStep());
                document.getElementById('backToStep1').addEventListener('click', () => this.previousStep());
                document.getElementById('calculateBtn').addEventListener('click', (e) => this.handlePrediction(e));
                document.getElementById('backToStep2').addEventListener('click', () => this.previousStep());
                document.getElementById('newEstimate').addEventListener('click', () => this.resetWizard());
            }

            selectScenario(scenario) {
                // Remove previous selection
                document.querySelectorAll('.demo-card').forEach(card => {
                    card.classList.remove('selected');
                });

                // Select new scenario
                const selectedCard = document.querySelector(`[data-scenario="${scenario}"]`);
                selectedCard.classList.add('selected');
                this.selectedScenario = scenario;
            }

            nextStep() {
                if (this.currentStep === 1 && this.selectedScenario) {
                    // Load scenario data into form
                    const data = demoScenarios[this.selectedScenario];
                    for (const [key, value] of Object.entries(data)) {
                        const element = document.getElementById(key);
                        if (element) element.value = value;
                    }
                }

                this.showStep(this.currentStep + 1);
            }

            previousStep() {
                this.showStep(this.currentStep - 1);
            }

            showStep(step) {
                // Hide all steps
                document.querySelectorAll('.wizard-step').forEach(s => s.classList.remove('active'));
                
                // Show current step
                document.getElementById(`step${step}`).classList.add('active');
                
                // Update progress
                this.updateProgress(step);
                
                // Update titles
                this.updateTitles(step);
                
                this.currentStep = step;
            }

            updateProgress(step) {
                document.querySelectorAll('.progress-step').forEach((el, index) => {
                    el.classList.remove('active', 'completed');
                    if (index + 1 < step) {
                        el.classList.add('completed');
                    } else if (index + 1 === step) {
                        el.classList.add('active');
                    }
                });
            }

            updateTitles(step) {
                const titles = ['Choose Your Scenario', 'Customize Parameters', 'Lead Time Estimate'];
                document.getElementById('panelTitle').textContent = titles[step - 1];
            }

            resetWizard() {
                this.currentStep = 1;
                this.selectedScenario = null;
                this.form.reset();
                document.querySelectorAll('.demo-card').forEach(card => card.classList.remove('selected'));
                this.showStep(1);
            }

            async handlePrediction(event) {
                event.preventDefault();
                this.showStep(3);
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
                
                // Remove existing urgency classes
                urgencyIndicator.classList.remove('low', 'normal', 'extended', 'critical');
                
                // Add appropriate urgency class based on urgency_level
                if (result.urgency_level) {
                    urgencyIndicator.classList.add(result.urgency_level);
                }
                
                predictionValue.innerHTML = `${result.predicted_lead_time} days`; 
                urgencyMessage.textContent = result.urgency_message; 
                
                const explanationList = document.getElementById('explanationList'); 
                explanationList.innerHTML = ''; 
                result.explanation.forEach(explanation => { 
                    const li = document.createElement('li'); 
                    li.className = 'explanation-item';
                    li.textContent = explanation; 
                    explanationList.appendChild(li); 
                }); 
                this.results.style.display = 'block'; 
            }
            
            showLoading() { 
                this.loading.style.display = 'block'; 
                this.results.style.display = 'none'; 
            }
            
            hideLoading() { 
                this.loading.style.display = 'none'; 
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
        
        document.addEventListener('DOMContentLoaded', () => { 
            new LeadTimeWizard(); 
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    global factor_options
    
    if not factor_options:
        print("Factor options not loaded, initializing...")
        try:
            generator = LeadTimeDataGenerator()
            factor_options = generator.get_factor_options()
            print("Factor options loaded successfully")
        except Exception as e:
            print(f"Failed to load factor options: {e}")
            factor_options = {
                'product_type': ['Electronics', 'Machinery', 'Auto Parts'],
                'quantity_range': ['Small (1-25)', 'Medium (26-100)', 'Large (101-500)'],
                'customer_region': ['North America', 'Europe', 'Asia'],
                'season': ['Spring (Mar-May)', 'Summer (Jun-Aug)', 'Fall (Sep-Nov)'],
                'product_complexity': ['Simple', 'Standard', 'Complex'],
                'supply_chain_status': ['Normal', 'Minor Delays', 'Major Delays'],
                'factory_load': ['Low (<60%)', 'Normal (60-80%)', 'High (80-95%)'],
                'priority_level': ['Standard', 'High Priority', 'Expedited']
            }
    
    print(f"Rendering page with {len(factor_options)} factor groups")
    return render_template_string(HTML_TEMPLATE, factor_options=factor_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        required_fields = [
            'product_type', 'quantity_range', 'customer_region', 'season',
            'product_complexity', 'supply_chain_status', 'factory_load', 'priority_level'
        ]
        
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        result = lead_time_model.predict(data)
        
        # Ensure all values are JSON serializable
        json_safe_result = {}
        for key, value in result.items():
            if isinstance(value, (np.floating, np.integer)):
                json_safe_result[key] = float(value)
            elif isinstance(value, np.ndarray):
                json_safe_result[key] = value.tolist()
            else:
                json_safe_result[key] = value
        
        return jsonify(json_safe_result)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': lead_time_model.model is not None,
        'factor_options_loaded': bool(factor_options),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/debug')
def debug():
    return jsonify({
        'factor_options_loaded': bool(factor_options),
        'factor_options': factor_options,
        'model_loaded': lead_time_model.model is not None,
        'available_factors': list(factor_options.keys()) if factor_options else []
    })

# Initialize model on startup
try:
    initialize_model()
    print("App initialization completed successfully")
except Exception as e:
    print(f"App initialization failed: {e}")
    if not factor_options:
        generator = LeadTimeDataGenerator()
        factor_options = generator.get_factor_options()
        print("Loaded factor options as fallback")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)