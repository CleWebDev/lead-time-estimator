#!/usr/bin/env python3
"""
Lead Time Estimator - Proof of Concept Data Generator

Simplified version with 8 user-selectable factors, each with 6 or fewer options.
Perfect for demonstrating how different inputs affect lead time predictions.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class LeadTimePOCGenerator:
    def __init__(self):
        # Factor 1: Product Type (5 options)
        self.product_types = {
            'Electronics': {'base_days': 12, 'variance': 0.3},
            'Machinery': {'base_days': 35, 'variance': 0.4},
            'Auto Parts': {'base_days': 8, 'variance': 0.2},
            'Chemicals': {'base_days': 18, 'variance': 0.35},
            'Textiles': {'base_days': 15, 'variance': 0.25}
        }
        
        # Factor 2: Order Quantity (4 options)
        self.quantity_ranges = {
            'Small (1-25)': {'multiplier': 0.9, 'range': (1, 25)},
            'Medium (26-100)': {'multiplier': 1.0, 'range': (26, 100)},
            'Large (101-500)': {'multiplier': 1.3, 'range': (101, 500)},
            'Bulk (500+)': {'multiplier': 1.6, 'range': (500, 2000)}
        }
        
        # Factor 3: Customer Region (6 options)
        self.regions = {
            'North America': {'shipping_days': 2, 'factor': 1.0},
            'Europe': {'shipping_days': 5, 'factor': 1.1},
            'Asia': {'shipping_days': 10, 'factor': 1.2},
            'South America': {'shipping_days': 8, 'factor': 1.15},
            'Middle East': {'shipping_days': 7, 'factor': 1.1},
            'Africa': {'shipping_days': 12, 'factor': 1.25}
        }
        
        # Factor 4: Season (4 options)
        self.seasons = {
            'Spring (Mar-May)': {'months': [3, 4, 5], 'factor': 1.0},
            'Summer (Jun-Aug)': {'months': [6, 7, 8], 'factor': 0.9},
            'Fall (Sep-Nov)': {'months': [9, 10, 11], 'factor': 1.2},
            'Winter (Dec-Feb)': {'months': [12, 1, 2], 'factor': 1.4}
        }
        
        # Factor 5: Product Complexity (4 options)
        self.complexity_levels = {
            'Simple': {'factor': 0.8},
            'Standard': {'factor': 1.0},
            'Complex': {'factor': 1.4},
            'Custom': {'factor': 1.8}
        }
        
        # Factor 6: Supply Chain Status (5 options)
        self.supply_chain_status = {
            'Normal': {'factor': 1.0},
            'Minor Delays': {'factor': 1.2},
            'Moderate Delays': {'factor': 1.5},
            'Major Delays': {'factor': 2.0},
            'Critical Issues': {'factor': 2.5}
        }
        
        # Factor 7: Factory Load (4 options)
        self.factory_loads = {
            'Low (<60%)': {'factor': 0.8},
            'Normal (60-80%)': {'factor': 1.0},
            'High (80-95%)': {'factor': 1.3},
            'Critical (95%+)': {'factor': 1.7}
        }
        
        # Factor 8: Priority Level (3 options)
        self.priority_levels = {
            'Standard': {'factor': 1.0},
            'High Priority': {'factor': 0.7},
            'Expedited': {'factor': 0.5}
        }
    
    def get_season_from_month(self, month: int) -> str:
        """Determine season from month number"""
        for season, info in self.seasons.items():
            if month in info['months']:
                return season
        return 'Spring (Mar-May)'  # Default fallback
    
    def calculate_lead_time(self, product_type: str, quantity_range: str, 
                          region: str, season: str, complexity: str,
                          supply_status: str, factory_load: str, 
                          priority: str) -> float:
        """Calculate lead time based on the 8 factors"""
        
        # Start with base days from product type
        base_days = self.product_types[product_type]['base_days']
        
        # Apply all factors
        total_factor = (
            self.quantity_ranges[quantity_range]['multiplier'] *
            self.regions[region]['factor'] *
            self.seasons[season]['factor'] *
            self.complexity_levels[complexity]['factor'] *
            self.supply_chain_status[supply_status]['factor'] *
            self.factory_loads[factory_load]['factor'] *
            self.priority_levels[priority]['factor']
        )
        
        # Calculate manufacturing days
        manufacturing_days = base_days * total_factor
        
        # Add shipping days
        shipping_days = self.regions[region]['shipping_days']
        
        # Total lead time
        total_days = manufacturing_days + shipping_days
        
        # Add some realistic variance
        variance = self.product_types[product_type]['variance']
        noise = np.random.normal(1.0, variance * 0.1)
        
        final_lead_time = total_days * noise
        
        # Ensure minimum of 1 day
        return max(1.0, final_lead_time)
    
    def generate_sample_quantity(self, quantity_range: str) -> int:
        """Generate a sample quantity within the range"""
        min_qty, max_qty = self.quantity_ranges[quantity_range]['range']
        return random.randint(min_qty, max_qty)
    
    def generate_training_data(self, n_orders: int = 5000) -> pd.DataFrame:
        """Generate training data with all combinations well represented"""
        
        print(f"Generating {n_orders} orders for POC...")
        
        orders = []
        
        for i in range(n_orders):
            if i % 1000 == 0:
                print(f"Processing order {i+1}/{n_orders}")
            
            # Randomly select from each factor
            product_type = random.choice(list(self.product_types.keys()))
            quantity_range = random.choice(list(self.quantity_ranges.keys()))
            region = random.choice(list(self.regions.keys()))
            season = random.choice(list(self.seasons.keys()))
            complexity = random.choice(list(self.complexity_levels.keys()))
            supply_status = random.choice(list(self.supply_chain_status.keys()))
            factory_load = random.choice(list(self.factory_loads.keys()))
            priority = random.choice(list(self.priority_levels.keys()))
            
            # Generate actual quantity
            quantity = self.generate_sample_quantity(quantity_range)
            
            # Calculate lead time
            lead_time = self.calculate_lead_time(
                product_type, quantity_range, region, season, 
                complexity, supply_status, factory_load, priority
            )
            
            # Create order record
            order = {
                'order_id': f'POC_{i+1:05d}',
                # The 8 key factors for user selection
                'product_type': product_type,
                'quantity_range': quantity_range,
                'customer_region': region,
                'season': season,
                'product_complexity': complexity,
                'supply_chain_status': supply_status,
                'factory_load': factory_load,
                'priority_level': priority,
                # Additional data for context
                'actual_quantity': quantity,
                'estimated_lead_time_days': round(lead_time, 1)
            }
            
            orders.append(order)
        
        df = pd.DataFrame(orders)
        
        print(f"\nPOC Dataset generation complete!")
        print(f"Total orders: {len(df)}")
        print(f"Lead time statistics:")
        print(f"  Mean: {df['estimated_lead_time_days'].mean():.1f} days")
        print(f"  Median: {df['estimated_lead_time_days'].median():.1f} days")
        print(f"  Min: {df['estimated_lead_time_days'].min():.1f} days")
        print(f"  Max: {df['estimated_lead_time_days'].max():.1f} days")
        
        return df
    
    def generate_demo_scenarios(self) -> pd.DataFrame:
        """Generate specific scenarios to demonstrate factor impact"""
        
        scenarios = []
        
        # Scenario 1: Best case
        scenarios.append({
            'scenario_name': 'Best Case',
            'product_type': 'Auto Parts',
            'quantity_range': 'Small (1-25)',
            'customer_region': 'North America',
            'season': 'Summer (Jun-Aug)',
            'product_complexity': 'Simple',
            'supply_chain_status': 'Normal',
            'factory_load': 'Low (<60%)',
            'priority_level': 'Expedited'
        })
        
        # Scenario 2: Worst case
        scenarios.append({
            'scenario_name': 'Worst Case',
            'product_type': 'Machinery',
            'quantity_range': 'Bulk (500+)',
            'customer_region': 'Africa',
            'season': 'Winter (Dec-Feb)',
            'product_complexity': 'Custom',
            'supply_chain_status': 'Critical Issues',
            'factory_load': 'Critical (95%+)',
            'priority_level': 'Standard'
        })
        
        # Scenario 3: Typical order
        scenarios.append({
            'scenario_name': 'Typical Order',
            'product_type': 'Electronics',
            'quantity_range': 'Medium (26-100)',
            'customer_region': 'Europe',
            'season': 'Spring (Mar-May)',
            'product_complexity': 'Standard',
            'supply_chain_status': 'Normal',
            'factory_load': 'Normal (60-80%)',
            'priority_level': 'Standard'
        })
        
        # Scenario 4: High priority rush
        scenarios.append({
            'scenario_name': 'Rush Order',
            'product_type': 'Chemicals',
            'quantity_range': 'Small (1-25)',
            'customer_region': 'Asia',
            'season': 'Fall (Sep-Nov)',
            'product_complexity': 'Complex',
            'supply_chain_status': 'Minor Delays',
            'factory_load': 'High (80-95%)',
            'priority_level': 'High Priority'
        })
        
        # Calculate lead times for scenarios
        for scenario in scenarios:
            lead_time = self.calculate_lead_time(
                scenario['product_type'],
                scenario['quantity_range'],
                scenario['customer_region'],
                scenario['season'],
                scenario['product_complexity'],
                scenario['supply_chain_status'],
                scenario['factory_load'],
                scenario['priority_level']
            )
            scenario['estimated_lead_time_days'] = round(lead_time, 1)
            scenario['actual_quantity'] = self.generate_sample_quantity(scenario['quantity_range'])
        
        return pd.DataFrame(scenarios)
    
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
    
    def save_data(self, df: pd.DataFrame, filename: str = 'lead_time_poc_data.csv'):
        """Save the generated data"""
        df.to_csv(filename, index=False)
        print(f"\nData saved to {filename}")
        
        # Save factor options for UI
        factor_options = self.get_factor_options()
        with open(filename.replace('.csv', '_factors.json'), 'w') as f:
            json.dump(factor_options, f, indent=2)
        
        print(f"Factor options saved to {filename.replace('.csv', '_factors.json')}")

def main():
    """Main function for POC data generation"""
    generator = LeadTimePOCGenerator()
    
    # Generate training data
    training_data = generator.generate_training_data(n_orders=5000)
    
    # Generate demo scenarios
    demo_scenarios = generator.generate_demo_scenarios()
    
    # Save training data
    generator.save_data(training_data, 'lead_time_poc_training.csv')
    
    # Save demo scenarios
    generator.save_data(demo_scenarios, 'lead_time_poc_scenarios.csv')
    
    # Display factor options
    print("\n" + "="*60)
    print("POC FACTOR OPTIONS (for user selection):")
    print("="*60)
    
    factor_options = generator.get_factor_options()
    for i, (factor, options) in enumerate(factor_options.items(), 1):
        print(f"\n{i}. {factor.replace('_', ' ').title()}:")
        for j, option in enumerate(options, 1):
            print(f"   {j}. {option}")
    
    print("\n" + "="*60)
    print("DEMO SCENARIOS:")
    print("="*60)
    print(demo_scenarios[['scenario_name', 'estimated_lead_time_days']].to_string(index=False))
    
    print(f"\nTraining data sample:")
    print(training_data[['product_type', 'quantity_range', 'customer_region', 
                        'season', 'estimated_lead_time_days']].head())

if __name__ == "__main__":
    main()