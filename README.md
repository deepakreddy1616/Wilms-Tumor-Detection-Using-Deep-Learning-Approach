🚨 Real-Time Emergency Logistics Routing Using Metaheuristic Algorithms

[![Python](https://img.shields.io/badge/python-3.8+-blue?style=flat-square&logo[![NetworkX](https://img.shields.io/badge/NetworkX-3.2+-purple?style=flat-square[![OSMnx](https://img.shields.io/badge/OSMnx-1.3+-bright[![HERE Maps](https://img.shields.io/badge/HERE%20API-traffic-blue?style=flat-square&logo[![License](https://img.shields.io/badge/License-MIT-yellow[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square

🎯 Project Overview
A real-time, AI-driven logistics routing system designed for disaster and emergency response. Integrates live traffic data and Grey Wolf Optimization (GWO) for dynamically finding optimal vehicle delivery routes. Major goal: Minimize aid delivery delays and maximize route reliability during crisis scenarios.

Problem Statement
In disasters (earthquakes, floods), traditional logistics is disrupted by road closures and unpredictable events.

Routing must adapt instantly to new obstacles, congestion, and shifting demand.

Humanitarian supply chains require equitable, efficient, data-driven decision-making—beyond what classical algorithms provide.

Solution
This project combines real-time data streams (from HERE Maps API) and a metaheuristic GWO algorithm to automatically and adaptively generate robust, optimal routes, even as conditions change.

📊 Key Results
Metric	Value / Result	Status
Avg. Travel Time	294.96 units (synthetic)	⚡ Fast
Route Reliability	97% feasible routes	✅ Robust
Avg. Runtime (25 nodes)	0.0129 s	🚀 Scalable
Real-Time Data/Adaptivity	Supported	✅
Multi-Criteria Fitness	Time, congestion, safety	✅
✨ Features
✅ Real-time Routing
 - Live traffic, closures, hazards from HERE Maps API
✅ Metaheuristic Optimization
 - GWO algorithm for complex, uncertain environments
✅ Dynamic Adaptation
 - Instantly adjusts to new disruptions or demand
✅ Multi-Objective Fitness
 - Travel time, congestion, and road safety
✅ Scalability and Speed
 - Solves large city sub-graphs in milliseconds
✅ Humanitarian Focus
 - Fair resource allocation, disaster-mitigation design
✅ Visualizations
 - Route and convergence visualizations with Folium and Matplotlib

🛠️ Technology Stack
Component	Technology
Programming Language	Python 3.8+
Graph Library	NetworkX, OSMnx
Optimization	Grey Wolf Optimization (custom)
Data Sources	HERE Maps API, OpenStreetMap
Visualization	Folium, Matplotlib
Scientific Computing	Numpy, Pandas
🚀 Quick Start
Prerequisites
Python 3.8 or higher

pip

Git

HERE Maps API key (register free)

Installation
bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/emergency-logistics-routing.git
cd emergency-logistics-routing

# 2. Set up virtual environment
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Add your HERE Maps API key to .env
Basic Usage
python
from src.routing_system import EmergencyRoutingProblem, RealTimeGWO

problem = EmergencyRoutingProblem(
    city_name="Hyderabad, Telangana, India",
    num_locations=10
)
optimizer = RealTimeGWO(
    problem=problem,
    num_wolves=20,
    max_iter=50
)
best_route, best_time, convergence = optimizer.optimize()

print(f"Best route: {best_route}")
print(f"Total travel time: {best_time}")

📈 How It Works
Real-Time Data Ingestion: Retrieves live traffic, incident, and network info (HERE Maps)

Weighted Graph Construction: Models city as graph, weights edges with time, safety, and congestion

GWO Optimization: Simulated wolf “agents” iteratively search for route minimizing time/cost/safety penalty

Fitness Calculation: Multi-objective function scoring speed, congestion, hazards

Visualization: Best route and all metrics visualized (Folium, matplotlib)

🧪 Model Highlights
Grey Wolf Optimization: Population-based; uses alpha, beta, delta wolves as leaders, updating routes iteratively

Dynamic Response: Auto-recomputes routes on closure or incident triggers

Multi-metric Evaluation: Optimizes not just for speed but also safety and reliability

📄 Results & Evaluation
Key Metrics (Sample Synthetic Test)
Metric	Value
Travel Time	294.96 units
Route Reliability	97%
Runtime (25 nodes)	0.013 s
Scalability	Linear (w.r.t nodes)
Convergence curves show improvement at each GWO iteration

Visual route maps generated for each test scenario

See paper for full benchmarks and evaluations

💡 Real-World Impact
Designed for:

Disaster relief agencies & humanitarian NGOs

Urban planners & smart city logistics

Researchers in AI for critical infrastructure

📚 Documentation
For full methodology, literature survey, mathematical formulation, and ablation studies:
📄 paper/Project-Paper.pdf

🔗 Related Resources
HERE Maps API

Grey Wolf Optimization paper

OSMnx Docs

📝 License
This project is licensed under the MIT License – see LICENSE for details.

🌟 If you found this useful, please ⭐ the repo and cite our work!
Built for emergency logistics in a changing world, powered by real-time data, metaheuristics, and Python.
