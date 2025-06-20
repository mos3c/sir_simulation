SIR Epidemic Transmission Model - Interactive Simulator
An interactive web application that simulates disease spread through populations using the mathematical SIR (Susceptible-Infected-Recovered) epidemiological model.

🎯 Overview
This educational tool demonstrates how infectious diseases spread through populations using the classic SIR compartmental model. Users can adjust various parameters to see how they affect epidemic dynamics in real-time

✨ Features
📊 Three Visualization Modes

Static Graph: Complete epidemic curve showing the full simulation
Animated Graph: Progressive build-up of the epidemic curve day by day
People Simulation: Visual animation with moving dots representing individuals

🎛️ Interactive Parameters

Population Size (N): Total number of people in the simulation
Initial Infected (I₀): Number of infected people at the start
Transmission Rate (β): Rate of disease transmission per contact
Recovery Rate (γ): Rate at which infected people recover
Simulation Duration: Number of days to simulate

📈 Real-Time Metrics

R₀ (Basic Reproduction Number): Average number of secondary infections
Peak Infections: Maximum number of infected individuals
Peak Day: When the infection peak occurs
Final Recovered: Total number of people who got infected

🎮 Control Features

Start/Pause/Reset simulation controls
Adjustable animation speed
Real-time population counters
Interactive parameter adjustment

🧮 Mathematical Model
The SIR model uses these differential equations:

dS/dt = -βSI/N
dI/dt = βSI/N - γI
dR/dt = γI

Where:

S: Susceptible population
I: Infected population
R: Recovered population
β: Transmission rate
γ: Recovery rate
N: Total population (S + I + R = N)
























