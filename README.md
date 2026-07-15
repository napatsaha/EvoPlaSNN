# EvoPlaSNN: Evolving Reward-modulated ANN-based Plasticity Rule for Spiking Neural Networks

Napat Sahapat, Sérgio F. Chevtchenko, Yeshwanth Bethi and Saeed Afshar

Main author of repository: Napat Sahapat

Presented in GECCO 2026

DOI: tba

---
# Introduction

Aim is to build a Spiking Neural Network (SNN) simulator from scratch, and apply a basic 
Evolutionary Algorithm to evolve the plasticity rules which control the weight updates of each SNN

---
# Overview

All of the code for the experiments are in the `src/` directory, which consists of the following modules:
- `common`: Contains helper functions and factory methods
- `evo`: All code relating to evolutionary optimisation
- `lrule`: Learning Rule representation
- `snn`: Spiking Neural Network, Simulator, Spike Coding, Neurons, Synapses and Plotting functions
- `rl`: Maze Environment for RL evaluation

---
