## Robust Supply Chain Networks with Monte Carlo Simulation 🏭
*Do you consider the fluctuation of your demand when you design a Supply Chain Network?*

<p align="center">
  <img align="center" src="https://miro.medium.com/max/1400/1*ygvI_dS3-aJ59DGAfplXnA.png">
</p>

### Objective
Build a simple methodology of Supply Chain Network Design that is considering the fluctuation of the demand.

### Introduction
Supply chain optimization makes the best use of data analytics to find an optimal combination of factories and distribution centres to meet the demand of your customers.

In many software and solutions in the market, the core structure behind is a Linear Programming Model.
Some of these models find the right allocation of factories to meet the demand and minimize the costs assuming a constant demand.

*What happens if the demand is fluctuating?*

Your network may lose robustness, especially if you have a very high seasonality of your demand (e-commerce, cosmetics, fast fashion).

### Article
In this [Article](https://towardsdatascience.com/robust-supply-chain-network-with-monte-carlo-simulation-21ef5adb1722), we will build a simple methodology to design a **Robust Supply Chain Network** using **Monte Carlo** simulation with Python.

### 📘 Your complete guide for Supply Chain Analytics
60+ case studies with source code, dummy data and mathematical concepts here 👉 [Analytics Cheat Sheet](https://bit.ly/supply-chain-cheat)

### Youtube Video
Click on the image below to access a full tutorial video to understand the concept behind this solution
<div align="center">
  <a href="https://www.youtube.com/watch?v=gF9ds3CH3N4"><img src="https://www.samirsaci.com/content/images/2023/02/Supply-Chain-Optimization.png" alt="Explainer Video Link"></a>
</div>

### Scenario
As the Head of Supply Chain Management of an international manufacturing company, you want to redefine the Supply Chain Network for the next 5 years.

#### Demand
It starts with the demand from your customers in 5 different markets (Brazil, USA, Germany, India and Japan).
<p align="center">
  <img align="center" src="https://miro.medium.com/max/900/1*kaitTBi4zOqq2nUarEa9Bg.png">
</p>

#### Supply Capacity
You can open factories in the five markets. There is a choice between low and high-capacity facilities.
<p align="center">
  <img align="center" src="https://miro.medium.com/max/1030/1*5_ZYKy3NlszS6uV2IiSadQ.png">
</p>

#### Objective: minimize the total cost of production and shipment
The objective is to design  a new transportation plan to increase the average size of trucks by delivering more stores per route.
<p align="center">
  <img align="center" src="https://miro.medium.com/max/1400/1*QvlfMEtHPS9aq5lCLfc1bQ.png">
</p>
                                                                                               
#### Demand Fluctuation
In this solution we will consider a fluctuating demand (Normal Distribution) per market.
<p align="center">
  <img align="center" src="https://miro.medium.com/max/1400/1*w6RHuzcgKzRFUicusEPgLg.png">
</p>

#### Methodology
We'll run 50 scenarios and run a solver to find the optimal network.
<p align="center">
  <img align="center" src="https://miro.medium.com/max/1400/1*2cmp3ZRNHwMarV_2a0He1g.png">
</p>

##### Solution
We'll then study the split of solutions and take the one that appears the most.
<p align="center">
  <img align="center" src="https://miro.medium.com/max/908/1*KZxf6N2-RlhIaV_zuzMSlA.png">
</p>
                                                                                               

## Code
This repository code you will find all the code used to explain the concepts presented in the article.

## About me 🤓
Senior Supply Chain Engineer with an international experience working on Logistics and Transportation operations. \
Have a look at my portfolio: [Data Science for Supply Chain Portfolio](https://samirsaci.com) \
Data Science for Warehousing📦, Transportation 🚚 and Demand Forecasting 📈 
