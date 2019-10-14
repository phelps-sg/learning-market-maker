## Electronic market-making using reinforcement-learning

This repo contains a Python implementation of the basic model described in Chan, Nicholas Tung, and Christian Shelton.
"An electronic market-maker." (2001).

### Example usage:

~~~python
import matplotlib.pyplot as plt
%run src/main/python/chan-and-shelton.py
sim = MarketSimulation(ThresholdPolicy(threshold=2))
fundamental_price, mm_price, order_imbalances, rewards, actions = sim.run()
plt.plot(mm_price)
~~~