## Electronic market-making using reinforcement-learning

This repo contains a Python implementation of the basic model described in Chan, Nicholas Tung, and Christian Shelton.
"An electronic market-maker." (2001).

### Example usage:

~~~python
import matplotlib.pyplot as plt
%run src/main/python/chan-and-shelton.py
learner, results = learning_experiment()
print("Greedy policy is ", learner.greedy_policy())
print("Q table: ", learner.as_DataFrame())
p_f, p_mm, imb, r, a = results
plt.figure()
plt.plot(p_f)
plt.plot(p_mm)
plt.show()
~~~