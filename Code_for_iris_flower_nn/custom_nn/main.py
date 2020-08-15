from defMain import run
import re

print("""
Change the number of iterations to change how many times the 
network trains on the training set. Too many iterations can cause the network to
overfit to the set and not notice the trends while too few can cause the network to 
underfit and not fully notice the trends
""")
iterin = input("Enter the number of iterations that the network should train for (press enter for default: 5000): ")
iterations = int(iterin) if iterin!="" else 5000

print("""
Change the Learning Rate, also known as alpha value, to change the rate at which the 
neural network learns and trains. The rate is often set by default at 0.01 but can be adjusted to be higher or lower to optimize the network's training
""")
lrin = input("Enter the learning rate of the network (press enter for default: 0.01): ")
learning_rate = int(lrin) if (lrin!="" and (len(re.findall('[0-9]+', lrin))==(len(lrin)-1))) else 0.01

print("Training for {} iterations with learning rate {}".format(iterations, learning_rate))
run(iters=iterations, alpha=learning_rate)