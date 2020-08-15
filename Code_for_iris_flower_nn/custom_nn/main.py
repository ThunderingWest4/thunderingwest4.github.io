from defMain import run
import re
from os import system

running = True

prevLR = []
prevIter = []
prevAccur = []
first = True
clear = lambda: system('clear')

while running: 
  
  #giving previous accuracy/iter/LR to allow person to fiddle and possibly get more accurate net
  if not first: 
    for i in range(len(prevLR)):
      print("Simulation #{}: Accuracy {} with {} iterations and a learning rate of {}".format(i+1, prevAccur[i], prevIter[i], prevLR[i]))
    print("----------------------------------------------------------------------")
  
  print("Change the number of iterations to change how many times the network trains on the training set. Too many iterations can cause the network to overfit to the set and not notice the trends while too few can cause the network to underfit and not fully notice the trends")
  iterin = input("Enter the number of iterations that the network should train for (press enter for default: 5000): ")
  iterations = int(iterin) if iterin!="" else 5000
  
  print("Change the Learning Rate, also known as alpha value, to change the rate at which the neural network learns and trains. The rate is often set by default at 0.01 but can be adjusted to be higher or lower to optimize the network's training")
  lrin = input("Enter the learning rate of the network (press enter for default: 0.01): ")
  learning_rate = float(lrin) if (lrin!="" and (len(re.findall('[0-9]+', lrin))==(len(lrin)-1))) else 0.01
  
  print("Training for {} iterations with learning rate {}".format(iterations, learning_rate))
  acc = run(iters=iterations, alpha=learning_rate)
  print("--------------------------------------------------")
  again = input("Would you like to run another simulation with different settings? (Y/N)")
  running = True if again.lower() != "n" else False
  
  #all should be same length
  prevLR.append(learning_rate)
  prevIter.append(iterations)
  prevAccur.append(acc)
  
  first = False
  
  #ensure that screen is clear when we run this
  clear()
  
  