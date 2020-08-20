import random
import time

def disp(string, *argv):
        fin = string
        for i in argv:
            fin += str(i)
        for c in fin:
            print(c, end='')
            time.sleep(0.04)
        #print(fin)
        print("")


class env():

    Map = {1 : "", 2 : "", 3 : "", 4 : "", 5 : "", 6 : "", 7 : "", 8 : "", 9 : "", 10 : "", 11 : "", 12 : "", 13 : "", 14 : "", 15 : "", 16 : "", 17 : "", 18 : "", 19 : "", 20 : ""}
    Pits = []
    Bats = []

    def env(self, starting):
        self.Start = starting
        self.genBat()
        self.genPit()

    def genWumpus(self, start):
        loc = start
        self.Start = start
        self.Map[self.Start] = "s"
        while(loc == start):
            loc = random.randint(1, 20)
        loc = int(loc)
        #self.Map[loc] = "w"
        
        return loc
        

    numPit = 5
    numBat = 3

    def genBat(self):
        for i in range(self.numBat):
            found = False
            while(found != True):
                #print("bat generating")
                test = random.randint(1, 20)
                #print(self.Bats)
                if(self.Map[test] == ""):
                    self.Bats.append(test)
                    self.Map[test] = "b"
                    found = True
            

    def genPit(self):
        for i in range(self.numPit):
            found = False
            while(found != True):
                #print("pit generating")
                test = random.randint(1, 20)
                if(self.Map[test] == ""):
                    self.Pits.append(test)
                    self.Map[test] = "p"
                    found = True

 