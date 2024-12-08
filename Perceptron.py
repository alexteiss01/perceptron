import math
def sigmoide(inp,lamb,der=False) : #la fonction d'activation sigmoïd et ça dérivée
    p=-inp*lamb
    if not der :
        try :
            return 1/(1+math.exp(p))
        except :
            if p>0 :
                return 0
            else :
                return 1
    else :
        try :
            e=math.exp(p)
            return (-lamb*e)/((1+e)**2)
        except :
            return 0

def redressement(inp,lamb,der=False) :
    p=inp*lamb
    if not der :
        if p<0 :
            return 0
        else :
            if p>1 :
                return 1
            else :
                return p
    if der :
        if p<0 or p>1 :
            return 0
        else :
            return lamb

class perceptron : #la classe du perceptron
    def __init__(self,lamb,bias,weight,act) : #lamb : lambda, un parametre de la fonction d'activation, bias : le biai, weight : liste du poids des connections, act : la fonction d'activation
        self.weight = weight
        self.bias = bias
        self.lamb = lamb
        self.act = act
    def getoutput(self,input) : #input est une liste, la liste des inputs
        Sum = 0
        for i in range(len(input)) : #applique les poids
            Sum=Sum+input[i]
        Sum=Sum+self.bias #ajoute le biai
        return self.act(Sum,self.lamb) #passe le tout dans la fonction d'activation
    def evaluate(self,input,expected) : #calcule l'erreur sur un output
        err=self.getoutput(input)
        return expected-err
    def moyevaluate(self,inputs,expecteds) :
        moy=0
        for i in range(len(inputs)) :
            moy=moy+self.evaluate(inputs[i],expecteds[i])
        moy=moy/len(inputs)
        return moy
    def learncycle(self,inputs,output,expected,step) : #un cycle d'apprentissage
        err=[]
        for i in range(len(output)) :
            err.append(self.evaluate(inputs[i],expected[i]))
        for i in range(len(err)) :
            a=0
            ERR=err[i]
            for j in range(len(inputs[i])) :
                a=a+inputs[i][j]*self.weight[j]
            ERR=ERR*self.act(a+self.bias,self.lamb,True)
            for j in range(len(self.weight)) :
                self.weight[j]=self.weight[j]+step*ERR*inputs[i][j]
            self.bias = self.bias+step*ERR
        
    def learn(self,inputs,expected,cycles,step) : #éxecute un certain nombre de cycle
        for i in range(cycles) :
            output=[]
            for inp in inputs :
                output.append(self.getoutput(inp))
            self.learncycle(inputs,output,expected,step)