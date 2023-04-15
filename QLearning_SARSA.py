import random
import copy

#Parsing the input file
def Parse(problem_file_name):
    file = open(problem_file_name, "r")
    txt = file.read()
    file.close()

    lines = txt.split('\n')

    alphaIndex = -1
    gammaIndex = -1
    epsilonIndex = -1
    episodeIndex = -1

    for i in range(0,len(lines)):
        if(lines[i] == "[alpha]"):
            alphaIndex = i
        elif(lines[i] == "[gamma]"):
            gammaIndex = i
        elif(lines[i] == "[epsilon]"):
            epsilonIndex = i
        elif(lines[i] == "[episode count]"):
            episodeIndex = i

    alpha = lines[alphaIndex+1]
    gamma = lines[gammaIndex+1]
    epsilon = lines[epsilonIndex+1]
    episodecount = lines[episodeIndex+1]
            
    return [float(alpha),float(gamma),float(epsilon),int(episodecount)]


#Game class
class TicTacToe:

    #Holds board and agents
    def __init__(this,X_Agent,O_Agent):
        this.board = [['-','-','-'],['-','-','-'],['-','-','-']]
        this.X_Agent = X_Agent
        this.O_Agent = O_Agent

    
    #For reset the game
    def Reset(this):
        this.board = [['-','-','-'],['-','-','-'],['-','-','-']]


    #Debugging purposes
    def Print(this):
        print("----------------------")
        print("Board")
        for i in range(0,3):
                print(this.board[i][0]+" "+this.board[i][1]+" "+this.board[i][2])
                
        print()
        winner = GetWinner(this.board)
        if(winner != 2):
            print("Game over")
            print("Winner : " , winner)
            
        print("----------------------")


    



    #Start game for episodecount
    def StartGame(this,episodecount):

        for i in range(0,episodecount):

            #use to observe episode process
            #if(i % 1000 == 0):
                #print("episode ",i)


            #Reset board
            this.Reset()

            #Tagging A,A',S,and S'
            previousActionX = this.X_Agent.GetMove(this.board)
            previousStateX = BoardToString(this.board)

            previousStateO = None

            #episode starts here
            while(True):

                #X takes action
                this.board[previousActionX[0]][previousActionX[1]] = 'X'

                #End check
                winner = GetWinner(this.board)
                if(winner != 2):
                    break

                newStateO = BoardToString(this.board)
                newActionO = this.O_Agent.GetMove(this.board)
                if(previousStateO != None):
                    #By observed A' and S' update table
                    this.O_Agent.Update(previousStateO,newStateO,previousActionO,newActionO,-winner)
                    
                    
                previousStateO = BoardToString(this.board)
                previousActionO = this.O_Agent.GetMove(this.board)
                #O Takes action
                this.board[previousActionO[0]][previousActionO[1]] = 'O'

                #End check
                winner = GetWinner(this.board)
                    
                if(winner != 2):
                    break
                
                newStateX = BoardToString(this.board)
                newActionX = this.X_Agent.GetMove(this.board)
                #By observed A' and S' update table
                this.X_Agent.Update(previousStateX,newStateX,previousActionX,newActionX,winner)
                previousStateX = newStateX
                previousActionX = newActionX



            #final update for end game
            this.X_Agent.Update(previousStateX,None,previousActionX,None,winner)
            this.O_Agent.Update(previousStateO,None,previousActionO,None,-winner)

            
                
#Some utility functions
def BoardToString(board):
    result = ""
    for i in board:
        for j in i:
            result += j
    return result


def StringToBoard(state):
    result = []
    for i in range(0,3):
        result.append([])
        for j in range(0,3):
            result[i].append(state[i*3+j])

    return result


#  1 -> X wins
# -1 -> O wins
#  0 -> Draw
#  2 -> Game is not over

def GetWinner(board):
    for i in range(0,3):
        first = board[i][0]
        second = board[i][1]
        third = board[i][2]

        if(first == second and second == third):
            sign = first
            if(sign == '-'):
                continue
            else:
                if(sign == 'X'):
                    return 1
                else :
                    return -1
            
    for i in range(0,3):
        first = board[0][i]
        second = board[1][i]
        third = board[2][i]
            
        if(first == second and second == third):
            sign = first
            if(sign == '-'):
                continue
            else:
                if(sign == 'X'):
                    return 1
                else :
                    return -1


    first = board[0][0]
    second = board[1][1]
    third = board[2][2]

    if(first == second and second == third):
        sign = first
        if(sign != '-'):
            if(sign == 'X'):
                return 1
            else :
                return -1


    first = board[0][2]
    second = board[1][1]
    third = board[2][0]

    if(first == second and second == third):
        sign = first
        if(sign != '-'):
            if(sign == 'X'):
                return 1
            else :
                return -1

    if(len(GetMovablePositions(board))==0):
        return 0

    return 2


def GetMovablePositions(board):
    lst = []
    for i in range(0,3):
        for j in range(0,3):
            if(board[i][j] == '-'):
                lst.append((i,j))
    return lst



#Super class for both SARSA and Qlearning
class Agent():
    def __init__(this,alpha,gamma,epsilon):
        this.QTable = {}
        this.alpha = alpha
        this.gamma = gamma
        this.epsilon = epsilon
        this.InitializeQTable()

    
    def InitializeQTable(this):
        allMoves = []
        for i in range(0,3):
            for j in range(0,3):
                allMoves.append((i,j))

        for move in allMoves:
            this.QTable[move] = []


    #Set reward value of action-state pair
    def UpdateQTable(this,action,state,reward):
        
        for val in this.QTable[action]:
            if(val[0] == state):
                this.QTable[action].remove(val)
                val = (val[0],reward)
                this.QTable[action].append(val)
                return

        print("Entry not found")


    #Given state action pair, get the reward from Q Table
    def GetReward(this,state,action):
        statesForAction = this.QTable[action]

        for val in statesForAction:
            if(val[0] == state):
                return val[1]

        this.QTable[action].append((state,0))

        return 0


    #Given state, get maximum reward (alternating actions)
    def GetMaximumReward(this,state):
        maximumReward = -9999999

        board = StringToBoard(state)

        possible_actions = GetMovablePositions(board)
        

        for action in possible_actions:
            
            exist = False

            statesOfAction = this.QTable[action]

            for val in statesOfAction:
                if(state == val[0]):
                    exist = True
                    
                    maximumReward = max(maximumReward,val[1])


            if(not exist):
                maximumReward = max(0,maximumReward)
                
        return maximumReward
        


    #Given state, get the best action (with maximum reward)
    def GetMoveFromQTable(this,board):

        possible_actions = GetMovablePositions(board)

        maxReward = -9999999
        maxAction = None

        key = BoardToString(board)

        for action in possible_actions:

            statesForAction = this.QTable[action]

            stateExist = False
            for state in statesForAction:
                if(state[0] == key):
                    stateExist = True
                    if(state[1] > maxReward):
                        maxAction = action
                        maxReward = state[1] 
                    break
                
            #If state-action pair not exists, insert with value 0
            if(not stateExist):
                this.QTable[action].append((key,0))
                if(maxReward < 0):
                    maxReward = 0
                    maxAction = action

        return maxAction
        

    #By epsilon probability, get a random possible action
    #Otherwise get the best action from table
    def GetMove(this,board):
        if(random.random() <= this.epsilon):
            possible_actions = GetMovablePositions(board)
            selected_action = possible_actions[random.randint(0,len(possible_actions)-1)]
            return selected_action

        return this.GetMoveFromQTable(board)
        



class Q_Agent(Agent):

    def __init__(this,alpha,gamma,epsilon):
        super().__init__(alpha,gamma,epsilon)
        this.alpha = alpha
        this.gamma = gamma
        this.epsilon = epsilon


    #Table update for Q Learner (Simply the formula)
    def Update(this,previousState,newState,previousAction,newAction,reward):
        Q_S_A = this.GetReward(previousState,previousAction)
        
        if(newState != None):
            Q_S_A_2 = this.GetMaximumReward(newState)
            newReward = Q_S_A + this.alpha * (reward + this.gamma*Q_S_A_2 - Q_S_A)
        else:
            newReward = Q_S_A + this.alpha * (reward - Q_S_A)
        
        this.UpdateQTable(previousAction,previousState,newReward)

        
    
class SARSA_Agent(Agent):

    def __init__(this,alpha,gamma,epsilon):
        super().__init__(alpha,gamma,epsilon)
        this.alpha = alpha
        this.gamma = gamma
        this.epsilon = epsilon

    #Table update for SARSA Learner (Simply the formula)
    def Update(this,previousState,newState,previousAction,newAction,reward):

        Q_S_A = this.GetReward(previousState,previousAction)

        if(newState != None):
            Q_S_A_2 = this.GetReward(newState,newAction)
            newReward = Q_S_A + this.alpha * (reward + this.gamma*Q_S_A_2 - Q_S_A)
        else:
            newReward = Q_S_A + this.alpha * (reward - Q_S_A)
        
        
        this.UpdateQTable(previousAction,previousState,newReward)

        

def SolveMDP(method_name,problem_file_name,random_seed):

    #parsing input
    inputs = Parse(problem_file_name)
    alpha = inputs[0]
    gamma = inputs[1]
    epsilon = inputs[2]
    episodecount = inputs[3]
    #-----------
    random.seed(random_seed)



    #Initialize agents
    SA = SARSA_Agent(alpha,gamma,epsilon)
    QA = Q_Agent(alpha,gamma,epsilon)

    #assign SARSA to X, QLearning to O
    if(method_name == "SARSA"):
        X_Agent = SA
        O_Agent = QA

    #assign QLearning to X, SARSA to O
    else:
        X_Agent = QA
        O_Agent = SA


    #Initialize game
    game = TicTacToe(X_Agent,O_Agent)

    #Start game for episode count given agents
    game.StartGame(episodecount)


    #Return the QTable used by X Player
    return X_Agent.QTable


#Outputs the tables into corresponding txt files

"""
Q_table = SolveMDP("SARSA","mdp1.txt",37)
with open('output_SARSA.txt', 'w') as f:
    f.write(str(Q_table))
    
Q_table = SolveMDP("Q-learning","mdp1.txt",462)
with open('output_Qlearning.txt', 'w') as f:
    f.write(str(Q_table))
"""


