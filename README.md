
## Problemstellung - MountainCar
MountainCar ist ein Reinforcement Learning Problem, welches in der PhD Thesis von Andrew Moore entstanden ist.
Das Problem ist simpel: Ein Auto muss von einen Tal aus einen Hügel hinnauffahren.
Da Schwerkraft auf das Auto wirkt und der Motor des Fahrzeugs eine zu geringe Kraft aufbringt, kann das Fahrzeug allerdings nicht einfach den Hang hinauf beschleunigen.
Das Auto muss also lernen Schwung zu gewinnen, indem es Richtung des gegenüber liegenden Hüges fährt bevor es das Ziel erreichen kann.
<br>

Das Toolkit Gym stellt für die Bearbeitung des Problems eine Environment bereit. (https://github.com/openai/gym/wiki/MountainCar-v0)
Das Fahrzeug hat drei mögliche Aktionen: Nach rechts fahren, nach links fahren oder nichts machen.
Die Umgebung führt die Aktion aus und gibt Position und die Geschwindigkeit des Autos als Beobachtungswerte, sowie einen einen Reward, aus.
Als Reward wird nach jeder Aktion -1 zurückgegeben wenn der Gipfel bei Position 0.5 nicht erreicht wurde, 
wird er erreicht ist der Reward 1. Für meine Lösung habe ich allerdings eine andere Rewardfunktion gewählt, dazu später mehr.
<br>
<br>
![alt text](https://raw.githubusercontent.com/georg030/MountainCar_QL/master/pictures/Actions.png)  
<br>
![alt text](https://raw.githubusercontent.com/georg030/MountainCar_QL/master/pictures/Observation.png)


## Q-Learning und die Fragestellung
Q-Learning ist eine Technik des Reinforcement-Learning, dessen Ziel es ist eine Optimale Policy zu erlernen.
Die optimale Policy soll immer die Action-Value Funktion bereitstellen, 
die in anbetracht der darauf resultierenden States und Actions den höchsten Reward verspricht.
Die Action-Value Funktion oder auch Q(a,s) gibt hierbei einen Q-Value aus,
welcher dem Value/Reward einer durchgeführten Aktion entpricht 
Die Policy ist hierbei ein Neuronales Netz, dessen Weights trainiert werden, um bei resultierenden States als Input die optimalen Q(a,s), bzw. die Aktionen mit maximalen Q-Values als Output zu erhalten. 
<br>
Für die Lösung des MountainCar Problems werde ich das Q-Learning mittels der Implementierung der Bellman Equation umsetzen. Anschließend will ich Experience Replay als Q-Learning Erweiterung implementieren und das standart Q-Learning, mit dem mit Experience Replay erweiterten, gegenüberstellen. Es kann sich jetzt natürlich die Frage stellen was Bellman Equation und Experience Replay bedeuten, darauf werde ich bei der Implementierung genauer eingehen.


## Implementierung
Für die Implementierung habe ich PyTorch genutzt, eine Open-Source Machine Learning Bibliothek für Python.
Kern der Fragestellung ist die Implementierung von Q-Learning und der Erweiterung mit Replay Experience. Hpyerparameter- und Netzwerktuning habe ich nur bis zu dem Punkt vorgenommen, bei dem beide Implementierungen unter gleichen Bedingungen funktionieren. Da das Tuning nicht Schwerpunkt dieser Arbeit ist werde ich wenn nur geringfügig darauf eingehen.

#### Netzwerk
Das Netzwerk ist ein einfaches Fully-Connected Network mit einen einzigen Hiddenlayer mit 100 Neuronen. Es gibt zwei Input Neuronen für die oberservierten States (Position/Velocity) und drei Output Neuronen für die möglichen Aktionen (links, nichts tun, rechts). 
Die weights sind mit der Default Standartverteilung von Torch initialisiert. Ich habe auch eine Initialisierung mit Xavier getestet, durch diese hat das Netzwerk zwar früher konvergiert aber final schlechtere Resultate erziehlt.
```python
class NN (nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.actions_size = 3
        self.states_size = 2
        self.hidden1 = 100
        # fully connected layer
        self.fc1 = nn.Linear(self.states_size, self.hidden1, bias=False)
        self.fc2 = nn.Linear(self.hidden1, self.actions_size, bias=False)

        #xavier weight initialisation
        # torch.nn.init.xavier_uniform(self.fc1.weight)
        # torch.nn.init.xavier_uniform(self.fc2.weight)


    def forward (self, input):
        model = torch.nn.Sequential (
            self.fc1,
            self.fc2
        )
        return model(input)

 ```

#### Interaktion mit der Environment

Die Interaktion mit der Environment stellt sich Episodisch dar. Wobei eine Episode in meiner Implementation aus Maximal 500 Schritten/Steps besteht. Die Environment wird resettet wenn es das Fahrzeug zum Ziel schafft oder die 500 Steps erreicht wurden, was dazu führt das das Fahrzeug wieder auf die Startposition zurückkehrt und die nächste Episode/Run beginnt. Jeder Schritt beginnt mit der Obversvation der Environment, man erhält also den aktuellen Status mit Position. Die Policy also das Netzwerk bekommt den State als Input und gibt daraufhin eine Q-Value Funktion(Q_0) zurück, ein Array mit drei Werten die die Values der drei möglichen Aktionen sind. Ein logischer Schritt wäre es nun die Aktion zu wählen die den höchsten Value/Reward verspricht(max(Q_0)), was sich auch Exploitation also Ausbeutung nennt. Da wir aber noch keine optimale Policy haben und die Values nicht den tatsächlichen Rewards entsprechen, ist es notwendig das für die Entdeckung möglicher höherer Values ein Teil der Aktionen Zufällig gewählt wird. Dieser Teil nennt sich Exploration (Erkundschaften). Es gibt also ein Abwägen zwischen Exploration und Exploitation welches durch den Parameter Epsilon bestimmt wird. Die Beeinflussung der Exploitation durch Epsilon wird Epsilon-Greedy (Greedy für Gier) genannt. In meiner Implementierung beträgt Epsilon den Wert 0.4 und besagt, dass 40% der Aktionen zufällig gewählt werden also der Exploration dienen. Der Wert von Epsilon sinkt allerdings nach jeder Erfolgreichen Episode. 
<br>
```python
for run in trange(RUNS):
    state_0 = env.reset()

    for step in range(STEPS):
        # get action-value function of state_0
        Q_0 = p_network(Variable(torch.from_numpy(state_0).type(torch.FloatTensor)))

        # epsilon probability of choosing a random action
        if np.random.rand(1) < epsilon:
            action = np.random.randint(0, 3)
        else:
            # choose max value action
            _, action = torch.max(Q_0, -1)  # returns values, indices
            action = action.item()
        # make next step and receive next state and reward, done true when successfull
        state_1, _, done, _ = env.step(action)
        
        # Rewardfunction: 
        # get reward based on car position
        reward = state_1[0] + 0.5
        # increase reward for task completion
        if state_1[0] >= 0.5:
            reward += 1
```
Der Reward ist bei Default -1 für jeden Step der das Ziel nicht erreicht hat und +1 wenn das Ziel erreicht wurde. Ich habe eine eine Rewardfunktion verwendet die die aktuelle Position berücksichtigt, da bei einem Reward von gleichbleibenden -1 kein lernen Stattfindet bis das Ziel durch Zufall erreicht wurde, was nur im seltenen Fall gelingt.
Diese hat sich bereits bei einer Implementierung erwiesen
[(hier der link)](https://medium.com/@ts1829/solving-mountain-car-with-q-learning-b77bf71b1de2#5abe).


### Optimierung mit Q-Learning

Jetzt wissen wir nach welcher Regel in jedem Step die nächste Aktion gewählt wird. Als nächstes muss die Bellman Equation erfüllt werden.
<br>
![screenshot](https://raw.githubusercontent.com/georg030/MountainCar_QL/master/pictures/BellmanEquation.png)
<br>
Denn die besagt, dass die Policy (NN) optimal werden muss, also Action-Values ausgibt die den jetzigen Reward addiert mit maximalen Value des nächsten States (mit einen Rauschen Gamma), entsprechen. Ist dies gegeben wird zu jedem State das auswählen der Aktion mit maximalen Value schließlich zum Erfolg also dem höchst möglichen Reward führen, was in unseren Fall der Gipfel ist.



Die Umsetzung lässt sich einfacher Anhand meiner Implementierung und Kommentare nachvollziehen:
```python
def optimize(Q_0):
    # get action-value function for state + 1
    Q_1 = p_network(Variable(torch.from_numpy(state_1).type(torch.FloatTensor)))
    # take action with max Value of Q_1
    maxQ_1, _ = torch.max(Q_1, -1)

    # Q-Target as copy of Q_O (Q_0 is the action value function of state 0)
    target_Q = Variable(Q_0.clone())
    
    #change max value action to satisfy Bellman Equation
    # -> Muliply the highest Action Value of Q_1 with Gamma and add received reward for current state
    target_Q[action] = reward + torch.mul(maxQ_1.detach(), GAMMA)

    # Calculate loss between Q_0 and Q-Target (Mean Squarred Error)
    loss = loss_function(Q_0, target_Q)

    # train model (backpropagation with loss) 
    # so that Q_0 approximates Q-Target
    p_network.zero_grad()
    loss.backward()
    optimizer.step()

```
Durch Q-Learning entsteht eine sehr hohe korrelation der aufeinanderfolgenden Aktionen, dies führt zu einen ineffizienten lernen. Der Gegenwärtige State bestimmt hierbei den nächsten wenn wir immer den maximalen Value folgen. Ist bei einen nicht optimalen Netzwerk die maximale Aktion immer "nichts tun", was dazu führen kann das man in einen schlechten Feedback Schleife hängenbleibt (Bad Feedback Loop).

### Experience Replay
Experience Replay (ER) ist eine Erweiterung des Q-Learnings. Hierbei wird wie vorher auch die Aktion mit Epsilon-Greedy gewählt. Allerdings wird die Experience in form einer Transition(State, Action, Reward, State +1) in jedem Schritt gespeichert.  
Um die Policy zu trainieren wird bei jedem Schritt ein Batch (hier Größe von 128) einer zufälligen Teilmenge aus dem Replay Memory genommen. Nun berechnet man für jede Experience wie vorher das Q-Target und den Loss zwischen Q-Target und Q-0. Anzumerken ist, dass es diesmal zwei Netzwerke gibt. Zusätzlich zu dem Policy-Netzwerk gibt es ein Target-Netzwerk, welches den Q-Value des States + 1 berechnet. Das Target-Netz ist eingefroren und übernimmt in Abstand von mehreren Schritten (hier 10) die Weights des Policy-Netzwerks. Dies gibt den Algorithmus eine höhere Stabilität. 
```python
def optimize_with_ER():
    if len(R_MEMORY) > BATCH_SIZE:
        # takes batch from Memory
        transitions = R_MEMORY.sample(BATCH_SIZE)
        batch = R_MEMORY.Transition(*zip(*transitions))
        states_0 = torch.stack(batch.state)
        actions_0 = torch.tensor(batch.action).view(BATCH_SIZE, 1)
        rewards = torch.tensor(batch.reward)
        states_1 = torch.tensor(batch.next_state)

        # get Q-Values according to taken actions(with epsilon-greedy)
        max_Qs_0 = p_network(states_0.float()).gather(1, actions_0)
        # get Q-Values from next state + 1 from from target-network
        max_Qs_1 = target_network(states_1.float()).max(1)[0]

        # Compute the expected Q values
        target_Qs = rewards + (max_Qs_1 * GAMMA)
        # calculate loss
        loss = loss_function(max_Qs_0, target_Qs.unsqueeze(1))

        #otimize network with brackpropagation
        p_network.zero_grad()
        loss.backward()
        optimizer.step()
```
### Vorteile von Experience Replay 
Das lernen von aufeinanderfolgenden Aktionen ist durch die hohe Korrelation problematisch und führt zu einen uneffizienten Lernen. Außerdem können dabei, die bereits erwähnten, "Bad Feedback Loops" entstehen. 
<br>
Indem die Erfahrungen beibehalten werden verhindern wir, dass das Netzwerk nur davon lernt, was es unmittelbar in der Umgebung tut. Es ermöglicht von einer Array von Erfahrungen zu lernen und dies bei jedem Schritt, wohingegen das standart Q-Learning nur von einer Aktion lernt. Experience Replay ist deshalb effizienter als das Einfache Q-Learning und kann auch mit weniger oder sich wiederholenden Daten besser Lernen. Die Gefahr von Bad Feedback Loops wird Aufgrund der geringen Korrelation der Erfahrungen verringert. Experience Replay führt ausserdem zu einer besseren Konvergenz und erziehlt daher bessere Ergebnisse. 

## Ergebnis
Erklärung zu den Grafiken: Die blauen Linien Zeigen an wieviele Schritte in einer Episode benötigt wurden um das Ziel zu erreichen. Geht die Linie bis zur 500 wurde das Ziel nicht erreicht. Die Organgene Kurve zeigt die durchschnittlichen Schritte der letzten 100 Episoden an. Das Ziel im Durschnitt von 90 Schritten zu erreichen würde ich als Optimum betrachten. 
<br>
![alt text](https://raw.githubusercontent.com/georg030/MountainCar_QL/master/pictures/basicResult.png)  
<br>
Bei des Basic Implementation zeigt sich dass auch das gelernte Modell noch häufig das Ziel innerhalb der 500 Schritte nicht erreicht und ist damit noch nicht sehr stabil. Dies zeigt sich auch durch die verschlechterung der Erfolgsrate innerhalb der letzten 400 Episoden. Generell ist bei einer durchschnittlichen Erfolgsrate von 182 in den letzten 200 Schritten, das optimum bei weitem verfehlt. 
<br>
<br>
![alt text](https://raw.githubusercontent.com/georg030/MountainCar_QL/master/pictures/ERResult.png)  
<br>
Die Implementation mit Replay Experience zeigt deutlich bessere Ergebnisse. Es ist wesentlich stabiler und hat nur in seltenen Fällen Auswüchse. Das Trainierte Modell erreicht immer das Ziel und ist mit einer durchschnittlichen Erfolgsrate von 122 nahe dem Optimum. 




