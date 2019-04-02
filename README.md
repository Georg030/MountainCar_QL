# MountainCar_QL

### Problemstellung - MountainCar
MountainCar ist ein Reinforcement Learning Problem, welches in der PhD Thesis von Andrew Moore entstanden ist.
Die Lösung des Problems ist simpel: Ein Auto muss von einen Tal aus einen Hügel hinnauffahren.
Da Schwerkraft auf das Auto wirkt und der Motor des Fahrzeugs eine zu geringe Kraft aufbringt, kann das Fahrzeug allerdings nicht einfach den Hang hinauf beschleunigen.
Das Auto muss also lernen Schwung zu gewinnen, indem es Richtung des gegenüber liegenden Hüges fährt bevor es das Ziel erreichen kann.
<br>

Das Toolkit Gym stellt für die Bearbeitung des Problems eine Environment bereit. (https://github.com/openai/gym/wiki/MountainCar-v0)
Das Fahrzeug hat drei mögliche Aktionen: Nach rechts fahren, nach links oder nichts machen.
Die Umgebung führ die Aktion aus und gibt als Beobachtungswerte die Position und die Geschwindigkeit aus, als auch einen Reward.
Als Reward wird nach jeder Aktion -1 zurückgegeben wenn der Gipfel bei Position 0.5 nicht erreicht wurde, 
wird er erreicht ist der Reward 1. Für meine Lösung habe ich allerdings eine andere Rewardfunktion gewählt, dazu später mehr.
<br>
<br>
![alt text](https://raw.githubusercontent.com/georg030/MountainCar_QL/master/pictures/Actions.png)  
<br>
![alt text](https://raw.githubusercontent.com/georg030/MountainCar_QL/master/pictures/Observation.png)


### Q-Learning und die Fragestellung
Q-Learning ist eine Technik des Reinforcement-Learning, dessen Ziel es ist eine Optimale Policy zu erlernen.
Die optimale Policy soll immer die Action-Value Funktion bereitstellen, 
die in anbetracht der darauf resultierenden States und Actions den höchsten Reward verspricht.
Die Action-Value Funktion oder auch Q(a,s) gibt hierbei einen Q-Value aus,
welcher dem Value/Reward einer durchgeführten Aktion entpricht 
Die Policy ist hierbei ein Neuronales Netz, dessen Weights trainiert werden, um bei resultierenden States als Input die optimalen Q(a,s), bzw. die Aktionen mit maximalen Q-Values als Output zu erhalten. 
<b>
Um das Mountain Car Problem zu lösen werde ich das Q-Learning mittels der Implementierung der Bellmann Equation umsetzen. Anschließend will ich Experience Replay als Q-Learning Erweiterung implementieren und das standart Q-Learning, mit dem mit Experience Replay erweiterten, gegenüberstellen. Es stellen sich jetzt natürlich die Fragen was Bellmann Equation und Experience Replay beudeuten, dazu werde ich bei der Implementierung genauer eingehen. 
  

### Implementierung

Für die Implementierung habe ich PyTorch genutzt, eine Open-Source Machine Learning Bibliothek für Python.
Kern der Fragestellung ist die Implementierung von Q-Learning und der Erweiterung mit Replay Experience. Hpyerparameter- und Netzwerktuning habe ich nur bis zu dem Punkt vorgenommen, bei dem beide Implementierungen unter gleichen Bedingungen funktionieren. 
