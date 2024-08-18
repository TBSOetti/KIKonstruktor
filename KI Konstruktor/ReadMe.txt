class PEncoder : 

            PEncoder oder PositionalEncoding(nn.Modul) ist für die Positionskodierung zuständig, da Transformermodelle keine Reihenfolge oder Sequenzen besitzen

PE       

            pe erstellt einen Container mit einem null Tensor, damit alles mögliche gespeichert werden kann
            Die Dimension des Tensors wird anhand seiner inputmöglichkeiten berechnet, in dem Fall mit der Wörterlänge (Sequenz) 5000 und Der Dimension d_model
            wie das Wort angezeigt wird in der Dimension, zB. in dem Fall Katze [1.2,2.4] wöäre es die zweite Dimension
            JE länger die Dimension, desto mehr Inhalt und Verbindungen können gespeichert werden, Beispiel siehe ReadMe     


Position 

            torch.arange hat die gleiche Funktion wie Range, nur als Tensor geschrieben 
            0 ist der Startwert 
            max_len, maximale Sequenzlänge 
            dtype=torch.float Werte werden als Gleitkommazahlen dargestellt 
            unsqueeze ist die Methode mit der eine neue Dimension erstellt wird, also von 1D zu 2D 


div_term 

            Zur Berechnung von Sinus und Kosinus, damit die relative Position von Wörtern erkannt wird selbst wenn die Wortreihenfolge nicht direkt im Modell kodiert ist
            float Wandelt Zahlen in Fließkommazahlen um da man eine Fließkommadivision benötigt wird 
            torch.log Der Logarhymthmus wird berechnet um die Sinus und Kosinuswellen zu variieren, welche für die PEncoder Berechnung essenziell ist 


Register_Buffer 

            Tensor pe wird gespeichert, damit er während des Trainings nicht verändert werden kann aber aufgenommen wird.
            pe wird ein Teil des Modells, allerdings wird es nicht benutzt um das Modell zu trainieren, es wird kein Parameter 


MultiHead 

            Attention Mechanismus ist eine Berechnungsart um verschiedenen Schlüsselwörter eine Gewichtung zu geben, damit der AttentionScore berechnet werden kann 
            Multihead ist genau das gleiche, allerdings werden verschiedene Werte berechnet um den AttentionScore zu verteilen, als Beispiel 
            Eine Katze klettert auf den Baum, Katze und Baum hätten z.B. die Werte Lebewesen, klettern hat das Attribut etwas tun usw 
            Mathematisch kein Bock zu erklären 


assert d_model

            Stellt sicher, das die Dimensionen gleichmäßig auf die Köpfe aufgeteilt werden kann 


Linear Q V K 

            Unterschiedliche Gewichtungen Dadurch lernt das System das es unterschiedliche Gewichtungen herrschen, gleichzeitig lernt das Programm in welcher Beziehung 
            und Gewichtungen Dinge zueinander stehen 
            Q = Query V = Value K = Key 


MHA     

            Query, Key und Value werden in Köpfe aufgeteilt 
            scores wird durch multiplikation von Query und Key berechnet 
            Man kann eine Maske anwenden um Positionen zu ignorieren
            Attention wird auf Value angewendet und zum Schluss kombiniert und transformiert 


FForward 

            Implementier ein zweischichtiges Neuronales Netzwerk, das auf jeden Token und Sequenz unabhängig angewendet wird 
            Erste Lineare Transformation ist linear1, wird auf die höhere Dimension d_ff projeziert 
            Dropout soll die Überregulierung verhindern 
            linear2 wird wieder auf die Ausgabe d_model projeziert 
            Relu wird benötigt um Muster zu erkennen, da es ansonsten eine lineare Ausgabemethode verwendet 


TransformerEncoderLayer 

            Eine Schicht vom Transformer Encoders der aus der SAttention und der FForward besteht
            self_attn und feed forward wird implementiert 
            norm ist fpr die Verteilung zuständig um Aktivierungen zu stabilisieren 


src 

            Zuerst wird die Eingabe normalisiert, dann wird self_attn drübergezogen 
            Eingabe wird addiert, dropout kommt als Wert hinzu 
            Dropout lässt bestimmte neuronale Netzwerke ausschalten um einer Überanpassung vorzubeugen dies hat die Vorteile 
            das zum einen das System robuster wird und dadurch weitere neuronale Netzwerke angleicht durch die zufälligkeit 
            sowie das das System  nicht zwangsweise abhängig von einzelnen Neuronen wird 



            TODO Liste 

            Fertigstellung des Grundgerüstes (TransformerEncoder bauen sowie das EIngabe Ausgabe Embedding und die Module in die Main.py einbauen)
            überarbeitung des Codes 
            Sicherstellung das das System nach Grundgerüst weiter ausgebaut und einzatzfähig gemacht wird 


