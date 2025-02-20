Il progetto consiste in un'implementazione di gossip learning utilizzando Node.js, PyTorch e Docker.

Per avviare il progetto è necessaria l'installazione sul sistema di Docker, Docker Compose, Node.js (versione 20 o superiore), Python, PyTorch e, opzionalmente, CUDA.

La rete neurale è costituita da 8 nodi, istanziati come container Docker e definiti nel file docker-compose.yml.
Attualmente, il dataset su cui viene effettuato l'apprendimento è GTSRB, definito all'interno del codice python nel file gossip_learning.js.
Per eseguire codice python su Node.js è stata usata la libreria python-bridge.
Per la comunicazione tra i nodi è stata usata la libreria libp2p.

Per avviare l'apprendimento occorre lanciare da riga di comando il comando docker-compose build e in seguito docker-compose up (eventualmente con l'opzione -d per eseguire i container in background). Per fermare e rimuovere i container occorre lanciare il comando docker-compose down.

Durante la build verranno scaricati sia i moduli di Node.js necessari per il progetto, sia il dataset su cui viene effettuato il training, salvati rispettivamente nelle cartelle node_modules/ e data/. Verrà inoltre creata una cartella models/, dove vengono salvati i modelli su cui i nodi effettuano il training.

AGGIORNAMENTO 19/11/2024: I problemi di connessione tra i peer sono stati risolti. Ho aggiunto dei log nella funzione client_update e nella creazione di train loader e test loader per verificare l'integrità dei dati e la correttezza dei valori. Sembra che il training venga effettuato correttamente.