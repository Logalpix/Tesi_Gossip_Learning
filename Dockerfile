# Utilizza un'immagine base con Node.js
FROM node:20

# Installare Python e pip
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# Creare un link simbolico da python3 a python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Crea e imposta la directory di lavoro
WORKDIR /app

# Creare un ambiente virtuale Python
RUN python3 -m venv /app/venv

# Attivare l'ambiente virtuale e installare le dipendenze Python
RUN /app/venv/bin/pip install --upgrade pip
RUN /app/venv/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copia il package.json e package-lock.json nella directory di lavoro
COPY package*.json ./

# Installa le dipendenze del progetto
RUN npm install

# Copia il resto dei file del progetto nella directory di lavoro
COPY . .

# Espone la porta necessaria
EXPOSE 4000

# Impostare l'ambiente virtuale Python come variabile d'ambiente
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Comando per eseguire il programma
CMD ["node", "gossip_learning.js"]