import process from 'node:process'
import { createLibp2p } from 'libp2p'
import { tcp } from '@libp2p/tcp'
import { mplex } from '@libp2p/mplex'
import { noise } from '@chainsafe/libp2p-noise'
import { peerIdFromString } from '@libp2p/peer-id'
import { multiaddr } from 'multiaddr'
import { mdns } from '@libp2p/mdns'
import { pipe } from 'it-pipe'
import toBuffer from 'it-to-buffer'
import pythonBridge from 'python-bridge'

import * as os from "os"
import fs from 'fs'
import { createHash } from 'node:crypto'

import AsyncLock from 'async-lock'

const lock = new AsyncLock()
const path_dir_models = 'models/'
var num_send_to_do = 1
var num_of_known_peers = 0

var peer_id_known_peers = []
var age_local_model = 0
var index_training = 0

const NUM_ROUNDS = 150

function get_position_str(string, subString, index) {
	return string.split(subString, index).join(subString).length
}

function get_ip_addr(){
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
      for (const net of interfaces[name]) {
          // Skip over non-IPv4 and internal (i.e., 127.0.0.1) addresses
          if (net.family === 'IPv4' && !net.internal) {
              return net.address;
          }
      }
  }
  throw new Error('Nessun indirizzo IP valido trovato');
}

function get_peerid_from_multiadd(multiadd){
	return multiadd.substring(get_position_str(multiadd, '/', 6)+1, get_position_str(multiadd, '/', 7))
}

function create_random_str(length) {
    let result = ''
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    const charactersLength = characters.length
    let counter = 0
    while (counter < length) {
      result += characters.charAt(Math.floor(Math.random() * charactersLength))
      counter += 1
    }
    return result
}

function sha256(content) {  
	return createHash('sha256').update(content).digest('hex')
}

function delay(time) {
  return new Promise(resolve => setTimeout(resolve, time))
}

async function on_model_received ({ stream }) {
  console.log("Metodo on_model_received invocato.")
	const result = await pipe(
		stream,
		async function * (source) {
			for await (const list of source) {
				yield list.subarray()
			}
		},
		toBuffer
    ).finally(() => {
		stream.close()
    })
	
	const model_buff = result.slice(0, result.length-2)
	var random_name_file = create_random_str(10) + '.pt'
	console.log('sha256 del file modello ricevuto: ', sha256(model_buff))
	fs.writeFileSync(path_dir_models + random_name_file, model_buff)
	
	const buffer_age = Buffer.from(result)
	var age_received_model = buffer_age.readUInt16BE(result.length-2)
	console.log('age modello ricevuto: ', age_received_model)
	
	if(index_training < NUM_ROUNDS){
    console.log("Sono al round " + index_training + " su " + NUM_ROUNDS)
		await lock.acquire('key', async() => {
			await python.ex`
			print("loading model received")
  logging.debug("Loading received model.")
	received_model.load_state_dict(torch.load(${path_dir_models} + ${random_name_file}))
			
			print("merging local model with received model")
  logging.debug("Merging local model with received model.")
	merge_models(local_model, int(${age_local_model}), received_model, int(${age_received_model}))
			
			print("training model")	
  logging.debug("Training model.")
	client_update(local_model, opt, train_loader, epoch=epochs)
			
	test_loss, acc = test(local_model, test_loader)
			print('%d-th round, test acc: %0.5f' % (${index_training}, acc))
  logging.debug('%d-th round, test acc: %0.5f' % (${index_training}, acc))
  logging.debug("Training iteration complete.")
			`
			index_training = index_training + 1
			num_send_to_do = num_send_to_do + 1
			age_local_model = age_local_model + 1
      console.log('Index training round: ' + index_training + " Age modello locale: " + age_local_model)
		})
	}
	
	fs.unlinkSync(path_dir_models + random_name_file)
}

var python = pythonBridge({
    python: 'python',
    stdio: ['pipe', process.stdout, process.stderr]
})

await python.ex`
import math
import random
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset   
torch.backends.cudnn.benchmark=True

import os
import logging
logging.basicConfig(level = logging.DEBUG)
	
num_sample_per_client_training = 2500
num_sample_test = 5000

epochs = 5
batch_size = 32


cfg = [32, 'M', 64, 'M', 128, 'M', 256, 256, 'M', 256, 256, 'M']


class MyModel(nn.Module):
  def __init__(self):
    super(MyModel, self).__init__()
    self.features = self._make_layers(cfg)
    self.classifier = nn.Sequential(
      nn.Linear(256, 128),
      nn.ReLU(True),
      nn.Linear(128, 64),
      nn.ReLU(True),
      nn.Linear(64, 10)
    )

  def forward(self, x):
    out = self.features(x)
    out = out.view(out.size(0), -1)
    out = self.classifier(out)
    output = F.log_softmax(out, dim=1)
    return output

  def _make_layers(self, cfg):
    layers = []
    in_channels = 3
    for x in cfg:
      if x == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
      else:
        layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                  nn.BatchNorm2d(x),
                  nn.ReLU(inplace=True)]
        in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


def client_update(client_model, optimizer, train_loader, epoch=5):
  logging.debug("Training avviato.")
  if len(train_loader) == 0:
    logging.error("Il train loader è vuoto. Verifica il dataset.")
  client_model.train()
  for e in range(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
      logging.debug(f"Processing batch {batch_idx}")
      if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
        #logging.debug(f"Data shape: {data.shape}, Target shape: {target.shape}")
        #logging.debug("Sto usando CUDA.")
      else:
        data, target = data.to('cpu'), target.to('cpu')
        #logging.debug(f"Data shape: {data.shape}, Target shape: {target.shape}")
        #logging.debug("CUDA non disponibile. Sto usando la CPU.")
      optimizer.zero_grad()
      output = client_model(data)
      loss = F.nll_loss(output, target)
      if not torch.isfinite(loss):
        logging.error("Loss non finita: controlla i dati e il modello.")
      #logging.debug(f"Loss: {loss.item()}")
      loss.backward()
      #logging.debug(f"Loss: {loss.item()}")
      optimizer.step()
      logging.debug(f"Epoch {e+1}, Loss: {loss.item()}")
  logging.debug("Training completato.")
  return loss.item()


def test(model, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
      else:
        data, target = data.to('cpu'), target.to('cpu')
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  acc = correct / len(test_loader.dataset)

  return test_loss, acc


def merge_models(local_model, age_local_model, received_model, age_received_model):
  local_model_dict = local_model.state_dict()
  received_model_dict = received_model.state_dict()
  for k in local_model_dict.keys():
    local_model_dict[k] = local_model_dict[k].float() * (age_local_model+1)
    received_model_dict[k] = received_model_dict[k].float() * (age_received_model+1)
    local_model_dict[k] = local_model_dict[k].float().add(received_model_dict[k].float())
    local_model_dict[k] = local_model_dict[k].float() / (age_local_model + age_received_model + 2)
  local_model.load_state_dict(local_model_dict)


def create_train_loader():
  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  train_data = datasets.CIFAR10('/app/data', train=True, download=True, transform= transform_train)

  train_data_split = torch.utils.data.random_split(train_data, [num_sample_per_client_training, train_data.data.shape[0] - num_sample_per_client_training])[0]
  train_loader = torch.utils.data.DataLoader(train_data_split, batch_size=batch_size, shuffle=True)
  logging.debug("Train loader creato.")
  logging.debug('Dimensioni train loader: %d' % len(train_loader))
  return train_loader


def create_test_loader():
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  test_data = datasets.CIFAR10('/app/data', train=False, download=True, transform=transform_test)

  test_data_split = torch.utils.data.random_split(test_data, [num_sample_test, test_data.data.shape[0] - num_sample_test])[0]
  test_loader = torch.utils.data.DataLoader(test_data_split, batch_size=batch_size, shuffle=True)
  #print(test_data_split.indices)
  logging.debug("Test loader creato.")
  logging.debug('Dimensioni test loader: %d' % len(test_loader))
  return test_loader
  
  
`

if (!fs.existsSync(path_dir_models)){
	fs.mkdirSync(path_dir_models)
}

const my_ip = get_ip_addr()

const createNode = async () => {
	const node = await createLibp2p({
		addresses: {
			listen: ['/ip4/' + my_ip + '/tcp/4000']
		},
		transports: [tcp()],
		streamMuxers: [mplex()],
		connectionEncryption: [noise()],
		peerDiscovery: [mdns()]
	})

	return node
}

const node = await createNode()
console.log('MY ADDRESS: ', node.getMultiaddrs(), '\n')

node.addEventListener('peer:discovery', async(evt) => {
	
	for(let i=0; i < evt.detail.multiaddrs.length; i++){
    //console.log("Evento discovery, iterazione n. " + i)
    //console.log("Numero multiaddr trovati: " + evt.detail.multiaddrs.length + "\n")
    console.log(evt.detail)
		if(evt.detail.multiaddrs[i].toString().includes('tcp')){
      console.log("Peer trovato.")
			//let peerid = get_peerid_from_multiadd(evt.detail.multiaddrs[i].toString())
      let peerid = evt.detail.id
			if(peer_id_known_peers.includes(peerid) == false){
        console.log("Aggiungo il peer.\n")
				peer_id_known_peers.push(peerid)
				await delay(20000)
				num_of_known_peers = num_of_known_peers + 1
        console.log("Peer totali: " + peer_id_known_peers.length + ", Peer conosciuti: " + num_of_known_peers)
			}
		}
	}
  //console.log("Evento discovery, sono fuori dal ciclo.\n")
})


await python.ex`
train_loader = create_train_loader()
if torch.cuda.is_available():
  logging.debug("CUDA disponibile e in uso")
  local_model = MyModel().cuda()
  received_model = MyModel().cuda()
else:
  logging.debug("CUDA non disponibile; utilizzo la CPU.")
  local_model = MyModel().to('cpu')
  received_model = MyModel().to('cpu')
opt = optim.SGD(local_model.parameters(), lr=0.1)
	
test_loader = create_test_loader()
`
console.log("Train loader e test loader creati.\n")

node.handle('/on_model_received', on_model_received)

const my_model_name = 'my_model_' + create_random_str(6) + '.pt'

while(num_of_known_peers < 1){
	await delay(1000)
}

var content_model_file
var age_local_model_to_send
for(let i=0; i< NUM_ROUNDS; i++){
  console.log("Sono nel ciclo per salvare il modello.\n")
	await lock.acquire('key', async() => {
		await python.ex`
		torch.save(local_model.state_dict(), ${path_dir_models} + ${my_model_name})
  logging.debug("Modello salvato.")
		`
		content_model_file = await fs.readFileSync(path_dir_models + my_model_name)
		num_send_to_do = num_send_to_do - 1
		age_local_model_to_send = age_local_model
	})
	let random_peer = Math.floor(Math.random() * num_of_known_peers)
	
	console.log('sha256 del file modello inviato: ', sha256(content_model_file))
	console.log('invio verso', peer_id_known_peers[random_peer], ' random_peer', random_peer)
	
	//const stream = await node.dialProtocol(peerIdFromString(peer_id_known_peers[random_peer]), '/on_model_received')
  const stream = await node.dialProtocol(peer_id_known_peers[random_peer], '/on_model_received')
	const buff_age = Buffer.alloc(2)
	buff_age.writeUInt16BE(age_local_model_to_send)
	const buff_final = Buffer.concat([content_model_file, buff_age], content_model_file.length+2)
	await pipe([buff_final], stream)

	while(num_send_to_do <= 0){
		await delay(2000)
	}
}