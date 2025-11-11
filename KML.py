# -----------------------------------------------------------------------------
#  Knowledge Module Learning (KML)
#  Copyright 2025 Agency for Science, Technology and Research (A*STAR)
#  Authors: Basura Fernando and the CFAR/IHPC KML Team
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from KML_commons import *
from abc import ABC, abstractmethod
from KML_commons import TextEncoder

def _normalize(o):
    return o / o.norm(dim=1, keepdim=True)

def getConsLoss(module, o_hat, o, criterion, ground_truth):
    o_hat = _normalize(o_hat)    
    o = _normalize(o)
    logits_per_text = module.logit_scale * o_hat @ o.t()   
    loss_contrastive = criterion(logits_per_text, ground_truth) 
    return loss_contrastive

class KGData(Dataset):
    
    @abstractmethod    
    def get_relation_list(self):
        pass


class KGDataSinglePositive(KGData):
    def __init__(self, relations_dict, small=False):
        data = []
        self.count = 0 
        self.entity_to_id = {}
        self.id_to_entity = {}
        entity_id = 0
        self.relation_names_list = list(relations_dict.keys())
        for relation_name in relations_dict:
            count = 0
            for head_and_tail in tqdm(relations_dict[relation_name]): 
                count = count + 1    
                if small and count > 100:
                    continue
                for entity_name  in head_and_tail:
                    if entity_name not in self.entity_to_id:
                        self.entity_to_id[entity_name] = entity_id
                        self.id_to_entity[entity_id] = entity_name
                        entity_id = entity_id + 1                        
                numeric_head_tail_pair = torch.Tensor([self.entity_to_id[head_and_tail[0]], self.entity_to_id[head_and_tail[1]]])

                data.append( (relation_name, numeric_head_tail_pair) )                    
        self.data = data
        print('Number of entities {}'.format( len(list(self.entity_to_id.keys()))))
        print('Number of relations {}'.format( len(self.relation_names_list)))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        relation_name, numeric_head_tail_pair = self.data[idx]
        head_entity_id = numeric_head_tail_pair[0]
        tail_entity_id = numeric_head_tail_pair[1]
        return relation_name, head_entity_id, tail_entity_id
    
    def get_relation_list(self):
        return self.relation_names_list



class KML(nn.Module):
    def __init__(self, mudule_name_list : list[str], 
                 kgdataset: KGData,
                 hidden_dim : int = 256,
                 embd_init_func: TextEncoder = None,
                 embedding_dim :int = 256,
                 act : nn.Module = nn.Tanh(),
                 temparature: int = 256
                 ):   
        

        super().__init__()
        self.module_diciotnary = nn.ModuleDict()             
                
        
        entity_size = len(list(kgdataset.entity_to_id.keys()))
        if embd_init_func is not None:
            print(f'Entity embeddings are initialized with function {embd_init_func.get_name()}')                                  
            W = torch.zeros(entity_size,embedding_dim)
            print(W.shape)
            for entity_name in tqdm(kgdataset.entity_to_id):                
                entity_id = kgdataset.entity_to_id[entity_name]                
                entity_embedding_init = embd_init_func.encode_text(entity_name)
                W[entity_id] = entity_embedding_init                        
            self.embeddings = nn.Embedding(entity_size , embedding_dim, _weight= W) # max_norm=1.0
        else:        
            print('Entity embeddings are randomly initialized.')            
            self.embeddings = nn.Embedding(entity_size , embedding_dim)

        for name in mudule_name_list:            
                self.module_diciotnary[name] = torch.nn.Sequential(
                    torch.nn.Linear(embedding_dim, hidden_dim),
                    act, 
                    torch.nn.Linear(hidden_dim, embedding_dim),)                                                          
                
            
            

        print(list(self.module_diciotnary.keys()))
        self.logit_scale = nn.Parameter(torch.ones([]) * 1.0 * temparature )
        print('self.logit_scale : ', self.logit_scale.item())

    
    
    def set_mode(self, mode):
        self.mode = mode

    
    def inference(self, entity_embedding, relations_list: list[str]):        
        if self.training:
            raise("Model is called to do inference. But still in training mode.")
        x = entity_embedding
        for relation_name in relations_list:                                                          
                x = self.module_diciotnary[relation_name](x)            
                x = x / x.norm(dim=1, keepdim=True)                            
        return x
    
    def train_with_q(self, entity_embedding, relations_list: list[str]):                
        x = entity_embedding
        for relation_name in relations_list:                                                          
                x = self.module_diciotnary[relation_name](x)            
                x = x / x.norm(dim=1, keepdim=True)                            
        return x

    
    def inference_with_intermediate_results(self, entity_embedding, relations_list: list[str]):        
        if self.training:
            raise("Model is called to do inference. But still in training mode.")
        x = entity_embedding
        outs = []
        for relation_name in relations_list:                                                          
                x = self.module_diciotnary[relation_name](x)            
                x = x / x.norm(dim=1, keepdim=True)    
                outs.append(x)                        
        return x, outs


    def forward(self, entity_id_list, relations_list: list[str]):                
        x = self.embeddings(entity_id_list.long()) 
        out = []                               
        for i, relation_name in enumerate(relations_list):                                                           
            x_out = self.module_diciotnary[relation_name](x[i]).squeeze().unsqueeze(dim=0)                    
            out.append(x_out)                
        x = torch.cat(out,dim=0)        
        x = x / x.norm(dim=1, keepdim=True)        
        return x

    def forward_parallel(self, entity_id_list, relations_list: list[str]):                
        x = self.embeddings(entity_id_list.long()) 
        outstates = {}                               
        for i, relation_name in enumerate(relations_list):                                                           
            outstates[i] = torch.jit.fork(self.module_diciotnary[relation_name], x[i])

        out = []
        for i, relation_name in enumerate(relations_list): 
            x_out = torch.jit.wait(outstates[i])
            out.append(x_out.squeeze().unsqueeze(dim=0))

        x = torch.cat(out,dim=0)        
        x = x / x.norm(dim=1, keepdim=True)        
        return x
   
def KML_Train(kgdataset: KGData, module: KML, batch_size, learning_rate, weight_decay, device, epochs, 
            inv_rel_func, post_Process_func = None, use_of_scheduler = False, use_parallel = False):
    print('KG size :', kgdataset.__len__())
    train_dataloader = DataLoader(kgdataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()    
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay = weight_decay)    
    if use_of_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3)
    module.to(device)
    optimizer.zero_grad()      
    for ep in range(epochs):        
        constrastive_total_loss = 0.        
        loss = None
        for idx, data in tqdm(enumerate(train_dataloader)):
            r, f1, f2 = data            
            f1 = f1.to(device)
            f2 = f2.to(device)        
            ground_truth = torch.arange(len(r),dtype=torch.long).to(device)         
            if use_parallel:
                f2_hat = module.forward_parallel(f1, list(r))                                        
            else:                
                f2_hat = module(f1, list(r))                                        

            r_inv = [inv_rel_func(ri) for ri in list(r)]
            if use_parallel:
                f1_hat = module.forward_parallel(f2, r_inv)  
            else:                
                f1_hat = module(f2, r_inv)  

            f1_gt = module.embeddings(f1.long())
            f2_gt = module.embeddings(f2.long())         

            # Contrastive            
            loss1 = getConsLoss(module, f2_hat, f2_gt, criterion, ground_truth)   
            loss2 = getConsLoss(module, f1_hat, f1_gt, criterion, ground_truth)   
            loss = loss1 + loss2
            
            constrastive_total_loss = constrastive_total_loss + loss.item()          
            optimizer.zero_grad()    
            loss.backward()    
            optimizer.step()            
        mean_contrastive_loss = constrastive_total_loss / idx
        print('{} loss {:.5f}'.format(ep,  mean_contrastive_loss  ))
        if use_of_scheduler:
            scheduler.step(mean_contrastive_loss)
            for param_group in optimizer.param_groups:
                print("train kg lr: ", param_group['lr'])       
        if post_Process_func is not None:
            post_Process_func(module, constrastive_total_loss, optimizer, ep)
    return module



