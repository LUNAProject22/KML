import torch
import clip
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer

def get_inverse_module_name(name):
    return name + '_INV'

def get_inverse_relation_names_for_PKVQA():
    inv_rel_map = {}
    inv_rel_map['HAS_GROUNDED_TOOL'] = 'USED_IN_STEP'
    inv_rel_map['HAS_PURPOSE'] = 'IS_SERVED_BY'
    inv_rel_map['HAS_NEXT_STEP'] = 'HAS_PREVIOUS_STEP'
    inv_rel_map['HAS_TASK'] = 'IN_DOMAIN'
    inv_rel_map['HAS_STEP'] = 'IN_TASK'
    inv_rel_map['HAS_ACTION'] = 'ACTION_IN_STEP'
    inv_rel_map['HAS_OBJECT'] = 'OBJECT_IN_STEP'
    inv_rel_map['HAS_SIMILAR_PURPOSE'] = 'HAS_SIMILAR_PURPOSE'
    return inv_rel_map

def get_inverse_relation_names_for_PKVQA_Module(module_name):
    inv_rel_map = {}
    inv_rel_map['HAS_GROUNDED_TOOL'] = 'USED_IN_STEP'
    inv_rel_map['HAS_PURPOSE'] = 'IS_SERVED_BY'
    inv_rel_map['HAS_NEXT_STEP'] = 'HAS_PREVIOUS_STEP'
    inv_rel_map['HAS_TASK'] = 'IN_DOMAIN'
    inv_rel_map['HAS_STEP'] = 'IN_TASK'
    inv_rel_map['HAS_ACTION'] = 'ACTION_IN_STEP'
    inv_rel_map['HAS_OBJECT'] = 'OBJECT_IN_STEP'
    inv_rel_map['HAS_SIMILAR_PURPOSE'] = 'HAS_SIMILAR_PURPOSE'

    inv_rel_map['USED_IN_STEP'] = 'HAS_GROUNDED_TOOL'
    inv_rel_map['IS_SERVED_BY'] = 'HAS_PURPOSE'
    inv_rel_map['HAS_PREVIOUS_STEP'] = 'HAS_NEXT_STEP'
    inv_rel_map['IN_DOMAIN'] = 'HAS_TASK'
    inv_rel_map['IN_TASK'] = 'HAS_STEP'
    inv_rel_map['ACTION_IN_STEP'] = 'HAS_ACTION'
    inv_rel_map['OBJECT_IN_STEP'] = 'HAS_OBJECT'
    


    return inv_rel_map[module_name]




class TextEncoder(ABC):

    @abstractmethod
    def encode_text(self, text):
        pass

    @abstractmethod
    def get_name(self):
        pass

   



class ClipTextEncoder(TextEncoder):

    def __init__(self):
        super().__init__()
        model, _ = clip.load("ViT-B/32", device='cuda')
        self.model = model
        print('ClipTextEncoder - VitB/32 text encoder is used.')
    
    def get_name(self):
        return 'ViT-B/32'

    def encode_text(self, text):
        text = clip.tokenize(text).to('cuda')    
        with torch.no_grad():
            text_features = self.model.encode_text(text).cpu().detach()      
        text_features = text_features.float()
        return text_features


class BertTextEncoder(TextEncoder):

    def __init__(self):
        super().__init__()
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.model = model
        print('bert-base-nli-mean-tokens is used.')
    
    def get_name(self):
        return 'bert-base-nli-mean-tokens'

    def encode_text(self, text):        
        with torch.no_grad():
            text_features = self.model.encode(text).squeeze()
            text_features = torch.from_numpy(text_features)
        text_features = text_features.float()
        text_features = text_features / text_features.norm()
        return text_features


class RobertaTextEncoder(TextEncoder):

    def __init__(self):
        super().__init__()
        model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
        self.model = model
        print('roberta-base-nli-stsb-mean-tokens.')
    
    def get_name(self):
        return 'roberta-base-nli-stsb-mean-tokens'

    def encode_text(self, text):        
        with torch.no_grad():
            text_features = self.model.encode(text).squeeze()
            text_features = torch.from_numpy(text_features)
        text_features = text_features.float()
        text_features = text_features / text_features.norm()
        return text_features
    
class MiniLMTextEncoder(TextEncoder):

    def __init__(self):
        super().__init__()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = model
        print('all-MiniLM-L6-v2.')
    
    def get_name(self):
        return 'all-MiniLM-L6-v2'

    def encode_text(self, text):        
        with torch.no_grad():
            text_features = self.model.encode(text).squeeze()
            text_features = torch.from_numpy(text_features)
        text_features = text_features.float()
        text_features = text_features / text_features.norm()
        return text_features



    
