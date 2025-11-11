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
import json
from copy import deepcopy
from torch.utils.data import Dataset
from KML import KML
from tqdm import tqdm
from KML import KGData
from KML_commons import *
import pickle
import os
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy



def get_KB_data():
    with open('dataset/cointrain_kgv2.json') as f:
        lines = f.readlines()
        schema = {}
        schema_rel = {}
        step_to_step = {}
        JD = None
        step_list = {}
        step_to_task = {}
        node_types = {}
        relational_data_mapping = {}
        for data_line in lines: # each line is a JSON
            jd  = json.loads(data_line)
            node_types[jd['type']] = set(jd.keys())
            if jd['type'] == 'node':
                if jd['labels'][0] == 'Step':
                    if jd['properties']['name'] not in step_list.keys():
                        step_list[jd['properties']['name']] = jd['properties']['stepid']
                        step_to_task[jd['properties']['stepid']] = jd['properties']['task']     
                if jd['labels'][0] not in schema.keys():            
                    schema[jd['labels'][0]] = list(jd['properties'].keys())
            if jd['type'] == 'relationship':
                entry = [jd['start']['properties']['name'], jd['end']['properties']['name']]
                if jd['label'] in relational_data_mapping.keys():
                    relational_data_mapping[jd['label']].append(entry)
                else:
                    relational_data_mapping[jd['label']] = [entry]
                if jd['label'] not in schema_rel.keys():
                    start_node = jd['start']['labels'][0]
                    end_node = jd['end']['labels'][0]
                    schema_rel[jd['label']] = '{}->{}'.format(start_node,end_node)
                
                if jd['label'] == 'HAS_NEXT_STEP':
                    if 'task' in jd['start']['properties'].keys():
                        task = jd['start']['properties']['task']
                    else:
                        task = jd['start']['properties']['name']
                    start_step = jd['start']['properties']['name']
                    end_step = jd['end']['properties']['name']
                    
        
        
        temp_map = deepcopy(relational_data_mapping)
        inv_rel_map = get_inverse_relation_names_for_PKVQA()
        for key in temp_map:
            for head,tail in temp_map[key]:
                if inv_rel_map[key] not in relational_data_mapping:
                   # relational_data_mapping[inv_rel_map[key]] = relational_data_mapping[inv_rel_map[key]] + [tail, head]                
                    relational_data_mapping[inv_rel_map[key]] = []
        


        return relational_data_mapping





class PKRQADataset(Dataset):    
    def __init__(self, small = False):
        
        if small == True:            
            self.validation_data = json.load( open('dataset/s4_QADataset_12Feb2025/val/validation_small_50.json'))
            print('Loading small val dataset from : s4_QADataset_12Feb2025/val/validation_small_50.json')          
            validation_step_pred = json.load( open('dataset/QA_25Oct24_validation_pred.json'))
        else:                        
            self.validation_data = json.load( open('dataset/s4_QADataset_12Feb2025/testing.json'))    
            print('Loading Large dataset : s4_QADataset_12Feb2025/testing.json')                               
            validation_step_pred = json.load( open('dataset/QA_25Oct24_testing_pred.json'))
        
        pred_dict = {}
        for item  in validation_step_pred:
            pred_dict[list(item.keys())[0]] = item[list(item.keys())[0]]
        self.pred_dict = pred_dict
        

    def __len__(self):
        return len(self.validation_data)
    
    def get_video(self, qid):
        E = [e_ for e_ in self.validation_data if e_['qid'] == qid]
        if len(E) > 0:
            return E[0]['video_id'] , E[0]['step']['segment']
        else:
            return None

    def __getitem__(self, idx):        
        options    = self.validation_data[idx]['options']        
        answer_idx = self.validation_data[idx]['answer']
        qType   = self.validation_data[idx]['quest_type']
        step       = self.validation_data[idx]['step']['label']  
        qid        = self.validation_data[idx]['qid']
        taskName   = self.validation_data[idx]['task_label']
        pred        = self.pred_dict[qid]        
        question   = self.validation_data[idx]['question']
        task_id    = self.validation_data[idx]['task_id']        
        return options, answer_idx , step, qType, qid, taskName , pred, question, task_id
    

class QADatasetTrain(Dataset):    
    def __init__(self):       
        
        
        self.validation_data = json.load( open('dataset/s4_QADataset_12Feb2025/train/training_small_100.json'))        
        print('MyQDatasetTrain--> s4_QADataset_12Feb2025/train/training_small_100.json')
        


    def __len__(self):
        return len(self.validation_data)

    def __getitem__(self, idx):        
        options    = self.validation_data[idx]['options']        
        answer_idx = self.validation_data[idx]['answer']        
        qType      = self.validation_data[idx]['quest_type']
            
        step       = self.validation_data[idx]['step']['label']
        
        qid        = self.validation_data[idx]['qid']
        taskName   = self.validation_data[idx]['task_label']
        question   = self.validation_data[idx]['question']
        pred       = None
        task_id    = self.validation_data[idx]['task_id']
        return options, answer_idx , step, qType, qid, taskName , pred, task_id



class KML_Programs():

    def get_predefined_program_for_qtype(self, gpt=20):
        """
        HAS_GROUNDED_TOOL : Step->GroundedTool
        HAS_PURPOSE : GroundedTool->Purpose
        HAS_NEXT_STEP : START->Step
        HAS_TASK : Domain->Task
        HAS_STEP : Task->START
        HAS_ACTION : Step->Action
        HAS_OBJECT : Step->Object
        HAS_SIMILAR_PURPOSE : Purpose->Purpose
        """

        PRGRAM = {}

        
        if gpt == 20:  # Programs generated by GPT4 using API on 10 / 10 / 2024
            PRGRAM['qa1_step2tool']         = {'input':'step', 'func': [['HAS_GROUNDED_TOOL']]}
            PRGRAM['qa2_bestNextStep']      = {'input':'step', 'func': [['HAS_NEXT_STEP']]}
            PRGRAM['qa3_nextStep']          = {'input':'step', 'func': [['HAS_NEXT_STEP']]}
            PRGRAM['qa6_precedingStep']     = {'input':'step', 'func': [['HAS_NEXT_STEP_INV']]}
            PRGRAM['qa7_bestPrecedingStep'] = {'input':'step', 'func': [['HAS_NEXT_STEP_INV']]}
            PRGRAM['qa8_toolNextStep']      = {'input':'step', 'func': [['HAS_NEXT_STEP', 'HAS_GROUNDED_TOOL']]}
            PRGRAM['qa9_bestInitial']       = {'input':'task', 'func': [['HAS_STEP']]}
            PRGRAM['qa10_bestFinal']        = {'input':'task', 'func': [['HAS_STEP']]}
            PRGRAM['qa11_domain']           = {'input':'task', 'func': [['HAS_TASK_INV']]}
            PRGRAM['qa12_toolPurpose']      = {'input':'step', 'func': [['HAS_GROUNDED_TOOL', 'HAS_PURPOSE']]}
            PRGRAM['qa13_actionPurpose']    = {'input':'step', 'func': [['HAS_ACTION', 'HAS_PURPOSE']]}
            PRGRAM['qa14_objectPurpose']    = {'input':'step', 'func': [['HAS_OBJECT', 'HAS_PURPOSE']]}        
            PRGRAM['qa15_ToolOtherPurpose'] = {'input':'step', 'func': [['HAS_GROUNDED_TOOL','HAS_PURPOSE']]}
            PRGRAM['qa16_ObjectOtherPurpose']        = {'input':'step', 'func': [['HAS_OBJECT', 'HAS_PURPOSE']]}
            PRGRAM['qa17_AlternativeTool']           = {'input':'step', 'func': [['HAS_GROUNDED_TOOL', 'HAS_PURPOSE','HAS_SIMILAR_PURPOSE','HAS_PURPOSE_INV']]}
            PRGRAM['qa18_TaskSameToolSamePurpose']   = {'input':'step', 'func': [['HAS_GROUNDED_TOOL', 'HAS_GROUNDED_TOOL_INV','HAS_STEP_INV']]}        
            PRGRAM['qa19_TaskSameObjectSamePurpose'] = {'input':'step', 'func': [['HAS_OBJECT','HAS_OBJECT_INV','HAS_STEP_INV']]}
        
        if gpt == 21:  # Programs generated by GPT4 using API on 31 / 01 / 2025
            PRGRAM['qa1_step2tool']         = {'input':'step', 'func': [['HAS_GROUNDED_TOOL']]}
            PRGRAM['qa2_bestNextStep']      = {'input':'step', 'func': [['HAS_NEXT_STEP'], ['HAS_NEXT_STEP','HAS_NEXT_STEP']]}
            PRGRAM['qa3_nextStep']          = {'input':'step', 'func': [['HAS_NEXT_STEP']]}
            PRGRAM['qa6_precedingStep']     = {'input':'step', 'func': [['HAS_NEXT_STEP_INV'],['HAS_NEXT_STEP_INV','HAS_NEXT_STEP_INV']]}
            PRGRAM['qa7_bestPrecedingStep'] = {'input':'step', 'func': [['HAS_NEXT_STEP_INV']]}
            PRGRAM['qa8_toolNextStep']      = {'input':'step', 'func': [['HAS_NEXT_STEP', 'HAS_GROUNDED_TOOL']]}
            PRGRAM['qa9_bestInitial']       = {'input':'task', 'func': [['HAS_STEP']]}
            PRGRAM['qa10_bestFinal']        = {'input':'task', 'func': [['HAS_STEP']]}
            PRGRAM['qa11_domain']           = {'input':'task', 'func': [['HAS_TASK_INV']]}
            PRGRAM['qa12_toolPurpose']      = {'input':'step', 'func': [['HAS_GROUNDED_TOOL', 'HAS_PURPOSE']]}
            PRGRAM['qa13_actionPurpose']    = {'input':'step', 'func': [['HAS_ACTION', 'HAS_PURPOSE']]}
            PRGRAM['qa14_objectPurpose']    = {'input':'step', 'func': [['HAS_OBJECT', 'HAS_PURPOSE']]}        
            PRGRAM['qa15_ToolOtherPurpose'] = {'input':'step', 'func': [['HAS_GROUNDED_TOOL','HAS_PURPOSE']]}
            PRGRAM['qa16_ObjectOtherPurpose']        = {'input':'step', 'func': [['HAS_OBJECT', 'HAS_PURPOSE']]}
            PRGRAM['qa17_AlternativeTool']           = {'input':'step', 'func': [['HAS_GROUNDED_TOOL', 'HAS_PURPOSE','HAS_SIMILAR_PURPOSE','HAS_PURPOSE_INV']]}
            PRGRAM['qa18_TaskSameToolSamePurpose']   = {'input':'step', 'func': [['HAS_GROUNDED_TOOL', 'HAS_GROUNDED_TOOL_INV','HAS_STEP_INV']]}        
            PRGRAM['qa19_TaskSameObjectSamePurpose'] = {'input':'step', 'func': [['HAS_OBJECT','HAS_OBJECT_INV','HAS_STEP_INV']]}


        if gpt == 3:  # LAMMA
            PRGRAM['qa1_step2tool']         = {'input':'step', 'func': ['HAS_GROUNDED_TOOL']}
            PRGRAM['qa2_bestNextStep']      = {'input':'step', 'func': ['HAS_NEXT_STEP']}
            PRGRAM['qa3_nextStep']          = {'input':'step', 'func': ['HAS_NEXT_STEP']}
            PRGRAM['qa6_precedingStep']     = {'input':'step', 'func': ['HAS_NEXT_STEP_INV']}
            PRGRAM['qa7_bestPrecedingStep'] = {'input':'step', 'func': ['HAS_NEXT_STEP_INV']}
            PRGRAM['qa8_toolNextStep']      = {'input':'step', 'func': ['HAS_NEXT_STEP', 'HAS_GROUNDED_TOOL']}
            PRGRAM['qa9_bestInitial']       = {'input':'task', 'func': ['HAS_STEP','HAS_NEXT_STEP','HAS_NEXT_STEP_INV']}
            PRGRAM['qa10_bestFinal']        = {'input':'task', 'func': ['HAS_STEP','HAS_NEXT_STEP','HAS_NEXT_STEP_INV']}
            PRGRAM['qa11_domain']           = {'input':'task', 'func': ['HAS_TASK_INV']}
            PRGRAM['qa12_toolPurpose']      = {'input':'step', 'func': ['HAS_GROUNDED_TOOL', 'HAS_PURPOSE']}
            PRGRAM['qa13_actionPurpose']    = {'input':'step', 'func': ['HAS_GROUNDED_TOOL', 'HAS_PURPOSE']}
            PRGRAM['qa14_objectPurpose']    = {'input':'step', 'func': ['HAS_OBJECT', 'HAS_PURPOSE']}
            PRGRAM['qa15_ToolOtherPurpose'] = {'input':'step', 'func': ['HAS_GROUNDED_TOOL', 'HAS_PURPOSE']}
            PRGRAM['qa16_ObjectOtherPurpose']        = {'input':'step', 'func': ['HAS_OBJECT', 'HAS_PURPOSE']}
            PRGRAM['qa17_AlternativeTool']           = {'input':'step', 'func': ['HAS_GROUNDED_TOOL', 'HAS_PURPOSE','HAS_PURPOSE_INV']}
            PRGRAM['qa18_TaskSameToolSamePurpose']   = {'input':'step', 'func': ['HAS_GROUNDED_TOOL', 'HAS_GROUNDED_TOOL_INV','HAS_STEP_INV']}
            PRGRAM['qa19_TaskSameObjectSamePurpose'] = {'input':'step', 'func': ['HAS_OBJECT', 'HAS_OBJECT_INV','HAS_STEP_INV','HAS_PURPOSE_INV','HAS_STEP_INV']}    

        if gpt == 4:  # Mistral
            PRGRAM['qa1_step2tool']         = {'input':'step', 'func': ['HAS_GROUNDED_TOOL']}
            PRGRAM['qa2_bestNextStep']      = {'input':'step', 'func': ['HAS_NEXT_STEP']}
            PRGRAM['qa3_nextStep']          = {'input':'step', 'func': ['HAS_NEXT_STEP']}
            PRGRAM['qa6_precedingStep']     = {'input':'step', 'func': ['HAS_NEXT_STEP_INV']}
            PRGRAM['qa7_bestPrecedingStep'] = {'input':'step', 'func': ['HAS_NEXT_STEP_INV']}
            PRGRAM['qa8_toolNextStep']      = {'input':'step', 'func': ['HAS_NEXT_STEP', 'HAS_GROUNDED_TOOL']}
            PRGRAM['qa9_bestInitial']       = {'input':'task', 'func': ['HAS_STEP','HAS_NEXT_STEP_INV']}
            PRGRAM['qa10_bestFinal']        = {'input':'task', 'func': ['HAS_STEP','HAS_NEXT_STEP']}
            PRGRAM['qa11_domain']           = {'input':'task', 'func': ['HAS_TASK_INV']}
            PRGRAM['qa12_toolPurpose']      = {'input':'step', 'func': ['HAS_GROUNDED_TOOL', 'HAS_PURPOSE']}
            PRGRAM['qa13_actionPurpose']    = {'input':'step', 'func': ['HAS_ACTION', 'HAS_ACTION_INV','HAS_OBJECT','HAS_OBJECT_INV','HAS_GROUNDED_TOOL','HAS_PURPOSE']}
            PRGRAM['qa14_objectPurpose']    = {'input':'step', 'func': ['HAS_OBJECT', 'HAS_OBJECT_INV','HAS_GROUNDED_TOOL','HAS_PURPOSE']}
            PRGRAM['qa15_ToolOtherPurpose'] = {'input':'step', 'func': ['HAS_GROUNDED_TOOL', 'HAS_PURPOSE','HAS_SIMILAR_PURPOSE']}
            PRGRAM['qa16_ObjectOtherPurpose']        = {'input':'step', 'func': ['HAS_OBJECT', 'HAS_OBJECT_INV', 'HAS_GROUNDED_TOOL', 'HAS_PURPOSE']}
            PRGRAM['qa17_AlternativeTool']           = {'input':'step', 'func': ['HAS_GROUNDED_TOOL', 'HAS_PURPOSE','HAS_PURPOSE_INV']}
            PRGRAM['qa18_TaskSameToolSamePurpose']   = {'input':'step', 'func': ['HAS_GROUNDED_TOOL', 'HAS_GROUNDED_TOOL_INV','HAS_STEP_INV']}
            PRGRAM['qa19_TaskSameObjectSamePurpose'] = {'input':'step', 'func': ['HAS_OBJECT', 'HAS_OBJECT_INV','HAS_STEP_INV']}    

        if gpt == 5:  # deepseek
            PRGRAM['qa1_step2tool']         = {'input':'step', 'func': [['HAS_GROUNDED_TOOL']]}
            PRGRAM['qa2_bestNextStep']      = {'input':'step', 'func': [['HAS_NEXT_STEP']]}
            PRGRAM['qa3_nextStep']          = {'input':'step', 'func': [['HAS_NEXT_STEP']]}
            PRGRAM['qa6_precedingStep']     = {'input':'step', 'func': [['HAS_NEXT_STEP_INV']]}
            PRGRAM['qa7_bestPrecedingStep'] = {'input':'step', 'func': [['HAS_NEXT_STEP_INV']]}
            PRGRAM['qa8_toolNextStep']      = {'input':'step', 'func': [['HAS_NEXT_STEP', 'HAS_GROUNDED_TOOL']]}
            PRGRAM['qa9_bestInitial']       = {'input':'task', 'func': [['HAS_STEP']]}
            PRGRAM['qa10_bestFinal']        = {'input':'task', 'func': [['HAS_STEP','HAS_NEXT_STEP']]}
            PRGRAM['qa11_domain']           = {'input':'task', 'func': [['HAS_TASK_INV']]}
            PRGRAM['qa12_toolPurpose']      = {'input':'step', 'func': [['HAS_GROUNDED_TOOL', 'HAS_PURPOSE']]}
            PRGRAM['qa13_actionPurpose']    = {'input':'step', 'func': [['HAS_ACTION', 'HAS_ACTION_INV','HAS_OBJECT','HAS_OBJECT_INV','HAS_GROUNDED_TOOL','HAS_PURPOSE']]}
            PRGRAM['qa14_objectPurpose']    = {'input':'step', 'func': [['HAS_OBJECT', 'HAS_OBJECT_INV','HAS_GROUNDED_TOOL','HAS_PURPOSE']]}
            PRGRAM['qa15_ToolOtherPurpose'] = {'input':'step', 'func': [['HAS_GROUNDED_TOOL', 'HAS_PURPOSE','HAS_SIMILAR_PURPOSE']]}
            PRGRAM['qa16_ObjectOtherPurpose']        = {'input':'step', 'func': [['HAS_OBJECT', 'HAS_OBJECT_INV', 'HAS_GROUNDED_TOOL', 'HAS_PURPOSE']]}
            PRGRAM['qa17_AlternativeTool']           = {'input':'step', 'func': [['HAS_GROUNDED_TOOL', 'HAS_PURPOSE','HAS_PURPOSE_INV']]}
            PRGRAM['qa18_TaskSameToolSamePurpose']   = {'input':'step', 'func': [
                            ['HAS_GROUNDED_TOOL', 'HAS_GROUNDED_TOOL_INV','HAS_STEP_INV'],                            
                                                                                ]}
            PRGRAM['qa19_TaskSameObjectSamePurpose'] = {'input':'step', 'func': [['HAS_OBJECT', 'HAS_OBJECT_INV','HAS_STEP_INV'],                                                                                
                                                                                ['HAS_STEP_INV']
                                                                                ]}    
            
        if gpt == 6:  # Qwen
            PRGRAM['qa1_step2tool']         = {'input':'step', 'func': [['HAS_GROUNDED_TOOL']]}
            PRGRAM['qa2_bestNextStep']      = {'input':'step', 'func': [['HAS_NEXT_STEP']]}
            PRGRAM['qa3_nextStep']          = {'input':'step', 'func': [['HAS_NEXT_STEP']]}
            PRGRAM['qa6_precedingStep']     = {'input':'step', 'func': [['HAS_NEXT_STEP_INV']]}
            PRGRAM['qa7_bestPrecedingStep'] = {'input':'step', 'func': [['HAS_NEXT_STEP_INV']]}
            PRGRAM['qa8_toolNextStep']      = {'input':'step', 'func': [['HAS_NEXT_STEP', 'HAS_GROUNDED_TOOL']]}
            PRGRAM['qa9_bestInitial']       = {'input':'task', 'func': [['HAS_STEP']]}
            PRGRAM['qa10_bestFinal']        = {'input':'task', 'func': [['HAS_STEP','HAS_NEXT_STEP']]}
            PRGRAM['qa11_domain']           = {'input':'task', 'func': [['HAS_TASK_INV']]}
            PRGRAM['qa12_toolPurpose']      = {'input':'step', 'func': [['HAS_GROUNDED_TOOL', 'HAS_PURPOSE']]}
            PRGRAM['qa13_actionPurpose']    = {'input':'step', 'func': [['HAS_ACTION', 'HAS_ACTION_INV','HAS_OBJECT','HAS_OBJECT_INV','HAS_GROUNDED_TOOL','HAS_PURPOSE']]}
            PRGRAM['qa14_objectPurpose']    = {'input':'step', 'func': [['HAS_OBJECT', 'HAS_OBJECT_INV','HAS_GROUNDED_TOOL','HAS_PURPOSE']]}
            PRGRAM['qa15_ToolOtherPurpose'] = {'input':'step', 'func': [['HAS_GROUNDED_TOOL', 'HAS_PURPOSE','HAS_SIMILAR_PURPOSE']]}
            PRGRAM['qa16_ObjectOtherPurpose']        = {'input':'step', 'func': [['HAS_OBJECT', 'HAS_OBJECT_INV', 'HAS_GROUNDED_TOOL', 'HAS_PURPOSE','HAS_SIMILAR_PURPOSE']]}
            PRGRAM['qa17_AlternativeTool']           = {'input':'step', 'func': [['HAS_GROUNDED_TOOL', 'HAS_PURPOSE', 'HAS_SIMILAR_PURPOSE' ,'HAS_PURPOSE_INV']]}
            PRGRAM['qa18_TaskSameToolSamePurpose']   = {'input':'step', 'func': [['HAS_GROUNDED_TOOL', 'HAS_PURPOSE' , 'HAS_SIMILAR_PURPOSE', 'HAS_PURPOSE_INV', 'HAS_GROUNDED_TOOL_INV','HAS_STEP_INV'],]}
            PRGRAM['qa19_TaskSameObjectSamePurpose'] = {'input':'step', 'func': [['HAS_OBJECT','HAS_OBJECT_INV','HAS_GROUNDED_TOOL','HAS_PURPOSE_INV','HAS_SIMILAR_PURPOSE','HAS_PURPOSE_INV','HAS_GROUNDED_TOOL_INV','HAS_STEP_INV']]}    

        
        



        inv_map  = get_inverse_relation_names_for_PKVQA()
        NEWPRGRAM = {}
        for qtype in PRGRAM:
            new_updated_programs = []
            programs = PRGRAM[qtype]['func']
            for program in programs:
                updated_program = []
                for func_name in program:
                    if func_name.endswith('_INV'):
                        new_func_name = inv_map[func_name.replace('_INV','')]      
                        updated_program.append(new_func_name)
                    else:
                        updated_program.append(func_name)
                new_updated_programs.append(updated_program)     
            NEWPRGRAM[qtype] = {}    
            NEWPRGRAM[qtype]['func'] = new_updated_programs
            NEWPRGRAM[qtype]['input'] = PRGRAM[qtype]['input']
        return NEWPRGRAM


    #print(get_PROGRAMS(20))


class QA():

    def __init__(self, kg_dataset, is_log = True, is_small = False):
        self.programGenerator = KML_Programs()
        self.myQDataset = PKRQADataset(small= is_small)
        self.kg_dataset = kg_dataset
        self.is_log = is_log


    def answer_question(self, module: KML, program_type = 20, topk_step_task = 5, device='cuda', exp_settings = None):
    
        results_correct_dict = {}
        results_total_dict = {}
        PRGRAM = self.programGenerator.get_predefined_program_for_qtype(program_type)            
        module.eval()        
        with torch.no_grad():
            for i in tqdm(range(self.myQDataset.__len__())):
                options, answer_idx , step_gt, qType, qid, taskName_gt , pred, question, task_id = self.myQDataset.__getitem__(i)                   

                if qType in PRGRAM:
                    inputs = PRGRAM[qType]['input']
                    if qType in results_total_dict.keys():
                        results_total_dict[qType] = results_total_dict[qType] + 1.
                    else:
                        results_total_dict[qType] = 1.   
                    
                    if inputs == 'step':                    
                        input_scores = pred['step_top5_scores'][0:topk_step_task]
                        input_classes = pred['step_top5_classes'][0:topk_step_task]                    
                        inputs_x = module.embeddings(torch.Tensor([self.kg_dataset.entity_to_id[cl] for cl in input_classes]).long().to(device))
                        U = torch.Tensor(input_scores).unsqueeze(dim=0).to(device)                     
                        U = U / U.sum()                    
                        inputs_x = torch.matmul(U, inputs_x)           

                    if inputs == 'task':                    
                        inputs_x = module.embeddings(torch.Tensor([self.kg_dataset.entity_to_id[cl] for cl in pred['task_top5_classes']]).long().to(device))                     
                        U = torch.Tensor(pred['task_top5_scores'][0:topk_step_task]).unsqueeze(dim=0).to(device) 
                        U = U / U.sum()
                        inputs_x = torch.matmul(U, inputs_x)                    


                    list_of_outs = []                  
                    for program_list in PRGRAM[qType]['func']:                      
                        out_x = module.inference(inputs_x, program_list) 
                        out_x = out_x / out_x.norm(dim=1, keepdim=True)             
                        list_of_outs.append(out_x)
                    inputs_x = torch.cat(list_of_outs,dim=0)         
                    
                    not_found = False
                    for iiidx, cl in enumerate(options):
                        if cl not in self.kg_dataset.entity_to_id:
                            cl = cl.replace(' ','')
                            if cl not in self.kg_dataset.entity_to_id:                            
                                not_found =True
                                print('{} not in entity list'.format(cl))
                                raise('Entity not found.')
                            else:
                                options[iiidx] =cl
                            
                    if not_found:
                        continue
                    
                            
                    idx = torch.Tensor([self.kg_dataset.entity_to_id[cl] for cl in options]).long()                
                    x_options = module.embeddings(idx.to(device)) 

                    x_options = x_options / x_options.norm(dim=1, keepdim=True)
                    
                    
                    scores  = module.logit_scale * inputs_x @ x_options.t()    
                    scores = scores / scores.norm(dim=1, keepdim=True)                
                    pred    = scores.argmax() %  scores.shape[1]                    
                    is_correct = pred == answer_idx                
                    if is_correct:
                        if qType in results_correct_dict.keys():
                            results_correct_dict[qType] = results_correct_dict[qType] + 1.
                        else:
                            results_correct_dict[qType] = 1.
                        

        keys = [key for key in results_correct_dict.keys()]
        sorted_key_list = []
        for i in range(20):
            for key in keys:        
                if key.startswith('qa{}_'.format(i)):
                    sorted_key_list.append(key)    
        all_results = []        
        total_correct = 0.
        total_count = 0.    
        if self.is_log:
            file = open('kml.results.txt','a')    
            if exp_settings is not None:
                file.write('\n' + exp_settings + '\n')   
        for key in sorted_key_list:
            qacc = results_correct_dict[key] / results_total_dict[key] *100.
            total_correct = total_correct + results_correct_dict[key]
            total_count = total_count + results_total_dict[key]
            all_results.append(qacc)
            to_print = '{} \t {:.3f}\n'.format(key.ljust(30), qacc)        
            print(to_print)   
            if self.is_log:
                file.write(to_print)
        macc = torch.Tensor(all_results).mean().item()
        acc = total_correct / total_count * 100.    
        to_print = '{} \t {:.3f}\n'.format('accuracy'.ljust(30), acc)
        print(to_print)     
        if self.is_log: 
            file.write(to_print) 
        to_print = '{} \t {:.3f}\n'.format('mean accuracy'.ljust(30), macc)
        print(to_print)     
        if self.is_log:
            file.write(to_print)  
            file.close()

    
        return acc, macc



class QA_Train():

    def __init__(self, kg_dataset):
        self.programGenerator = KML_Programs()
        self.myQDataset = QADatasetTrain()
        self.kg_dataset = kg_dataset
        # Here use the small validation set to find optimal setting.
        self.QA = QA(kg_dataset = kg_dataset , is_log=False, is_small= True)


    def train_on_question_db(self, module, qlr = 0.001, wd = 0.01, qepochs = 10, qbs = 32, device = 'cuda', program_type = 20):

        PRGRAM = self.programGenerator.get_predefined_program_for_qtype(program_type)            
        best_acc = 0
        best_macc = 0    
        criterion = nn.CrossEntropyLoss()    
        optimizer = torch.optim.AdamW(module.parameters(), lr=qlr, weight_decay = wd)    
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3)
        module.train()        
        best_module = None    
        accumilation_size = 1
        for ep in range(qepochs):
            loss_value = 0
            module.train()
            for i in tqdm(range(self.myQDataset.__len__())):
                options, answer_idx , step, qType, qid, taskName , pred, task_id = self.myQDataset.__getitem__(i)
                if qType in PRGRAM:              
                    inputs = PRGRAM[qType]['input']
                    if inputs == 'step':                        
                        inputs_x = module.embeddings(torch.Tensor([self.kg_dataset.entity_to_id[step]]).long().to(device))
                    if inputs == 'task':
                        inputs_x = module.embeddings(torch.Tensor([self.kg_dataset.entity_to_id[taskName]]).long().to(device))        


                    list_of_outs = []                  
                    for program_list in PRGRAM[qType]['func']:                      
                        out_x = module.train_with_q(inputs_x, program_list)                         
                        list_of_outs.append(out_x)
                    inputs_x = torch.cat(list_of_outs,dim=0)  

                    
                    
                    not_found = False
                    for iiidx, cl in enumerate(options):
                        if cl not in self.kg_dataset.entity_to_id:
                            cl = cl.replace(' ','')
                            if cl not in self.kg_dataset.entity_to_id:                            
                                not_found =True
                                print('{} not in entity list'.format(cl))
                                raise('Entity not found.')
                            else:
                                options[iiidx] =cl
                            
                    if not_found:
                        continue                    
                            
                    idx = torch.Tensor([self.kg_dataset.entity_to_id[cl] for cl in options]).long()                
                    x_options = module.embeddings(idx.to(device)) 
                    x_options = x_options / x_options.norm(dim=1, keepdim=True)

                    
                    
                    scores = module.logit_scale * inputs_x @ x_options.t()                    
                    if scores.shape[0] > 1:
                        scores = scores.mean(dim=0).unsqueeze(dim=0)                        
                    gtLBL = torch.Tensor([answer_idx]).long().to(device)                    
                    loss = criterion(scores, gtLBL)
                    loss_value = loss_value + loss.item()
                    if qbs > 1:
                        loss = loss / accumilation_size
                        accumilation_size = accumilation_size + 1.
                    loss.backward()

                    if (i+1) % qbs == 0:                        
                        optimizer.step()
                        optimizer.zero_grad()    
                        accumilation_size = 1.


            optimizer.zero_grad()  
            if accumilation_size > 1.   and qbs > 1:                
                optimizer.step()
            
            
            acc_, macc_ = self.QA.answer_question(module , program_type )
            scheduler.step(macc_)
            for param_group in optimizer.param_groups:
                print("lr: ", param_group['lr'])

            if best_macc < macc_:
                best_module = copy.deepcopy(module)
                best_macc = macc_
                best_acc = acc_
            print('\n[{}] [Loss for qa {}]  [M.Acc: {}]  [Acc: {}]  debug [accumilation_size {}]'.format(ep, loss_value, best_macc, best_acc, accumilation_size))
        return best_acc, best_macc, best_module
