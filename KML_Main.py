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

from KML import KML
from KML import KGData, KGDataSinglePositive
from KML import KML_Train
from KML_utils import get_KB_data
from KML_utils import QA , QA_Train
from KML_commons import get_inverse_relation_names_for_PKVQA_Module
import torch
import os


import argparse

parser = argparse.ArgumentParser(description="NS")

parser.add_argument("--hdim",                   default=128,     type=int,     help="Hidden dimenstion of the KML module.")
parser.add_argument("--bs",                     default=32,       type=int,     help="KG training batch size, ie. the number of triplets.")
parser.add_argument("--qbs",                    default=1,        type=int,     help="Question training batch size.")
parser.add_argument("--epochs",                 default=100,      type=int,     help="Number of KG training epochs.")
parser.add_argument("--qepochs",                default=100,      type=int,     help="Number of QA training epochs.")
parser.add_argument("--qlr",                    default=0.001,    type=float,   help="Learning rate for QA learning.")
parser.add_argument("--lr",                     default=0.01,     type=float,   help="Learning rate for KG learning.")
parser.add_argument("--wd",                     default=0.01,     type=float,   help="Weight decay for KG learning")
parser.add_argument("--embsize",                default=256,      type=int,     help="Size of the KG entity embedding.")
parser.add_argument("--temp",                   default=32,       type=float,   help="Temperature")
parser.add_argument("--embinit",                default=None,     type=str,     help="Embedding initialization function.")
parser.add_argument("-s", "--isscheduler",      action="store_true" ,           help="Use scheduler.")
parser.add_argument("--program_type",           default=20,      type=int,      help="The program type to use.")

args = parser.parse_args()  
if args.temp == None:
    args.temp = args.bs 

if args.embinit is None:
    embd_init_func = None
elif args.embinit == 'clip':    
    from KML_commons import ClipTextEncoder
    embd_init_func = ClipTextEncoder()
    args.embsize = 512
elif args.embinit == 'bert':
    from KML_commons import BertTextEncoder
    embd_init_func =  BertTextEncoder()
    args.embsize = 768
elif args.embinit == 'mini':
    from KML_commons import MiniLMTextEncoder
    embd_init_func =  MiniLMTextEncoder()
    args.embsize = 384
elif args.embinit == 'roberta':
    from KML_commons import RobertaTextEncoder
    embd_init_func =  RobertaTextEncoder()
    args.embsize = 768

print('\n\n')
print(args) 
print('\n\n')

kg_dataset = KGDataSinglePositive(get_KB_data())
relations_list = kg_dataset.get_relation_list()





kml_model = KML(mudule_name_list=relations_list, 
                kgdataset= kg_dataset, 
                hidden_dim=args.hdim, 
                embd_init_func= embd_init_func, 
                embedding_dim=args.embsize, 
                temparature=args.temp)
nN = torch.tensor([p.view(-1).size()[0] for p in kml_model.parameters()]).sum()
print('params : {}'.format(nN))

if os.path.exists('kml_ckpt') == False:
    os.makedirs('kml_ckpt')
model_save_name = 'kml_ckpt/kml_model_h{}.b{}.e{}.lr{}.wd{}.emb{}.temp{}.s{}-init.{}.pt'.format(
    args.hdim, args.bs, args.epochs, args.lr, args.wd, args.embsize, args.temp, args.isscheduler, args.embinit)

is_KG_Based_Trainining = False
is_QA_Based_Training = False

        


if is_KG_Based_Trainining:

    # Start KG Based Training.
    kml_model = KML_Train(kgdataset     = kg_dataset, 
            module        = kml_model, 
            batch_size    = args.bs, 
            learning_rate = args.lr, 
            weight_decay  = args.wd, 
            device        = 'cuda', 
            epochs        = args.epochs,
            inv_rel_func  = get_inverse_relation_names_for_PKVQA_Module,
            post_Process_func = None,
            use_of_scheduler = True
            )
    torch.save(kml_model.state_dict(), model_save_name)
    print('Model saved at :', model_save_name)

else:
    print('model is loading')
    kml_model.load_state_dict(torch.load(model_save_name))
    kml_model.to('cuda')

# Start QA Based Training (optional).
if is_QA_Based_Training:    
    QA_STRING = f'QA_TRAINED_MODEL_qlr{args.qlr}_qepochs{args.qepochs}_qbs{args.qbs}'
    model_save_name = 'kml_ckpt/kml_model_h{}.b{}.e{}.lr{}.wd{}.emb{}.temp{}.s{}-init.{}--{}.pt'.format(
        args.hdim, args.bs, args.epochs, args.lr, args.wd, args.embsize, args.temp, args.isscheduler, args.embinit, QA_STRING)

    print('qa save model : ', model_save_name)

    if os.path.exists(model_save_name) == False:

        qA_Train = QA_Train(kg_dataset=kg_dataset)
        best_acc, best_macc, best_module = qA_Train.train_on_question_db(kml_model, qlr = args.qlr, wd = args.wd, qepochs = args.qepochs, qbs = args.qbs, device = 'cuda', program_type = args.program_type)
        torch.save(best_module.state_dict(), model_save_name)
        kml_model = best_module

# DO Final evaluation with full QA Test set.
qa_module = QA(kg_dataset=kg_dataset)
exp_settings = str(args)
qa_module.answer_question(module=kml_model, exp_settings=exp_settings, program_type=args.program_type)
