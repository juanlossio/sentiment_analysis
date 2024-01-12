import numpy as np
import pandas as pd
import transformers
import torch
import os
import matplotlib as plt
#
from transformers import AutoTokenizer, OPTForSequenceClassification
#
from torch.utils.data import TensorDataset
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
#
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
#
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#
from transformers import AdamW, get_linear_schedule_with_warmup
#
from sys import argv


script, opt_option, device_ = argv

#########
## General parameters
#########

EPOCHS_ = 7 
MAX_LENGTH_ = 128 
BATCH_SIZES = [4, 8, 16, 32]
LEARNING_RATES = [3e-4, 1e-4, 5e-5, 3e-5, 1e-5]
opt_models = ['125m', '350m', '1.3b', '2.7b', '6.7b', '13b', '30b', '66b', '175b']


print('*********************************************')
print('+++++++++++++++++++++++++++++++++++++++++++++')
print('\nStarting execution\n')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"]=device_

#
#if torch.cuda.is_available():
#    device = torch.device('cuda')
#    print('There are %d GPU(s) available.' % torch.cuda.device_count())
#    print('We will use the GPU:', torch.cuda.get_device_name(0))
#    print('We will use the GPU:', device)
#else:
#    device = torch.device("cpu")
#    print('No GPU available, using the CPU instead.')

#device = torch.device("cpu")
device = torch.device('cuda')
print('We will use the GPU: {}'.format(device))
#
#
model_pretrained = "facebook/opt-{}".format(opt_models[int(opt_option)])
print('Model Pretrained =========================== {}'.format(model_pretrained))

print('\n+++++++++++++++++++++++++++++++++++++++++++++')
print('*********************************************')



#################################################################################
#################################################################################
################################## D A T A S E T S ##############################
#################################################################################
#################################################################################


path_train = '[PATH_TO_YOUR_TRAINING_DATASET]'
path_validation = '[PATH_TO_YOUR_VALIDATION_DATASET]'
cols_ = ['text','label']

df_train = pd.read_excel(path_train)
df_val = pd.read_excel(path_validation)


print("\n\n------------------------")
print("Training dataset: ")
print("------------------------")
print(df_train.shape)
print(df_train['label'].value_counts().sort_index())

print("\n\n------------------------")
print("Validation dataset: ")
print("------------------------")
print(df_val.shape)
print(df_val['label'].value_counts().sort_index())
print("\n\n")




#################################################################################
###################################
###  Encoding the Data
## Global variables
###################################
#################################################################################

tokenizer = AutoTokenizer.from_pretrained(model_pretrained)


encoded_data_train = tokenizer.batch_encode_plus(
    #sentences, 
    df_train.text.values.tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    max_length=MAX_LENGTH_,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df_val.text.values.tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    max_length=MAX_LENGTH_,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)



input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df_train.label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df_val.label.values)

dataset_train = TensorDataset(input_ids_train,attention_masks_train,labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)





target_names = ['Neutral', 'Positive', 'Negative']


from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted'), f1_score(labels_flat, preds_flat, average='macro'), accuracy_score(labels_flat, preds_flat)
    #return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    #label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    print(classification_report(labels_flat, preds_flat, target_names=target_names))
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        #print(f'Class: {label_dict_inverse[label]}')
        print(f'Class: {label}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')




def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


list_f_score = []
list_of_elem_results = []

for BATCH_SIZE_ in BATCH_SIZES:
    for LEARNING_RATE_ in LEARNING_RATES:

        torch.cuda.empty_cache()

        ##################
        ##### DATA LOADER
        ##################
        dataloader_train = DataLoader(dataset_train, sampler=SequentialSampler(dataset_train), batch_size=BATCH_SIZE_)
        dataloader_validation = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=BATCH_SIZE_)


        ######################
        ####### M O D E L ####
        ######################
        model = OPTForSequenceClassification.from_pretrained(
            model_pretrained,
            num_labels = len(df_train['label'].unique()),
            output_attentions=False,
            output_hidden_states=False
        )
        model.to(device)


        ##################
        ##### OPTIMIZER
        ##################    
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=len(dataloader_train)*EPOCHS_)


        print('*********************************************')

        for epoch in range(1, EPOCHS_+1):   
            #print("Batch size: {}\t-\tLearning rate: {}\t-\tEpoch: {}".format(BATCH_SIZE_, LEARNING_RATE_, epoch))
            #print("Batch size: {}\t-\tLearning rate: {}\t-\tEpoch: {}".format(
            #    BATCH_SIZES.index(BATCH_SIZE_)+1, LEARNING_RATES.index(LEARNING_RATE_)+1, epoch))

            ##################
            ##### TRAINING
            ##################
            model.train()
            loss_train_total = 0

            print("\nBatch size: {}, \t-\tLearning rate: {}, \t-\tEpoch: {}".format(BATCH_SIZE_, LEARNING_RATE_, epoch))


            for batch in dataloader_train:

                model.zero_grad()
                
                batch = tuple(b.to(device) for b in batch)
                
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[2],
                         }
                
                outputs = model(**inputs)
                
                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                         
                
            torch.save(model.state_dict(), '[PATH_MODELS_TO_SAVE]/OPT_{}_batch_{}_lr_{}_epoch_{}.model'.format(
                    opt_models[int(opt_option)],
                    BATCH_SIZES.index(BATCH_SIZE_)+1, LEARNING_RATES.index(LEARNING_RATE_)+1, epoch
                ))
            

            loss_train_avg = loss_train_total/len(dataloader_train)            
            val_loss, predictions, true_vals = evaluate(dataloader_validation)
            val_f1, val_f1_macro, val_accuracy = f1_score_func(predictions, true_vals)

            print(f"Training Loss: {loss_train_total}, \t -- \t Training Loss AVG: {loss_train_avg}")
            print("Validation Loss: {}, \t -- \t F-score (Weighted): {}, \t--\t Fs-core (Macro): {}, \t--\t Accuracy: {}".format(val_loss, val_f1, val_f1_macro, val_accuracy))


            list_f_score.append(val_f1)

            list_of_elem_results.append(["opt-{}".format(opt_models[int(opt_option)]), BATCH_SIZE_, LEARNING_RATE_, epoch, loss_train_total, loss_train_avg, val_loss, 
                val_f1, val_f1_macro, val_accuracy])


print('\n\n')
print('*********************************************')
print('+++++++++++++++++++++++++++++++++++++++++++++')
array_F = np.array(list_f_score)
print("List of F-scores: {}".format(array_F))
print("Max: {}".format(array_F.max()))
print("Arg Max: {}".format(np.argmax(array_F)))



df_res = pd.DataFrame(data=list_of_elem_results, columns=['OPT','BATCH_SIZE','LEARNING_RATE', 'EPOCH', 'loss_train_total', 
    'loss_train_avg', 'val_loss', 'val_f1_weighted', 'val_f1_macro', 'val_accuracy'])

df_res.to_excel("./results/aux_tables/opt-{}.xlsx".format(opt_models[int(opt_option)]), index=None)
print('*********************************************')
print('+++++++++++++++++++++++++++++++++++++++++++++')
print('\nEnd of execution\n')
print('+++++++++++++++++++++++++++++++++++++++++++++')
print('*********************************************')
