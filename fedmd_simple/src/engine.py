import os
import copy
import numpy as np
import torch
import math
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.optim import SGD,Adam
from data_utils import generate_alignment_data,data

def train_one_model(model,train_dataloader,test_dataloader,optimizer,epoch,device,criterion, min_delta=0.01,patience=3,
                    with_softmax = True,EarlyStopping=False,is_val = True):
    model.to(device)
    all_train_loss, all_train_acc, all_val_loss, all_val_acc = [],[],[],[]
    for iter in range(epoch):
        model.train()

        train_loss = []
        train_acc = []
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            log_probs = model(images)
            if with_softmax:
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                acc = torch.mean((labels ==torch.argmax(log_probs,dim=-1)).to(torch.float32))
                train_acc.append(acc.item())
                train_loss.append(loss.item())
            else:
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
        if with_softmax:
            all_train_loss.append(sum(train_loss)/len(train_loss))
            all_train_acc.append(sum(train_acc)/len(train_acc))
            if is_val:
                val_loss,val_acc = val_one_model(model,test_dataloader,criterion,device)
                all_val_loss.append(val_loss)
                all_val_acc.append(val_acc)
                if EarlyStopping and len(all_val_acc)>patience:
                    if max(all_val_acc[-patience:])-min(all_val_acc[-patience:])<=min_delta:
                        break
    return all_train_loss,all_train_acc,all_val_loss,all_val_acc

def val_one_model(model,dataloader,criterion=None,device= torch.device('cuda')):
    model.eval()
    acc = []
    loss_out = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            if criterion is not None:
                loss = criterion(log_probs, labels)
                loss_out.append(loss.item())
            acc_ = torch.mean((labels == torch.argmax(log_probs, dim=-1)).to(torch.float32))
            acc.append(acc_.item())
        if criterion is not None:
            return sum(loss_out)/len(loss_out),sum(acc)/len(acc)
        else:
            return sum(acc)/len(acc)


def get_cosdist(a,b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down
    return 

def get_models_logits(raw_logits, threshold, N_models):  #input:  

    flatten_logits = []
    cosdist = []
    tmp_cosdist = []
    tmp_logits = []
    models_logits = []
    # threshold = -1 ############# CHANGE ###############
    add_model_count = 0

    for index in range(N_models):
        flatten_logits.append(raw_logits[index].flatten())

    for index in range(N_models):
        for index2 in range(N_models):
            if index == index2:
                tmp_cosdist.append(1)
            else:
                tmp_cosdist.append(get_cosdist(flatten_logits[index],flatten_logits[index2]))
        cosdist.append(tmp_cosdist)   #cosdist = [model1's all cosdist list,model2's all cosdist list,...]
        tmp_cosdist = []

    #judge threshold

    for index in range(N_models):
        for index2 in range(N_models):
            if cosdist[index][index2] > threshold: #cosdist (-1,1), cosdist small means angle big
                #print(index,index2,cosdist[index][index2])
                tmp_logits.append(raw_logits[index2])
                add_model_count += 1
        print("model_num of model {0} used".format(index))
        print("-- {0} --".format(add_model_count))
        tmp_logits = np.sum(tmp_logits,axis=0)
        tmp_logits /= add_model_count
        models_logits.append(tmp_logits)
        add_model_count = 0
        tmp_logits = []

    return models_logits

def predict(model,dataarray,device):
    model.eval()
    out= []
    bs = 32
    dataarray = dataarray.astype(np.float32)
    with torch.no_grad():
        for ind in range(0,len(dataarray),bs):
            data = dataarray[ind:(ind+bs)]
            data = torch.from_numpy(data).to(device)

            logit = model(data)
            out.append(logit.cpu().numpy())
    out = np.concatenate(out)
    return out

def train_models(models, X_train, y_train, X_test, y_test,
                 device = 'cpu',save_dir = "./", save_names = None,
                 early_stopping = True, min_delta = 0.001,num_workers=0,
                 batch_size = 128, epochs = 20, is_shuffle=True,patience=3
                ):
    '''
    Train an array of models on the same dataset. 
    We use early termination to speed up training. 
    '''
    
    resulting_val_acc = []
    record_result = []

    for n, model in enumerate(models):
        print("Training model ", n)
        model.to(device)
        train_dataloader = DataLoader(data(X_train, y_train), batch_size=batch_size,shuffle=is_shuffle,
                                      sampler=None,batch_sampler= None,num_workers= num_workers,drop_last = False)
        test_dataloader = DataLoader(data(X_test, y_test), batch_size=batch_size,
                                      sampler=None,batch_sampler= None,num_workers= num_workers,drop_last = False)
        optimizer = Adam(model.parameters(),lr=0.001)
        criterion = nn.CrossEntropyLoss().to(device)
        train_loss,train_acc,val_loss,val_acc = train_one_model(model, train_dataloader,test_dataloader, optimizer,
                                                                epochs, device, criterion,  min_delta,patience,
                        EarlyStopping=early_stopping,is_val=True)
        
        resulting_val_acc.append(val_acc[-1])
        record_result.append({"train_acc": train_acc,
                              "val_acc": val_acc,
                              "train_loss": train_loss,
                              "val_loss": val_loss})

        if save_dir is not None:
            save_dir_path = os.path.abspath(save_dir)
            #make dir
            os.makedirs(save_dir_path,exist_ok=True)
            if save_names is None:
                file_name = os.path.join(save_dir_path , "model_{0}".format(n) + ".pt")
            else:
                file_name = os.path.join(save_dir_path , save_names[n] + ".pt")
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epochs}
            torch.save(state,file_name)

    
    print("pre-train accuracy: ")
    print(resulting_val_acc)
        
    return record_result


class FedMD():
    def __init__(self, parties, public_dataset,
                 private_data, total_private_data,
                 private_test_data,
                 FedMD_params,
                 model_init_params,
                 calculate_theoretical_upper_bounds_params,
                 device='cuda'):

        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_alignment = FedMD_params['N_alignment']

        self.N_rounds = FedMD_params['N_rounds']
        self.N_logits_matching_round = FedMD_params['N_logits_matching_round']
        self.logits_matching_batchsize = FedMD_params['logits_matching_batchsize']
        self.N_private_training_round = FedMD_params['N_private_training_round']
        self.private_training_batchsize = FedMD_params['private_training_batchsize']
        self.device = device

        print("calculate the theoretical upper bounds for participants: ")
        self.upper_bounds = []
        self.pooled_train_result = []
        #
        # 参数
        epochs = calculate_theoretical_upper_bounds_params['epochs']
        min_delta= calculate_theoretical_upper_bounds_params['min_delta']
        patience= calculate_theoretical_upper_bounds_params['patience']
        batch_size= calculate_theoretical_upper_bounds_params['batch_size']
        is_shuffle= calculate_theoretical_upper_bounds_params['is_shuffle']
        num_workers = calculate_theoretical_upper_bounds_params['num_workers']
        for model in parties:
            model_ub = copy.deepcopy(model)
            train_dataloader = DataLoader(data(total_private_data["X"], total_private_data["y"]), batch_size=batch_size,
                                          shuffle=is_shuffle,
                                          sampler=None, batch_sampler=None, num_workers=num_workers, drop_last=False)
            test_dataloader = DataLoader(data(private_test_data["X"], private_test_data["y"]), batch_size=batch_size,
                                         sampler=None, batch_sampler=None, num_workers=num_workers, drop_last=False)

            optimizer = Adam(model_ub.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()


            train_loss, train_acc, val_loss, val_acc = train_one_model(model_ub, train_dataloader, test_dataloader,
                                                                       optimizer,
                                                                       epochs, self.device, criterion, min_delta, patience,
                                                                       EarlyStopping=False, is_val=True)


            self.upper_bounds.append(val_acc[-1])
            self.pooled_train_result.append({"val_acc": val_acc,
                                             "acc": train_acc})

            del model_ub
        print("the upper bounds are:", self.upper_bounds)

        self.collaborative_parties = []
        self.init_result = []

        print("start model initialization: ")

        epochs = model_init_params['epochs']
        min_delta= model_init_params['min_delta']
        patience= model_init_params['patience']
        batch_size= model_init_params['batch_size']
        is_shuffle= model_init_params['is_shuffle']
        num_workers = model_init_params['num_workers']
        self.num_workers =num_workers
        for i in range(self.N_parties):
            model = parties[i]
            
            train_dataloader = DataLoader(data(private_data[i]["X"], private_data[i]["y"]), batch_size=batch_size, shuffle=is_shuffle,
                                          sampler=None, batch_sampler=None, num_workers=num_workers, drop_last=False)
            test_dataloader = DataLoader(data(private_test_data["X"], private_test_data["y"]), batch_size=batch_size,
                                         sampler=None, batch_sampler=None, num_workers=num_workers, drop_last=False)
            optimizer = Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()


            train_loss, train_acc, val_loss, val_acc = train_one_model(model, train_dataloader, test_dataloader,
                                                                       optimizer,
                                                                       epochs, self.device, criterion, min_delta, patience,
                                                                       EarlyStopping=True, is_val=True)

            self.collaborative_parties.append(model)

            self.init_result.append({"val_acc": val_acc,
                                     "train_acc": train_acc,
                                     "val_loss": val_loss,
                                     "train_loss": train_loss,
                                     })
        # print('model initialization are:')
        # END FOR LOOP

        print("finish model initialization: ")
    def collaborative_training(self):
        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0
        device = torch.device('cuda')

        threshold = -1

        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = generate_alignment_data(self.public_dataset["X"],
                                                     self.public_dataset["y"],
                                                     self.N_alignment)

            print("round ", r)

            print("update logits ... ")
            # update logits
            logits = []
            logits_test1 = []     #adddddddddddd
            model_idx = 0         #adddddddddddd
            for model in self.collaborative_parties:
                model_idx += 1
                X_data = copy.deepcopy(alignment_data["X"])
                if len(X_data.shape)==4:
                    X_data = np.transpose(X_data, (0, 3, 1, 2))
                else:
                    X_data = np.repeat(X_data[:,None],repeats=3,axis=1)
                logits.append(predict(model, X_data, device))
                #if r == self.N_rounds:                                   #adddddddddddd
                logits_test1.append(predict(model, X_data, device))  #adddddddddddd
            logits = np.sum(logits,axis =0)
            logits /= self.N_parties

            #--------------------
            logits_models = get_models_logits(logits_test1,threshold,self.N_parties)

            #--------------------

            if r == self.N_rounds:                                   #adddddddddddd
                #print("logits_test1:")                               #adddddddddddd
                #print(logits_test1)                               #adddddddddddd
                print("logits_models:")                       #adddddddddddd
                print(logits_models)                                        #adddddddddddd
                print("logits_shape:")                       #adddddddddddd
                print(logits.shape)                                        #adddddddddddd

            # test performance
            print("test performance ... ")

            for index, model in enumerate(self.collaborative_parties):
                dataloader =  DataLoader(data(self.private_test_data["X"], self.private_test_data["y"]), batch_size=32,
                                          shuffle=True,
                                          sampler=None, batch_sampler=None, num_workers=self.num_workers, drop_last=False)
                acc = val_one_model(model, dataloader, criterion=None, device=torch.device('cuda'))

                collaboration_performance[index].append(acc)
                print(collaboration_performance[index][-1])


            r += 1
            if r > self.N_rounds:
                break

            print("updates models ...")
            #
            for index, model in enumerate(self.collaborative_parties):
                print("model {0} starting alignment with public logits... ".format(index))

                train_dataloader = DataLoader(data(alignment_data["X"], logits_models[index]), batch_size=self.logits_matching_batchsize,  ########## SWITCH: logits<=>logits_models[index]
                                              shuffle=True,
                                              sampler=None, batch_sampler=None, num_workers=self.num_workers, drop_last=False)
                test_dataloader = None
                optimizer = Adam(model.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                epoch = self.N_logits_matching_round


                train_one_model(model, train_dataloader, test_dataloader, optimizer, epoch, self.device, criterion,
                                    with_softmax = False,EarlyStopping=False, is_val=False)

                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))

                train_dataloader = DataLoader(data(self.private_data[index]["X"], self.private_data[index]["y"]),
                                              batch_size=self.private_training_batchsize,shuffle=True,
                                              sampler=None, batch_sampler=None, num_workers=self.num_workers, drop_last=False)
                test_dataloader = None
                optimizer = Adam(model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                epoch = self.N_private_training_round

                train_one_model(model, train_dataloader, test_dataloader, optimizer, epoch, self.device, criterion,
                                    EarlyStopping=False, is_val=False)

                print("model {0} done private training. \n".format(index))
            # END FOR LOOP

        # END WHILE LOOP
        return collaboration_performance