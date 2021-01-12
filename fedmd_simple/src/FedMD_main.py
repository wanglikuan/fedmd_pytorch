import os
import pickle
import torch
from options import args_parser
from data_utils import getdataset
from model import cnn_2layer_fc_model, cnn_3layer_fc_model
from engine import train_models,FedMD



CANDIDATE_MODELS = {"2_layer_CNN": cnn_2layer_fc_model, 
                    "3_layer_CNN": cnn_3layer_fc_model} 

def pretrain(pre_train_data,config_info):
    '''第一步，预训练'''
    '''载入模型'''
    parties = []
    args, conf_dict, model_saved_dir, save_dir_path, device = config_info
    n_classes = len(conf_dict["private_classes"])+len(conf_dict['public_classes'])
    for i, item in enumerate(conf_dict["models"]):
        model_name = item["model_type"]
        model_params = item["params"]
        tmp = CANDIDATE_MODELS[model_name](n_classes=n_classes,
                                           **model_params)
        print("model {0} : {1}".format(i, conf_dict["model_saved_names"][i]))
        parties.append(tmp)

        del model_name, model_params, tmp

    '''预训练模型或者载入预训练模型'''
    if not args.use_pretrained_model:
        # END FOR LOOP
        X_train, y_train, X_test, y_test = pre_train_data
        pre_train_result = train_models(parties,
                                        X_train, y_train,
                                        X_test, y_test, device=device,
                                        save_dir=model_saved_dir, save_names=conf_dict["model_saved_names"],
                                        early_stopping=conf_dict["early_stopping"],
                                        **conf_dict["pre_train_params"]
                                        )
        with open(os.path.join(save_dir_path, 'pre_train_result.pkl'), 'wb') as f:
            pickle.dump(pre_train_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:

        for model_name, model in zip(conf_dict["model_saved_names"], parties):
            checkpoint = torch.load(os.path.join(model_saved_dir, model_name + '.pt'))
            model.load_state_dict(checkpoint['net'])

    return parties

def main():
    '''参数导入'''
    args = args_parser()
    with open(args.conf, "r") as f:
        conf_dict = eval(f.read())

    '''新建保存路径'''
    sub_path = os.path.join(args.dataset, 'iid' if args.iid else 'no_iid')
    save_dir_path = os.path.abspath(os.path.join(conf_dict["result_save_dir"], sub_path))
    model_saved_dir = os.path.abspath(os.path.join(conf_dict["model_saved_dir"], sub_path))

    os.makedirs(save_dir_path, exist_ok=True)
    os.makedirs(model_saved_dir, exist_ok=True)


    '''gpu'''
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu is not None else 'cpu'

    '''导入数据'''
    pre_train_data, FeMD_data = getdataset(args, conf_dict)
    public_dataset, private_data, total_private_data, private_test_data = FeMD_data

    # 第一步 预训练
    config_info = args, conf_dict, model_saved_dir, save_dir_path, device
    parties = pretrain(pre_train_data, config_info)

    '''联邦蒸馏学习'''
    fedmd = FedMD(parties,
                  public_dataset=public_dataset,
                  private_data=private_data,
                  total_private_data=total_private_data,
                  private_test_data=private_test_data,
                  FedMD_params = conf_dict['FedMD_params'],
                  model_init_params=conf_dict['model_init_params'],
                  calculate_theoretical_upper_bounds_params=conf_dict['calculate_theoretical_upper_bounds_params'],
                  device=device)
    '''计算理论上线和初始化各个客户模型'''
    initialization_result = fedmd.init_result
    pooled_train_result = fedmd.pooled_train_result
    '''联邦蒸馏学习'''
    collaboration_performance = fedmd.collaborative_training()

    # 保存联邦蒸馏学习结果
    with open(os.path.join(save_dir_path, 'init_result.pkl'), 'wb') as f:
        pickle.dump(initialization_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_dir_path, 'pooled_train_result.pkl'), 'wb') as f:
        pickle.dump(pooled_train_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_dir_path, 'col_performance.pkl'), 'wb') as f:
        pickle.dump(collaboration_performance, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
        