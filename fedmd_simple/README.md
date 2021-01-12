
## 文件说明
data_utils.py:  数据加载和预处理
engine.py： 所有训练过程、预训练、联邦蒸馏训练、测试
FedMD_main.py： 主函数
model.py： 网络模型
options.py： 参数设置
plot_data.py： 结果图
sampling.py： 数据采样方法
conf： 四种配置文件

## 参数说明
### options参数
option里面参数只需要修改 --conf， --use_pretrained_model和  --gpu
不需要修改dataset和iid，自动从conf生成

## 运行示例  
```
python3 FedMD_main.py --conf ../conf/EMNIST_imbalance_conf.json  
```


## 输出
pretrain_model:下面是预训练模型.  
save:下面是输出的结果, 包括: 
1. pre_train_result: 模型预训练（在public数据集上训练和预测）
2. init_result: 模型在private数据集上进行训练
3. col_performance: 联邦蒸馏训练
4. pooled_train_result: 在所有的private上训练的理论最高结果



## 画图
plot_data.py 指定root_path
