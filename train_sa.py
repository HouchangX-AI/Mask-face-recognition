# 路径置顶
import sys 
import os 
sys.path.append(os.getcwd()) 
# 导入包 
from torch.nn.modules.distance import PairwiseDistance
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import time
# 导入文件
# from Models.Model_for_facenet import model, optimizer_model, start_epoch, flag_train_multi_gpu
from Data_loader.Data_loader_facenet import train_dataloader, test_dataloader
from Losses.Triplet_loss import TripletLoss
from validate_on_LFW import evaluate_lfw
from config import config
from Models.Only_attention import Resnet34_attention

# 随机种子
seed = 0
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)



print("Using {} model architecture.".format(config['model']))
start_epoch = 0
if config['model'] == "resnet34_attention":
    model = Resnet34_attention()

flag_train_gpu = torch.cuda.is_available()
flag_train_multi_gpu = False
if flag_train_gpu and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model.cuda()
    flag_train_multi_gpu = True
    print('Using multi-gpu training.')
elif flag_train_gpu and torch.cuda.device_count() == 1:
    model.cuda()
    print('Using single-gpu training.')

# optimizer
print("Using {} optimizer.".format(config['optimizer']))
if config['optimizer'] == "sgd":
    optimizer_model = torch.optim.SGD(model.parameters(), lr=config['Learning_rate'])
    
elif config['optimizer'] == "adagrad":
    optimizer_model = torch.optim.Adagrad(model.parameters(), lr=config['Learning_rate'])
    
elif config['optimizer'] == "rmsprop":
    optimizer_model = torch.optim.RMSprop(model.parameters(), lr=config['Learning_rate'])
    
elif config['optimizer'] == "adam":
    optimizer_model = torch.optim.Adam(model.parameters(), lr=config['Learning_rate'])

if os.path.isfile(config['resume_path']):
    print("\nLoading checkpoint {} ...".format(config['resume_path']))
    checkpoint = torch.load(config['resume_path'])
    start_epoch = checkpoint['epoch']
    # if flag_train_multi_gpu:
    #     model.module.load_state_dict(checkpoint['model_state_dict'])
    # else:
    #     model.load_state_dict(checkpoint['model_state_dict'])
    pretrained_dict = checkpoint['model_state_dict']
    model_dict=model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
    print("\nCheckpoint loaded: start epoch from checkpoint = {}\nRunning for {} epochs.\n".format(start_epoch,config['epochs']+start_epoch))


# 打卡时间、epoch
total_time_start = time.time()
start_epoch = start_epoch
end_epoch = start_epoch + config['epochs']
# 导入l2计算的
l2_distance = PairwiseDistance(2).cuda()
# 为了打日志先预制个最佳auc和最佳acc在前头
best_roc_auc = -1
best_accuracy = -1
print('Countdown 3 seconds')
time.sleep(1)
print('Countdown 2 seconds')
time.sleep(1)
print('Countdown 1 seconds')
time.sleep(1)

# epoch大循环
for epoch in range(start_epoch, end_epoch):
    print("\ntraining on TrainDataset! ...")
    epoch_time_start = time.time()
    attention_loss_sum = 0
    num = 0


    model.train() # 训练模式
    for name, param in model.named_parameters():
        if 'sa1' not in name:
            param.requires_grad = False
        else:
            print('{} requires grad\n====================================================================='.format(name))
            param.requires_grad = True
    # step小循环
    progress_bar = enumerate(tqdm(train_dataloader))
    for batch_idx, (batch_sample) in progress_bar:

        # 获取本批次的数据
        # 取出三张人脸图(batch*图)
        anc_img = batch_sample['anc_img'].cuda()
        pos_img = batch_sample['pos_img'].cuda()
        neg_img = batch_sample['neg_img'].cuda()
        # 取出三张mask图(batch*图)
        mask_anc = batch_sample['mask_anc'].cuda()
        mask_pos = batch_sample['mask_pos'].cuda()
        mask_neg = batch_sample['mask_neg'].cuda()

        # 模型运算
        # 前向传播过程-拿模型分别跑三张图，生成embedding和loss（在训练阶段的输入是两张图，输出带loss，而验证阶段输入一张图，输出只有embedding）
        anc_attention_loss = model((anc_img, mask_anc))
        pos_attention_loss = model((pos_img, mask_pos))
        neg_attention_loss = model((neg_img, mask_neg))

        

        attention_loss = torch.cat([anc_attention_loss, pos_attention_loss, neg_attention_loss])
        num += len(attention_loss)
        attention_loss = torch.mean(attention_loss).cuda()
        attention_loss = attention_loss.type(torch.FloatTensor)

        # 计算总损失
        LOSS = attention_loss


        # 反向传播过程
        optimizer_model.zero_grad()
        LOSS.backward()
        optimizer_model.step()

        # 记录log相关信息
        # 计算本个批次内的困难样本数量
        # 计算这个epoch内的总三元损失和计算损失所用的困难样本个数
        attention_loss_sum += attention_loss.item()


    # 计算这个epoch里的平均损失
    epoch_time_end = time.time()
    avg_attention_loss = attention_loss_sum / num


    # 打印日志内容
    print('Epoch {}:\n \
           train_log:\tatt_loss: {:.4f}\thard_sample: {}\ttrain_time: {}'.format(
            epoch+1,
            avg_attention_loss,
            num,
            (epoch_time_end - epoch_time_start)/3600
            
        )
    )

    # 保存日志文件
    with open('logs/lfw_{}_log_triplet.txt'.format(config['model']), 'a') as f:
        val_list = [
                'epoch: ' + str(epoch+1) + '\t',
                'train:\t',
                'att_loss: ' + str('%.4f' % avg_attention_loss) + '\t',
                'hard_sample: ' + str(num) + '\t',
                'train_time: ' + str('%.4f' % ((epoch_time_end - epoch_time_start)/3600))
        ]
        log = ''.join(str(value) for value in val_list)
        f.writelines(log + '\n')


    # 保存模型权重
    if save:
        state = {
            'epoch': epoch+1,
            'embedding_dimension': config['embedding_dim'],
            'batch_size_training': config['train_batch_size'],
            'model_state_dict': model.state_dict(),
            'model_architecture': config['model'],
            'optimizer_model_state_dict': optimizer_model.state_dict()
        }
        # 
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()
        # For storing best euclidean distance threshold during LFW validation
        # if flag_validate_lfw:
            # state['best_distance_threshold'] = np.mean(best_distances)
        # 
        torch.save(state, 'Model_training_checkpoints/model_{}_triplet_epoch_{}_roc{:.4f}.pt'.format(config['model'], epoch+1, roc_auc))

# Training loop end
total_time_end = time.time()
total_time_elapsed = total_time_end - total_time_start
print("\nTraining finished: total time elapsed: {:.2f} hours.".format(total_time_elapsed/3600))




