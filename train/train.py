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
from Models.Model_for_facenet import model, optimizer_model, start_epoch, flag_train_multi_gpu
from Data_loader.Data_loader_facenet import train_dataloader, test_dataloader
from Losses.Triplet_loss import TripletLoss
from validate_on_LFW import evaluate_lfw
from config import config

# 随机种子
seed = 0
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

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
    triplet_loss_sum = 0
    attention_loss_sum = 0
    num_hard = 0


    model.train() # 训练模式
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
        anc_embedding, anc_attention_loss = model((anc_img, mask_anc))
        pos_embedding, pos_attention_loss = model((pos_img, mask_pos))
        neg_embedding, neg_attention_loss = model((neg_img, mask_neg))

        # 寻找困难样本
        # 计算embedding的L2
        pos_dist = l2_distance.forward(anc_embedding, pos_embedding)
        neg_dist = l2_distance.forward(anc_embedding, neg_embedding)
        # 找到满足困难样本标准的样本
        all = (neg_dist - pos_dist < config['margin']).cpu().numpy().flatten()
        hard_triplets = np.where(all == 1)
        if len(hard_triplets[0]) == 0:
            continue
        
        # 选定困难样本——困难embedding
        anc_hard_embedding = anc_embedding[hard_triplets].cuda()
        pos_hard_embedding = pos_embedding[hard_triplets].cuda()
        neg_hard_embedding = neg_embedding[hard_triplets].cuda()
        # 选定困难样本——困难样本对应的attention loss
        hard_anc_attention_loss = anc_attention_loss[hard_triplets]
        hard_pos_attention_loss = pos_attention_loss[hard_triplets]
        hard_neg_attention_loss = neg_attention_loss[hard_triplets]

        # 损失计算
        # 计算这个批次困难样本的三元损失
        triplet_loss = TripletLoss(margin=config['margin']).forward(
            anchor=anc_hard_embedding,
            positive=pos_hard_embedding,
            negative=neg_hard_embedding
        ).cuda()
        # 计算这个批次困难样本的attention loss（这个loss实际上在forward过程里已经计算了，这里就是整合一下求个mean）
        hard_attention_loss = torch.cat([hard_anc_attention_loss, hard_pos_attention_loss, hard_neg_attention_loss])
        hard_attention_loss = torch.mean(hard_attention_loss).cuda()
        hard_attention_loss = hard_attention_loss.type(torch.FloatTensor)
        # 计算总顺势
        # LOSS = triplet_loss + hard_attention_loss
        LOSS = triplet_loss

        # 反向传播过程
        optimizer_model.zero_grad()
        LOSS.backward()
        optimizer_model.step()

        # 记录log相关信息
        # 计算本个批次内的困难样本数量
        num_hard += len(anc_hard_embedding)
        # 计算这个epoch内的总三元损失和计算损失所用的困难样本个数
        triplet_loss_sum += triplet_loss.item()
        attention_loss_sum += hard_attention_loss.item()


    # 计算这个epoch里的平均损失
    avg_triplet_loss = 0 if (num_hard == 0) else triplet_loss_sum / num_hard
    avg_attention_loss = 0 if (num_hard == 0) else attention_loss_sum / num_hard
    avg_loss = avg_triplet_loss + avg_attention_loss
    epoch_time_end = time.time()


    # 出测试集准确度
    print("Validating on TestDataset! ...")
    model.eval() # 验证模式
    with torch.no_grad(): # 不传梯度了
        distances, labels = [], []

        progress_bar = enumerate(tqdm(test_dataloader))
        for batch_index, (data_a, data_b, label) in progress_bar:
            # ‘img1, img2, issame’
            # data_a, data_b, label这仨是一批的矩阵
            data_a = data_a.cuda()
            data_b = data_b.cuda()
            label = label.cuda()

            output_a, output_b = model(data_a), model(data_b)
            distance = l2_distance.forward(output_a, output_b)  
            # 列表里套矩阵
            labels.append(label.cpu().detach().numpy())
            distances.append(distance.cpu().detach().numpy())
        # 展平
        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])

        true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
            tar, far = evaluate_lfw(
                distances=distances,
                labels=labels
            )


    # 打印并保存日志
    # 从之前的文件里读出来最好的roc和acc，并进行更新
    if os.path.exists('logs/lfw_{}_log_triplet.txt'.format(config['model'])):
        with open('logs/lfw_{}_log_triplet.txt'.format(config['model']), 'r') as f:
            lines = f.readlines()
            my_line = lines[-3]
            my_line = my_line.split('\t')
            best_roc_auc = float(my_line[3].split(':')[1])
            best_accuracy = float(my_line[5].split(':')[1])

    # 确定什么时候保存权重：最后一个epoch就保存，AUC出现新高就保存
    save = False
    if config['save_last_model'] and epoch == end_epoch - 1:
        save = True
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        save = True
    if np.mean(accuracy) > best_accuracy:
        best_accuracy = np.mean(accuracy)

    # 打印日志内容
    print('Epoch {}:\n \
           train_log:\tLOSS: {:.4f}\ttri_loss: {:.4f}\tatt_loss: {:.4f}\thard_sample: {}\ttrain_time: {}\n \
           test_log:\tAUC: {:.4f}\tACC: {:.4f}+-{:.4f}\trecall: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\t'.format(
            epoch+1,
            avg_loss,
            avg_triplet_loss,
            avg_attention_loss,
            num_hard,
            (epoch_time_end - epoch_time_start)/3600,
            roc_auc,
            np.mean(accuracy),
            np.std(accuracy),
            np.mean(recall),
            np.std(recall),
            np.mean(precision),
            np.std(precision),
        )
    )

    # 保存日志文件
    with open('logs/lfw_{}_log_triplet.txt'.format(config['model']), 'a') as f:
        val_list = [
                'epoch: ' + str(epoch+1) + '\t',
                'train:\t',
                'LOSS: ' + str('%.4f' % avg_loss) + '\t',
                'tri_loss: ' + str('%.4f' % avg_triplet_loss) + '\t',
                'att_loss: ' + str('%.4f' % avg_attention_loss) + '\t',
                'hard_sample: ' + str(num_hard) + '\t',
                'train_time: ' + str('%.4f' % ((epoch_time_end - epoch_time_start)/3600))
        ]
        log = ''.join(str(value) for value in val_list)
        f.writelines(log + '\n')
        val_list = [
                'epoch: ' + str(epoch+1) + '\t',
                'test:\t',
                'auc: ' + str('%.4f' % roc_auc) + '\t',
                'best_auc: ' + str('%.4f' % best_roc_auc) + '\t',
                'acc: ' + str('%.4f' % np.mean(accuracy)) + '+-' + str('%.4f' % np.std(accuracy)) + '\t',
                'best_acc: ' + str('%.4f' % best_accuracy) + '\t',
                'recall: ' + str('%.4f' % np.mean(recall)) + '+-' + str('%.4f' % np.std(recall)) + '\t',
                'precision: ' + str('%.4f' % np.mean(precision)) + '+-' + str('%.4f' % np.std(precision)) + '\t',
                'best_distances: ' + str('%.4f' % np.mean(best_distances)) + '+-' + str('%.4f' % np.std(best_distances)) + '\t',
                'tar_m: ' + str('%.4f' % np.mean(tar)) + '\t',
        ]
        log = ''.join(str(value) for value in val_list)
        f.writelines(log + '\n')
        val_list = [
                'epoch: ' + str(epoch+1) + '\t',
                'config:\t',
                'LR: ' + str(config['Learning_rate']) + '\t',
                'optimizer: ' + str(config['optimizer']) + '\t',
                'embedding_dim: ' + str(config['embedding_dim']) + '\t',
                'pretrained: ' + str(config['pretrained']) + '\t',
                'image_size: ' + str(config['image_size'])
        ]
        log = ''.join(str(value) for value in val_list)
        f.writelines(log + '\n' + '\n')

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




