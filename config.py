# 配置文件
config = dict()



# 测试路径（不戴口罩）
# config['train_data_path'] = 'datasets_test/test_train'
# config['train_data_index'] = 'datasets_test/test_train.csv'
# config['train_triplets_path'] = 'datasets/test_train.npy'
# config['LFW_data_path'] = 'datasets_test/lfw_funneled'
# config['LFW_pairs'] = 'datasets_test/LFW_pairs.txt'

config['resume_path'] = 'Model_training_checkpoints/model_resnet34_cheahom_triplet_epoch_20_roc0.9337.pt'

config['model'] = 'resnet34_attention' # resnet18\34\50\101\inceptionresnetv2 resnet34_cbam resnet34_cheahom resnet34_attention
config['optimizer'] = 'adam'      # sgd\adagrad\rmsprop\adam
config['predicter_path'] = 'Data_preprocessing/shape_predictor_68_face_landmarks.dat'

config['Learning_rate'] = 0.00001
config['image_size'] = 200        # inceptionresnetv2————299
config['epochs'] = 2

config['train_batch_size'] = 50
config['test_batch_size'] = 50

config['margin'] = 0.5
config['embedding_dim'] = 128
config['pretrained'] = False
config['save_last_model'] = True
config['num_train_triplets'] = 100000
config['num_workers'] = 4


config['train_data_path'] = 'Datasets/vggface2_train_250_face'
config['mask_data_path'] = 'Datasets/vggface2_train_250_mask'
config['train_data_index'] = 'Datasets/vggface2_train_250_face.csv'
config['train_triplets_path'] = 'Datasets/training_triplets_' + str(config['num_train_triplets']) + '.npy'
config['test_pairs_paths'] = 'Datasets/test_pairs.npy'
config['LFW_data_path'] = 'Datasets/lfw_funneled'
config['LFW_pairs'] = 'Datasets/LFW_pairs.txt'