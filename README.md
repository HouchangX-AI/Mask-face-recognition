# 口罩人脸识别(Mask-face-recognition)
## 项目由来
由于疫情影响大家都带上了口罩，原来的人脸识别就不好事了，怎么才能让人脸识别认识戴口罩的你捏？就需要想想办法。。
## 项目数据
所有数据均采用公开数据集
正常人脸训练数据：VGGFace2，链接：http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/   
正常人脸测试数据：LFW(Labeled Faces in the Wild)，链接：http://vis-www.cs.umass.edu/lfw/   
口罩人脸数据：Real-World-Masked-Face-Dataset，链接：https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset   
## 模型
以标准人脸识别模型FaceNet为主线，添加Attention结构，使其能更好的聚焦于人脸上半部（眼睛四周），以便提高模型精确度。
## 代码咋跑
1. 下载数据(就是下数据啦）：   
建一个Datasets文件夹，把下载的VGGFace2的原始数据(VGGFace2_train文件)、LFW原始数据(lfw_funneled)、LFW配对文件(LFW_pairs.txt)，这仨都放到Datasets文件夹中
2. 清洗小图(原始数据质量不过关，可以把尺寸太小的图删了)：   
使用Data_preprocessing/kill_img.py文件，在preprocess函数中设定图像尺寸大小（现在可能用的是250？也就是只保留边长250以上的图片），而后将data_path指定到VGGFace2_train文件夹（这里是原位操作，所以就直接在这个文件夹里删除所有小图了，如果不想这样的话可以先备份一遍数据。。。），然后运行了就会显示删了多少图、保留了多少图、总图数是多少。大约要1-2小时？
3. 建数据清单(先把所有数据文件信息存到csv文件里以后就不用每次都读了）：   
使用Data_preprocessing/Make_csv_file.py，输入数据文件夹路径、输出csv文件路径，然后就能跑了，保存的csv格式形如（序号，图片名称，人名）。大约要1分钟？
4. 切人脸/对齐/生成mask(数据预处理，用OpenCV把图像中的人脸切下来，做2轴对齐，在根据68人脸特征点生成Attention模块用的mask图)：
使用Data_preprocessing/Image_processing.py，其中的‘shape_predictor_68_face_landmarks.dat’文件需要自己百度下一下，就是OpenCV库的68人脸特征点预测模型，然后定义输入data路径、输出人脸图路径、输出mask图路径，跑就行了。大约要20个小时？
5. 准备训练(确认config文件内容正确)：
resume_path：没有的时候直接写'None'就行了或者瞎写都没啥影响
model：这些是原版模型resnet18\34\50\101\inceptionresnetv2，这个是刚用CBAM的模型resnet34_cbam，这个是更新Attention结构之后的模型resnet34_cheahom，这个是最后想分开训练的时候用的模型resnet34_attention
num_train_triplets：这个是随机生成训练三元组的数量，10万就是10万个三元组共计30万张图
train_triplets_path：这个是随机生成训练三元组的保存路径
test_pairs_paths：这个是随机生成测试三元组的保存路径
6. 模型训练(就是训练)
然后就可以train.py了
7. 检查Attention出来的热图：
使用show_layers.py就行
