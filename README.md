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
