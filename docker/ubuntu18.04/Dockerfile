##制作基础镜像,加载requirments.txt文件安装python依赖包  cuda10.0+cudnn7.5+tensorflow-gpu-1.13+python3.6+ubuntu18.04
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
MAINTAINER gaojing <gaojing850063636@sina.com>                                                                             
##使用apt-get命令安装或者直接源码安装
RUN apt-get -y update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip  
ADD requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt 
