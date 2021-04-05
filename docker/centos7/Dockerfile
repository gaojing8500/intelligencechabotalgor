##制作基础镜像,加载requirments.txt文件安装python依赖包  cuda10.0+cudnn7.5+tensorflow-gpu-1.13+python3.6+centos7
FROM nvidia/cuda:10.0-cudnn7-runtime-centos7
MAINTAINER gaojing <gaojing850063636@sina.com>                                                                             
ADD requirements.txt /root/requirements.txt
##使用yum命令安装或者直接源码安装
RUN yum update
RUN yum install -y python36-pip
RUN pip3 install --upgrade pip 
RUN pip install -r /root/requirements.txt
