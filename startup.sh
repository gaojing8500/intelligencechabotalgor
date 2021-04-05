#!/bin/bash
echo "docker constainer start up "
#判断容器是否已经启动
if nvidia-docker ps -a | grep intelligence_chatbot_service_v1.1
then 
    echo "现有容器启动，先删除现有容器再重新启动"
    nvidia-docker rm -f intelligence_chatbot_service_v1.1
    nvidia-docker run -d -p 0.0.0.0:9890:9890 --name intelligence_chatbot_service_v1.1 -v /home/gaojing/IntelligenceAlgor/models:/root/models registry.uih/com.uih.uplus/chatbot_algor_service:v1.1 /bin/bash
    
else
    echo "没有现有容器，启动新的容器"
    nvidia-docker run -d -p 0.0.0.0:9890:9890 --name intelligence_chatbot_service_v1.1 -v /home/gaojing/IntelligenceAlgor/models:/root/models registry.uih/com.uih.uplus/chatbot_algor_service:v1.1 /bin/bash
fi