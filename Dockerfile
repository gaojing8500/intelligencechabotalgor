##打包镜像
FROM docker-nvidia-gpu-ubuntu18.04-base:v1.1
MAINTAINER gaojing <gaojing850063636@sina.com>                                                                             
ADD IntelligenceAlgor /root/IntelligenceAlgor
ARG workdir=/root/IntelligenceAlgor/service/
WORKDIR ${workdir}
#CMD ["python","/root/textmatch/text_macth_algor_service.py"]
ENTRYPOINT ["python3", "chatbot_algor_service.py"]

                                                              
 