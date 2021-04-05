## 智能助手的发展
* 在讲问答系统与对话系统之前，先来点背景知识，随着物质生活极大满足，人们对机器智能需求日渐增加，近年来，市面上出现很多智能助手（Apple Siri 阿里小蜜等）、智能音响（天猫精灵 Alexa等），那如何衡量智能助手智能水平了？参考OREILLY对智能助手分级指标，如下图所示
![OREILLY对智能助手分级指标](https://s1.ax1x.com/2020/05/30/tMVGwR.png)

#### Level 1:
* 消息助手（目前手机的消息提醒服务）
#### Level 2: 
* FAQ 助手(一问一答，社区问答InfQA)
#### Level 3: 
* 基于上下文助手（微软小冰 阿里小蜜等）
#### Level 4: 
* 个人助理（微软小冰，目前还达不到这层级，智能音箱）
#### Level 5: 
*高级智能助手（无需交互，自发组织处理事务）

### 智能助手
* 智能助手作为现在移动设备或者桌面设备必不可少应用，在[Medium](https://medium.com/)杂志[ChatbotMagazine](https://chatbotsmagazine.com/)专栏中分别统计了智能助手在应用领域、构建平台、AI工具平台和分析平台四个方面分布，如下图所示
![ChatbotMagazine](https://s1.ax1x.com/2020/05/30/tMnp7j.png)
* 其中很多应用基本都测试过，基本流程式对话设计（无NLU）、基于NLU（意图识别和实体抽取）+对话流程式设计和 基于完整对话pipline流程构建（NLU DM Palicy和NLG）,详细可以参看知乎[问答系统与对话系统](https://zhuanlan.zhihu.com/p/93023782)

## 问答系统与对话系统的概述
* 简答的概述一下问答系统与对话系统相关知识点及方法，其实在之前对问答系统与对话系统还只是比较浅薄理解，真正做智能问答项目后，这里面的水很深,挑战很大
### 问答系统
* 问答系统是人与机器交流比较简单直接方法，是IR（信息检索）系统比较高级形式，相比单纯IR系统，问答系统更加直接回答用户所求。问答系统从数据层面上来。比如Q-Q数据、Q-A数据和Q-D数据，大致分为检索问答、阅读理解问答和基于知识库问答（知识图谱或者结构化知识库）。当然根据答案输出的数据形式分类检索式和生成式
#### 检索式问答系统
* 检索式问答是构建问答系统比较直接简单的方式，类似IR系统，比较容易理解，这类问题映射到NLP领域的问题是啥了？属于那类问题了，这里不谈IR技术，有兴趣同学详细了解[搜索中query理解与应用](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247499470&idx=1&sn=6a6e80673353fb485a854ed2cffc5dcb&chksm=fbd74ca2cca0c5b4fb0622140eb9fa3cd15c860d06b24a4c7fc37e8b3db614af8c4ad9cbe0d3&mpshare=1&scene=24&srcid=0525As77HwaBQIgBKDyZbSUL&sharer_sharetime=1590372296485&sharer_shareid=bb12138cbf7121360054152c6932a462&ascene=14&devicetype=android-29&version=27000e39&nettype=WIFI&abtest_cookie=AAACAA%3D%3D&lang=zh_CN&exportkey=A5L7ER3P75%2FoSgNkw4kLTmg%3D&pass_ticket=7KFoV70a5QT8nW7rQJgYUpRuhEusJtjYE75Syw%2Fh%2BwpN9uRoqgk7PY%2B1KTYYWzbk&wx_header=1)。如何使用户获取查询的知识了，比如FAQ问题，这类问题非常常见，通过做好FAQ知识库，通过召回算法进行query相似度召唤
* 非常明显这类问题属于NLP的匹配问题，可以称之为文本匹配算法。一般是套路是 query ->query理解/解析->召回模型-相似度模型->候选排序->答案，如下图所示
![Image](https://pic4.zhimg.com/80/v2-57e8b8acd12250618369ea73c527ddc8.png)


#### 基于知识图谱的问答系统
* 在讲基于知识图谱的问答系统之前，先问几个问题，何利用现有的结构化数据了做问答或者检索了？，一般企业结构化数据存储在Mysql或者Oracle扥数据库中，数据库查询语言主要是SQL语句，那么对用户的query进行意图理解或者语义解析，再转为SQL语句查询就可以做问答系统或者IR系统了
##### [百度的Text-to-SQL](https://mp.weixin.qq.com/s?__biz=MzUxNzk5MTU3OQ==&mid=2247487028&idx=1&sn=7b6767878b7f6b891fc69e408f248ef1&chksm=f98ef3c0cef97ad6bf7ab1e244131c1084d2bd44e41f7dbd45dfa2bde255822077c2352348e7&scene=21#wechat_redirect)
* 百度Text-to-sql技术专门解决如何与结构化数据进行检索，如下图所示
![text-to-sql](https://mmbiz.qpic.cn/mmbiz_png/uYIC4meJTZ1ibzoXCwJRS7QlR6Utmb5GIbKs1btFoLdpu8f154fwNGcke8pp0QrPf8ZI7CR4icx9jBhx87McxaicA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
#### 阅读理解问答

### 对话系统
* 前面讲的是问答系统，那么可能会问，问答系统与对话系统的之间有什么区别，其实这两者之间是相互相成，比如说对话系统 缩放到单论对话，那就是问答系统


#### 参考文献
* [深度文本匹配在智能客服中的应用](https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/83353437)
* [百度Text-to-SQL 上篇](https://mp.weixin.qq.com/s?__biz=MzUxNzk5MTU3OQ==&mid=2247487028&idx=1&sn=7b6767878b7f6b891fc69e408f248ef1&chksm=f98ef3c0cef97ad6bf7ab1e244131c1084d2bd44e41f7dbd45dfa2bde255822077c2352348e7&scene=21#wechat_redirect)
* [百度Text-to-SQL 下篇](https://mp.weixin.qq.com/s/5lTLW5OOuRMo2zjbzMxr_Q)
* [ROCLING 2019 | 基于深度语义匹配，上下文相关的问答系统](https://mp.weixin.qq.com/s?__biz=MzU2MDkwMzEzNQ==&mid=2247485037&idx=1&sn=9464335e428a07238031e7227631fd52&chksm=fc01a0ddcb7629cb93d5e7a3a738da982aa896b4bc0ad24e737cd55a850852443e226264e4b8&mpshare=1&scene=24&srcid=0509LBUNEA7lcENDMXmCgFKm&sharer_sharetime=1589018672047&sharer_shareid=bb12138cbf7121360054152c6932a462&ascene=14&devicetype=android-29&version=27000e39&nettype=cmnet&abtest_cookie=AAACAA%3D%3D&lang=zh_CN&exportkey=AwHe29u2V1Kt9qmOHcPVT3A%3D&pass_ticket=Q26wC1JiLhUxREQ20ZPdIAnIPyJM1GuD35PGKJCzhGzTXXabOFUwEQEJwrKkCMbn&wx_header=1)
* [PaperWeekly 第37期 | 论文盘点：检索式问答系统的语义匹配模型（神经网络篇）](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247484629&idx=1&sn=20d73aaa6a2ec72a1dd702b3d92cd0ef&chksm=96e9db55a19e52435bad47d97e63f8c74b062a310c29d3e91c78d276756889a16803ca05cfb5&mpshare=1&scene=24&srcid=0415m3WYsko8xLW7SzaISYP0&sharer_sharetime=1586956486078&sharer_shareid=bb12138cbf7121360054152c6932a462&ascene=14&devicetype=android-29&version=27000e39&nettype=cmnet&abtest_cookie=AAACAA%3D%3D&lang=zh_CN&exportkey=Ax24NQGuNjqVPIHhsg8S4dM%3D&pass_ticket=Q26wC1JiLhUxREQ20ZPdIAnIPyJM1GuD35PGKJCzhGzTXXabOFUwEQEJwrKkCMbn&wx_header=1)
* [学术派 | 爱奇艺深度语义表示学习的探索与实践](https://mp.weixin.qq.com/s/YGBvWIENE9TASvb_t_Pebw)