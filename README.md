# fine-tune-ner
# find a model in models
要对这个BERT模型进行微调首先就要找到一个合适的初始模型。那么就可以先使用一个较小的数据集进行测试。
这里首先使用的是24条对话，对所有模型进行一个简单的评估。

安装相应的环境：
```
pip install -r requirements.txt
```

配置config.yaml文件
```
vim config.yaml
```

首先使用脚本把对话内容中的角色都提取出来：
```
python write_names_to_data.py
```
编辑输出的相应角色名称文件，把其中不是advisor的名字删除掉，得到一个专门的advisor列表


获得evaluation使用的数据集：
```
python order_extract.py
```

开始进行evaluation：
```
python eval_models_v1.py
```

相应的badcases以及precision的log存储在：
```
badceses/   log/
```



