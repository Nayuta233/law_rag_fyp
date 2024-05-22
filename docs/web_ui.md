# Web UI 运行说明

## 1. 初始化

输入config文件地址与选择[milvus|pipeline]方案，如果对中国法律进行构建，请选择Chinese，如果对新加坡法律进行构建，请选择English。选择好后，点击第一个`initilize`，第一次将下载模型，可能速度较慢，完成后，会显示`CLI initilized`

## 2. 构建索引

在选择命令中选择`Build up context`，然后在问题中选择要构建的语料，例如 `./data/Chinese_law/criminal_general_provisions.txt` 会将《中华人民共和国刑法·总则》导入。可以在控制台看到embedding生成进度。 在问题中输入点击`submit`，注意，中国法律语料库较大。如果选择 `./data/Chinese_law/` 会将该目录下所有文件进行索引构建，耗费时间较长，针对大规模语料库建议使用“Zilliz Cloud Pipelines方案”。

## 3. 输入问题

在选择命令中选择`ask`或者`ask+return retrieved context`，然后在问题中输入正式问题，点击`submit`，即可得到回答。