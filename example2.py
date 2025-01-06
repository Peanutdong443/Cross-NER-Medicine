from fastNLP import Vocabulary
from fastNLP import DataSet

# 创建示例数据集
dataset = DataSet({'lattice': ['我', '喜欢', '自然语言处理', '。', '你', '呢', '？']})
# 创建lattice_vocab对象
lattice_vocab = Vocabulary()
# 使用from_dataset方法构建词汇表
lattice_vocab.from_dataset(dataset, field_name='lattice')

# 查看词汇表中的所有标记
print("All lattices in the vocabulary:", lattice_vocab)
# 使用index_dataset函数将'lattice'字段的文本数据转换为索引，并将结果存储到新字段'lattice'
lattice_vocab.index_dataset(dataset, field_name='lattice', new_field_name='lattice_indices')
# 查看词汇表的大小
print("Vocabulary size:", len(lattice_vocab))
# 查看转换后的数据集
print(dataset)
import torch

cid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# 使用列表推导式创建PyTorch张量，并将数据类型更改为int64
tensor_list = [torch.tensor([int(x) for x in sublist]).to(torch.int64) for sublist in cid]
print(tensor_list)
# 打印结果

2222222222222