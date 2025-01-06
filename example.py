from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

pretrained_weights = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
model = RobertaModel.from_pretrained(pretrained_weights)
sentences = [
"Adversaries may use Valid Accounts to interact with remote systems using Windows Remote Management (WinRM). The adversary may then perform actions as the logged-on user.",
"Comparing client-server request and response payloads to a baseline profile to identify outliers.",
"Analyzing failed connections in a network to detect unauthorized activity.",
"Establishing baseline communities of network hosts and identifying statistically divergent inter-community communication.",
"Detecting anomalies that indicate malicious activity by comparing the amount of data downloaded versus data uploaded by a host.",
"Collecting network communication protocol metadata and identifying statistical outliers.",
"Detection of an unauthorized remote live terminal console session by examining network traffic to a network host.",
"Monitoring geolocation data of user logon attempts and comparing it to a baseline user behavior profile to identify anomalies in logon location.",
"Restricting network traffic originating from any location.",
"Asset vulnerability enumeration enriches inventory items with knowledge identifying their vulnerabilities."
]

# 初始化字典来存储
tokens = {'input_ids': [], 'attention_mask': []}
vec = []
for sentence in sentences:
    # 编码每个句子并添加到字典
    new_tokens = tokenizer.encode_plus(sentence, max_length=15, truncation=True, padding='max_length', return_tensors='pt')
    # print(new_tokens['input_ids'][0])
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])


# 将张量列表重新格式化为一个张量
tokens['input_ids'] = torch.stack(tokens['input_ids'])
tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

outputs = model(**tokens)
# print(outputs.keys()) #odict_keys(['last_hidden_state', 'pooler_output'])

embeddings = outputs.last_hidden_state
# print(embeddings.shape) #torch.Size([4, 15, 768])

attention_mask = tokens['attention_mask']
# print(attention_mask.shape) #torch.Size([4, 15])

mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
# print(mask.shape) #torch.Size([4, 15, 768])

masked_embeddings = embeddings * mask
# print(masked_embeddings.shape) #torch.Size([4, 15, 768])

summed = torch.sum(masked_embeddings, 1)
# print(summed.shape) #torch.Size([4, 768])

summed_mask = torch.clamp(mask.sum(1), min=1e-9)
# print(summed_mask.shape) #torch.Size([4, 768])

mean_pooled = summed / summed_mask
# print(mean_pooled.shape) #torch.Size([4, 768])

mean_pooled = mean_pooled.detach().numpy()

result = cosine_similarity([mean_pooled[0]], mean_pooled[1:])
for r in result[0]:
    print(r)

print('----', max(result[0]))