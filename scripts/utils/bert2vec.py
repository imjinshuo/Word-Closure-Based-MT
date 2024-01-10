from transformers import AutoTokenizer, AutoModel, logging
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

logging.set_verbosity_error()
device = torch.device('cuda:0')
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

tokenizer1 = AutoTokenizer.from_pretrained('bert-base-chinese')
model1 = AutoModel.from_pretrained('bert-base-chinese').to(device)

tokenizer2 = AutoTokenizer.from_pretrained('bert-base-cased')
model2 = AutoModel.from_pretrained('bert-base-cased').to(device)


def bert_text_preparation(text, tokenizer):
    tokenized_input = tokenizer(text, is_split_into_words=True)
    tokenized_text = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])
    tokenized_index = tokenized_input.word_ids()
    tokens_tensor = tokenized_input['input_ids']
    segments_tensors = tokenized_input['attention_mask']
    return tokenized_text, tokenized_index, tokens_tensor, segments_tensors


def get_bert_embedding(tokens_tensor, segments_tensors, model):
    with torch.no_grad():
        tokens_tensor = torch.tensor([tokens_tensor], device=device)
        segments_tensors = torch.tensor([segments_tensors], device=device)
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs['last_hidden_state'].cpu()
    token_embeddings = hidden_states[-1]
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]
    return list_token_embeddings

def sim_bert(lan, text1, text2, tokens_idx1, tokens_idx2):
    tokens_idx1.sort()
    tokens_idx2.sort()
    if lan =='zh':
        tokenized_text1, tokenized_index1, tokens_tensor1, segments_tensors1 = bert_text_preparation(text1, tokenizer1)
        tokenized_text2, tokenized_index2, tokens_tensor2, segments_tensors2 = bert_text_preparation(text2, tokenizer1)
        list_token_embeddings1 = get_bert_embedding(tokens_tensor1, segments_tensors1, model1)
        list_token_embeddings2 = get_bert_embedding(tokens_tensor2, segments_tensors2, model1)
        word_indexes1 = [idx for idx, idv in enumerate(tokenized_index1) if idv in tokens_idx1]
        word_indexes2 = [idx for idx, idv in enumerate(tokenized_index2) if idv in tokens_idx2]
        word_embeddings1 = [list_token_embeddings1[this_index] for this_index in word_indexes1]
        word_embeddings2 = [list_token_embeddings2[this_index] for this_index in word_indexes2]
        mean_word_embeddings1 = np.mean(word_embeddings1, axis=0)
        mean_word_embeddings2 = np.mean(word_embeddings2, axis=0)
        this_sim = cosine_similarity(mean_word_embeddings1.reshape(1, -1), mean_word_embeddings2.reshape(1, -1))[0][0]
        return this_sim
    elif lan == 'en':
        tokenized_text1, tokenized_index1, tokens_tensor1, segments_tensors1 = bert_text_preparation(text1, tokenizer2)
        tokenized_text2, tokenized_index2, tokens_tensor2, segments_tensors2 = bert_text_preparation(text2, tokenizer2)
        list_token_embeddings1 = get_bert_embedding(tokens_tensor1, segments_tensors1, model2)
        list_token_embeddings2 = get_bert_embedding(tokens_tensor2, segments_tensors2, model2)
        word_indexes1 = [idx for idx, idv in enumerate(tokenized_index1) if idv in tokens_idx1]
        word_indexes2 = [idx for idx, idv in enumerate(tokenized_index2) if idv in tokens_idx2]
        word_embeddings1 = [list_token_embeddings1[this_index] for this_index in word_indexes1]
        word_embeddings2 = [list_token_embeddings2[this_index] for this_index in word_indexes2]
        mean_word_embeddings1 = np.mean(word_embeddings1, axis=0)
        mean_word_embeddings2 = np.mean(word_embeddings2, axis=0)
        this_sim = cosine_similarity(mean_word_embeddings1.reshape(1, -1), mean_word_embeddings2.reshape(1, -1))[0][0]
        return this_sim
