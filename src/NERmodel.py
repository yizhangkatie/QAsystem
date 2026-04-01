from torch import nn
import os
import pickle
from transformers import BertModel, BertTokenizer
import ahocorasick
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch


class rule_find:
    def __init__(self):
        self.idx2type = idx2type = [
            "疾病",
            "人群",
            "食物",
            "食物分组",
            "药品",
            "药物成分",
            "药物剂型",
            "药物性味",
            "证候",
            "症状",
            "中药功效",
            "疾病分组",
            "药品分组",
        ]
        self.type2idx = type2idx = {
            "疾病": 0,
            "人群": 1,
            "食物": 2,
            "食物分组": 3,
            "药品": 4,
            "药物成分": 5,
            "药物剂型": 6,
            "药物性味": 7,
            "证候": 8,
            "症状": 9,
            "中药功效": 10,
            "疾病分组": 11,
            "药品分组": 12,
        }
        self.ahos = [ahocorasick.Automaton() for i in range(len(self.type2idx))]

        for type in idx2type:
            with open(
                os.path.join('../data', 'ent_aug', f'{type}.txt'), encoding='utf-8'
            ) as f:
                all_en = f.read().split('\n')
            for en in all_en:
                en = en.split(' ')[0]
                if len(en) >= 2:
                    self.ahos[type2idx[type]].add_word(en, en)
        for i in range(len(self.ahos)):
            self.ahos[i].make_automaton()

    def find(self, sen):
        rule_result = []
        mp = {}
        all_res = []
        all_ty = []
        for i in range(len(self.ahos)):
            now = list(self.ahos[i].iter(sen))
            all_res.extend(now)
            for j in range(len(now)):
                all_ty.append(self.idx2type[i])
        if len(all_res) != 0:
            all_res = sorted(all_res, key=lambda x: len(x[1]), reverse=True)
            for i, res in enumerate(all_res):
                be = res[0] - len(res[1]) + 1
                ed = res[0]
                if be in mp or ed in mp:
                    continue
                rule_result.append((be, ed, all_ty[i], res[1]))
                for t in range(be, ed + 1):
                    mp[t] = 1
        return rule_result


class tfidf_alignment:
    def __init__(self):
        eneities_path = os.path.join('../data', 'ent_aug')
        files = os.listdir(eneities_path)
        files = [docu for docu in files if '.txt' in docu]

        self.tag_2_embs = {}
        self.tag_2_tfidf_model = {}
        self.tag_2_entity = {}
        for ty in files:
            with open(os.path.join(eneities_path, ty), 'r', encoding='utf-8') as f:
                entities = f.read().split('\n')
                entities = [
                    ent
                    for ent in entities
                    if len(ent.split(' ')[0]) <= 15 and len(ent.split(' ')[0]) >= 1
                ]
                en_name = [ent.split(' ')[0] for ent in entities]
                ty = ty.strip('.txt')
                self.tag_2_entity[ty] = en_name
                tfidf_model = TfidfVectorizer(analyzer="char")
                embs = tfidf_model.fit_transform(en_name).toarray()
                self.tag_2_embs[ty] = embs
                self.tag_2_tfidf_model[ty] = tfidf_model

    def align(self, ent_list):
        new_result = {}
        for s, e, cls, ent in ent_list:
            ent_emb = self.tag_2_tfidf_model[cls].transform([ent])
            sim_score = cosine_similarity(ent_emb, self.tag_2_embs[cls])
            max_idx = sim_score[0].argmax()
            max_score = sim_score[0][max_idx]

            if max_score >= 0.5:
                new_result[cls] = self.tag_2_entity[cls][max_idx]
        return new_result


class Bert_Model(nn.Module):
    def __init__(self, model_name, hidden_size, tag_num, bi):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.gru = nn.RNN(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=bi,
        )
        if bi:
            self.classifier = nn.Linear(hidden_size * 2, tag_num)
        else:
            self.classifier = nn.Linear(hidden_size, tag_num)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, label=None):
        bert_0, _ = self.bert(x, attention_mask=(x > 0), return_dict=False)
        gru_0, _ = self.gru(bert_0)
        pre = self.classifier(gru_0)
        if label is not None:
            loss = self.loss_fn(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1).squeeze(0)


def get_data(path, max_len=None):
    all_text, all_tag = [], []
    with open(path, 'r', encoding='utf8') as f:
        all_data = f.read().split('\n')

    sen, tag = [], []
    for data in all_data:
        data = data.split(' ')
        if len(data) != 2:
            if len(sen) > 2:
                all_text.append(sen)
                all_tag.append(tag)
            sen, tag = [], []
            continue
        te, ta = data
        sen.append(te)
        tag.append(ta)
    if max_len is not None:
        return all_text[:max_len], all_tag[:max_len]
    return all_text, all_tag


# 找出tag(label)中的所有实体及其下表，为实体动态替换/随机掩码策略/实体动态拼接做准备
def find_entities(tag):
    result = []
    label_len = len(tag)
    i = 0
    while i < label_len:
        if tag[i][0] == 'B':
            type = tag[i].strip('B-')
            j = i + 1
            while j < label_len and tag[j][0] == 'I':
                j += 1
            result.append((i, j - 1, type))
            i = j
        else:
            i = i + 1
    return result


def build_tag2idx(all_tag):
    tag2idx = {'<PAD>': 0}
    for sen in all_tag:
        for tag in sen:
            tag2idx[tag] = tag2idx.get(tag, len(tag2idx))
    return tag2idx


def merge(model_result_word, rule_result):
    result = model_result_word + rule_result
    result = sorted(result, key=lambda x: len(x[-1]), reverse=True)
    check_result = []
    mp = {}
    for res in result:
        if res[0] in mp or res[1] in mp:
            continue
        check_result.append(res)
        for i in range(res[0], res[1] + 1):
            mp[i] = 1
    return check_result


def get_ner_result(model, tokenizer, sen, rule, tfidf_r, device, idx2tag):
    sen_to = tokenizer.encode(sen, add_special_tokens=True, return_tensors='pt').to(
        device
    )
    pre = model(sen_to).tolist()

    pre_tag = [idx2tag[i] for i in pre[1:-1]]
    model_result = find_entities(pre_tag)
    model_result_word = []
    for res in model_result:
        word = sen[res[0] : res[1] + 1]
        model_result_word.append((res[0], res[1], res[2], word))
    rule_result = rule.find(sen)

    merge_result = merge(model_result_word, rule_result)
    tfidf_result = tfidf_r.align(merge_result)
    return tfidf_result


def run(sen):
    all_text, all_label = get_data("../data/ner_data_aug.txt")
    if os.path.exists('../tmp_data/tag2idx.pkl'):
        with open('../tmp_data/tag2idx.pkl', 'rb') as f:
            tag2idx = pickle.load(f)
    else:
        tag2idx = build_tag2idx(all_label)
        with open('tmp_data/tag2idx.pkl', 'wb') as f:
            pickle.dump(tag2idx, f)

    idx2tag = list(tag2idx)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    hidden_size = 128
    bi = True
    model_name = '../model'  # bert_base_chinese
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = Bert_Model(model_name, hidden_size, len(tag2idx), bi)
    model = model.to(device)
    rule = rule_find()
    tfidf_r = tfidf_alignment()
    return get_ner_result(model, tokenizer, sen, rule, tfidf_r, device, idx2tag)


# if __name__ == "__main__":
#     while True:
#         a = input()
#         print(run(a))
