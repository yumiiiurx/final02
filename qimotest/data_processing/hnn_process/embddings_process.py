'''

从大词典中获取特定于于语料的词典
将数据处理成待打标签的形式
'''

import numpy as np
import pickle
from gensim.models import KeyedVectors


# 词向量文件保存成bin文件
def trans_bin(path1, path2):
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)
    # 如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(path2)
    '''n
    读取用一下代码
    model = KeyedVectors.load(embed_path, mmap='r')
    '''


def get_new_dict(type_vec_path, type_word_path, final_vec_path, final_word_path):
    """
    该函数用于生成词典和词向量，从type_vec_path和type_word_path读取数据，其中type_vec_path中存储了预训练的词向量，
    type_word_path中存储了所有的词标签。在生成词典和词向量时，会为PAD、SOS、EOS和UNK添加对应的向量，并在type_word_path中
    删除这些标签，这四个标签在后续的模型中有特殊用途。生成的词典将通过pickle格式储存在final_word_path，
    而生成的词向量将通过pickle格式储存在final_vec_path。

    Args:
    type_vec_path: str，预训练的词向量路径
    type_word_path: str，所有的词标签路径
    final_vec_path: str，生成的词向量路径
    final_word_path: str，生成的词典路径

    Return:
    None
    """

    # 加载转换文件
    model = KeyedVectors.load(type_vec_path, mmap='r')
    with open(type_word_path, 'r') as f:
        total_word = eval(f.read())
        f.close()

    # 输出词向量
    word_dict = ['PAD', 'SOS', 'EOS', 'UNK']  # 其中0 PAD_ID,1SOS_ID,2E0S_ID,3UNK_ID
    fail_word = []

    rng = np.random.RandomState(None)
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()
    unk_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    sos_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    eos_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()

    word_vectors = [pad_embedding, sos_embediing, eos_embediing, unk_embediing]

    for word in total_word:
        try:
            word_vectors.append(model.wv[word])  # 加载词向量
            word_dict.append(word)
        except:
            print(word)
            fail_word.append(word)

    word_vectors = np.array(word_vectors)
    word_dict = dict(map(reversed, enumerate(word_dict)))

    with open(final_vec_path, 'wb') as file:
        # 通过pickle格式储存词向量
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        # 通过pickle格式储存词典
        pickle.dump(word_dict, file)

    # 打印存储的信息
    print(f"Total words in type_word_path: {len(total_word)}")
    print(f"Words that successfully find vectors: {len(word_dict) - 4}")
    print(f"Words that fail to find vectors: {len(fail_word)}")
    print(f"Total words in the dictionary: {len(word_dict)}")
    print(f"Length of the word vectors: {len(word_vectors)}")
    print("完成")


# 得到词在词典中的位置
def get_index(type, text, word_dict):
    location = [1] if type == 'code' else [0]
    if type == 'code':
        len_c = len(text)
        if len_c + 1 < 350:
            for i in range(len_c):
                index = word_dict.get(text[i], word_dict.get('UNK'))
                location.append(index)
        else:
            for i in range(348):
                index = word_dict.get(text[i], word_dict.get('UNK'))
                location.append(index)
        location.append(2)
    else:
        if len(text) > 0 and text[0] != '-10000':
            for i in range(len(text)):
                index = word_dict.get(text[i], word_dict.get('UNK'))
                location.append(index)

    return location


# 将文本中所有词语转化为它在词典中的位置
def get_index(text_type, text, word_dict):
    tokenizer = word_dict['tokenizer']
    if text_type == 'code':
        max_len = 350
    else:
        max_len = 25
    tokenized_text = [tokenizer.convert_tokens_to_ids(text.split()[i]) if i < max_len else 0 for i in
                      range(len(text.split()))]
    return tokenized_text


# 序列化处理语料
def serialize_corpus(word_dict_path, type_path, final_type_path):
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)

    with open(type_path, 'r') as f:
        corpus = eval(f.read())

    total_data = []

    for ques in corpus:
        qid, Si, Si1, code, query, block_length, label = ques
        Si_word_list = get_index('text', Si[0], word_dict)
        Si1_word_list = get_index('text', Si1[0], word_dict)
        query_word_list = get_index('text', query, word_dict)

        # 处理Si
        Si_word_list = Si_word_list[:100] + [0] * (100 - len(Si_word_list)) if len(Si_word_list) <= 100 else Si_word_list[:100]
        # 处理Si1
        Si1_word_list = Si1_word_list[:100] + [0] * (100 - len(Si1_word_list)) if len(Si1_word_list) <= 100 else Si1_word_list[:100]
        # 处理code
        code = code[:350] + [0] * (350 - len(code)) if len(code) >= 350 else code + [0] * (350 - len(code))
        # 处理query
        query_word_list = query_word_list[:25] + [0] * (25 - len(query_word_list)) if len(query_word_list) <= 25 else query_word_list[:25]

        one_data = [qid, [Si_word_list, Si1_word_list], [code], query_word_list, block_length, label]
        total_data.append(one_data)

    with open(final_type_path, 'wb') as file:
        pickle.dump(total_data, file)


def get_new_dict_append(type_vec_path, previous_dict, previous_vec, append_word_path, final_vec_path,
                        final_word_path):
    '''
    扩充词向量和词典，并保存为新的文件
    type_vec_path: 原始词向量文件路径
    previous_dict: 原始词典文件路径
    previous_vec: 原始词向量文件路径
    append_word_path: 扩充的词文件路径
    final_vec_path: 保存的新词向量文件路径
    final_word_path: 保存的新词典文件路径
    '''
    model = KeyedVectors.load(type_vec_path, mmap='r')  # 加载词向量模型

    with open(previous_dict, 'rb') as f:
        pre_word_dict = pickle.load(f)  # 加载原始词典

    with open(previous_vec, 'rb') as f:
        pre_word_vec = pickle.load(f)  # 加载原始词向量

    with open(append_word_path, 'r') as f:
        append_word = eval(f.read())  # 加载需要扩充的词

    # 输出原始词典信息
    print(type(pre_word_vec))
    word_dict = list(pre_word_dict.keys())
    print(len(word_dict))
    word_vectors = pre_word_vec.tolist()
    print(word_dict[:100])

    fail_word = []  # 存放无法找到的词
    print(len(append_word))
    rng = np.random.RandomState(None)
    unk_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    h = []

    for word in append_word:
        try:
            word_vectors.append(model.wv[word])
            word_dict.append(word)
        except:
            fail_word.append(word)

    print(len(word_dict))  # 扩充后的词典大小
    print(len(word_vectors))  # 扩充后的词向量大小
    print(len(fail_word))  # 扩充失败的词数
    print(word_dict[:100])

    word_vectors = np.array(word_vectors)
    word_dict = dict(map(reversed, enumerate(word_dict)))

    # 保存新词典和新词向量
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

    print("完成")


# -------------------------参数配置----------------------------------
# python 词典 ：1121543 300
if __name__ == '__main__':
    ps_path = '../hnn_process/embeddings/10_10/python_struc2vec1/data/python_struc2vec.txt'  # 239s
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin'  # 2s

    sql_path = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.txt'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

    # trans_bin(sql_path,sql_path_bin)
    # trans_bin(ps_path, ps_path_bin)
    # 113440 27970(2) 49409(12),50226(30),55993(98)

    # ==========================  ==========最初基于Staqc的词典和词向量==========================

    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'

    # txt存储数组向量，读取时间：30s,以pickle文件存储0.23s,所以最后采用pkl文件

    # get_new_dict(ps_path_bin,python_word_path,python_word_vec_path,python_word_dict_path)
    # get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # =======================================最后打标签的语料========================================
    # sql 待处理语料地址
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    # sql大语料最后的词典
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'

    # sql最后的词典和对应的词向量
    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sql_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'
    # get_new_dict(sql_path_bin, final_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)
    # get_new_dict_append(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql,
    # sql_final_word_vec_path,sql_final_word_dict_path)

    staqc_sql_f = '../hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    large_sql_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'
    # Serialization(sql_final_word_dict_path, new_sql_staqc, staqc_sql_f)
    # Serialization(sql_final_word_dict_path, new_sql_large, large_sql_f)

    # python
    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'

    # python最后的词典和对应的词向量
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'

    # get_new_dict(ps_path_bin, final_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)
    # get_new_dict_append(ps_path_bin, python_word_dict_path, python_word_vec_path, large_word_dict_python,
    # python_final_word_vec_path,python_final_word_dict_path)

    # 处理成打标签的形式
    staqc_python_f = '../hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'
    large_python_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl'
    # Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)
    Serialization(python_final_word_dict_path, new_python_large, large_python_f)

    print('序列化完毕')
    # test2(test_python1,test_python2,python_final_word_dict_path,python_final_word_vec_path)