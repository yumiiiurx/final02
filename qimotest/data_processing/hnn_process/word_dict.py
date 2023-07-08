import pickle

def load_pickle(filename):
    return pickle.load(open(filename, 'rb'), encoding='iso-8859-1')
#加载pickle文件的函数

def get_vocab(corpus1, corpus2):
    word_vocab = set()
    #处理第一个语料库
    for i in range(len(corpus1)):
        for j in range(len(corpus1[i][1][0])):
            word_vocab.add(corpus1[i][1][0][j])
        for j in range(len(corpus1[i][1][1])):
            word_vocab.add(corpus1[i][1][1][j])
        for j in range(len(corpus1[i][2][0])):
            word_vocab.add(corpus1[i][2][0][j])
        for j in range(len(corpus1[i][3])):
            word_vocab.add(corpus1[i][3][j])
    #处理第二个语料库
    for i in range(len(corpus2)):
        for j in range(len(corpus2[i][1][0])):
            word_vocab.add(corpus2[i][1][0][j])
        for j in range(len(corpus2[i][1][1])):
            word_vocab.add(corpus2[i][1][1][j])
        for j in range(len(corpus2[i][2][0])):
            word_vocab.add(corpus2[i][2][0][j])
        for j in range(len(corpus2[i][3])):
            word_vocab.add(corpus2[i][3][j])

    print(len(word_vocab))
    return word_vocab


def vocab_processing(filepath1, filepath2, save_path):   #构建初步词汇表并保存到文件的函数
    with open(filepath1, 'r') as f:
        total_data1 = eval(f.read())

    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())

    vocab = get_vocab(total_data1, total_data2)

    with open(save_path, "w") as f:
        f.write(str(vocab))


def final_vocab_processing(filepath1, filepath2, save_path):
    #构建最终词汇表并保存到文件的函数
    with open(filepath1, 'r') as f:    #第一个文件路径
        total_data1 = set(eval(f.read()))

    with open(filepath2, 'r') as f:    #第二个文件路径
        total_data2 = eval(f.read())

    vocab = get_vocab(total_data2, total_data2)

    word_set = vocab.difference(total_data1)

    print(len(total_data1))
    print(len(word_set))

    with open(save_path, "w") as f:
        #保存路径
        f.write(str(word_set))


if __name__ == "__main__":
    # ====================获取staqc的词语集合===============
    python_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/python_hnn_data_teacher.txt'
    python_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/python_staqc_data.txt'
    python_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/sql_hnn_data_teacher.txt'
    sql_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/sql_staqc_data.txt'
    sql_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/sql_word_vocab_dict.txt'

    # vocab_processing(python_hnn, python_staqc, python_word_dict)
    # vocab_processing(sql_hnn, sql_staqc, sql_word_dict)
    # ====================获取最后大语料的词语集合的词语集合===============
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabeled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlabeled.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'
    final_vocab_processing(sql_word_dict, new_sql_large, large_word_dict_sql)
    # vocab_processing(new_sql_staqc, new_sql_large, final_word_dict_sql)

    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabeled_data.txt'
    new_python_large ='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlabeled.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    # final_vocab_processing(python_word_dict, new_python_large, large_word_dict_python)
    # vocab_processing(new_python_staqc, new_python_large, final_word_dict_python)


#函数名采用小写字母和下划线的形式，符合Python命名规范。
#修改变量名word_vacab为word_vocab，使其更符合变量的实际含义。
#使用with语句来自动关闭文件。
#修正了函数名的拼写错误。
#采用difference()函数来获取两个集合的差集。



