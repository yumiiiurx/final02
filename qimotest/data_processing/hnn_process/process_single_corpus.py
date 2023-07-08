#修改了函数名，采用小写字母和下划线的命名风格。
#移除了不必要的文件关闭语句，并使用with语句来自动关闭文件。
#修正了函数名的拼写错误。
#使用列表推导式和range(len())来简化代码。

import pickle
from collections import Counter


def load_pickle(filename):
    return pickle.load(open(filename, 'rb'), encoding='iso-8859-1')


def single_list(arr, target):
    return arr.count(target)


# staqc: 把语料中的单候选和多候选分隔开
def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    with open(filepath, 'r') as f:
        total_data = eval(f.read())

    qids = [total_data[i][0][0] for i in range(len(total_data))]
    result = Counter(qids)

    total_data_single = []
    total_data_multiple = []
    for i in range(len(total_data)):
        if result[total_data[i][0][0]] == 1:
            total_data_single.append(total_data[i])
        else:
            total_data_multiple.append(total_data[i])

    with open(save_single_path, 'w') as f:
        f.write(str(total_data_single))

    with open(save_multiple_path, 'w') as f:
        f.write(str(total_data_multiple))


# large: 把语料中的单候选和多候选分隔开
def data_large_processing(filepath, save_single_path, save_multiple_path):
    total_data = load_pickle(filepath)
    qids = [total_data[i][0][0] for i in range(len(total_data))]
    result = Counter(qids)
    total_data_single = []
    total_data_multiple = []
    for i in range(len(total_data)):
        if result[total_data[i][0][0]] == 1:
            total_data_single.append(total_data[i])
        else:
            total_data_multiple.append(total_data[i])

    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)

    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)


# 把单候选只保留其qid
def single_unlabel_to_label(path1, path2):
    total_data = load_pickle(path1)
    labels = [[total_data[i][0], 1] for i in range(len(total_data))]
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))
    with open(path2, "w") as f:
        f.write(str(total_data_sort))


if __name__ == "__main__":
    # 将staqc_python中的单候选和多候选分开
    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = '../hnn_process/ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = '../hnn_process/ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)

    # 将staqc_sql中的单候选和多候选分开
    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = '../hnn_process/ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = '../hnn_process/ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)

    # 将large_python中的单候选和多候选分开
    large_python_path = '../hnn_process/ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = '../hnn_process/ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)

    # 将large_sql中的单候选和多候选分开
    large_sql_path = '../hnn_process/ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = '../hnn_process/ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    large_sql_single_label_save = '../hnn_process/ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = '../hnn_process/ulabel_data/large_corpus/single/python_large_single_label.txt'
    single_unlabel_to_label(large_sql_single_save, large_sql_single_label_save)
    single_unlabel_to_label(large_python_single_save, large_python_single_label_save)




