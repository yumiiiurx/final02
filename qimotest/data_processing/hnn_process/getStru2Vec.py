'''

并行分词
'''
import pickle
from python_structured import *
from sqlang_structured import *
from multiprocessing import Pool as ThreadPool

sys.path.append("..")


def multipro_python_query(data_list):

    result = list(map(python_query_parse, data_list))
    return result


def multipro_python_code(data_list):
    result = list(map(python_code_parse, data_list))
    return result
"""
    并行处理Python代码，使用python_code_parse()函数进行解析

    Args:
        data_list (list): 需要处理的数据列表

    Returns:
        list: 处理完成后的结果列表
"""

def multipro_python_context(data_list):
    result = []
    for line in data_list:
        result.append(['-10000'] if line == '-10000' else python_context_parse(line))
    return result


def parse_data_list(data_list):

    result = []
    for line in data_list:
        if line == '-10000':
            # 如果当前元素为'-10000'，则将一个包含单个元素'-10000'的列表添加到结果列表中
            result.append(['-10000'])
        else:
            # 如果当前元素不为'-10000'，则依次执行三种解析操作，并将其结果添加到结果列表中
            result.append(sqlang_context_parse(line))
            result.append(sqlang_query_parse(line))
            result.append(sqlang_code_parse(line))
    return result


def parse_helper(data_list, parse_func, split_num):
    """
    辅助函数：将数据列表分块并使用指定的解析函数对每个块进行解析，返回结果列表。
    :param data_list: 数据列表
    :param parse_func: 解析函数
    :param split_num: 分块大小
    :return: 结果列表
    """
    # 将数据列表分块
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]
    # 使用线程池调用指定的解析函数
    pool = ThreadPool(10)
    result_list = pool.map(parse_func, split_list)
    pool.close()
    pool.join()
    # 将每个块的解析结果合并到一个列表中
    cut_list = []
    for p in result_list:
        cut_list += p
    print('条数：%d' % len(cut_list))
    return cut_list


def parse_python(python_list, split_num):
    """
    :param python_list: 原始数据列表
    :param split_num: 分块大小
    :return: 元组，包含四个子列表和一个列表
    """
    # 获取acont1子列表
    acont1_data = [i[1][0][0] for i in python_list]
    acont1_cut = parse_helper(acont1_data, multipro_python_context, split_num)

    # 获取acont2子列表
    acont2_data = [i[1][1][0] for i in python_list]
    acont2_cut = parse_helper(acont2_data, multipro_python_context, split_num)

    # 获取query子列表
    query_data = [i[3][0] for i in python_list]
    query_cut = parse_helper(query_data, multipro_python_query, split_num)

    # 获取code子列表
    code_data = [i[2][0][0] for i in python_list]
    code_cut = parse_helper(code_data, multipro_python_code, split_num)

    # 获取所有元素的第一个子元素构成的列表
    qids = [i[0] for i in python_list]
    print(qids[0])
    print(len(qids))

    # 返回四个子列表组成的元组，以及原始数据列表中所有元素的第一个子元素构成的列表
    return acont1_cut, acont2_cut, query_cut, code_cut, qids


def parse_sqlang(sqlang_list, split_num):
    acont1_data = [i[1][0][0] for i in sqlang_list]

    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    pool = ThreadPool(10)
    acont1_list = pool.map(multipro_sqlang_context, acont1_split_list)
    pool.close()
    pool.join()
    acont1_cut = []
    for p in acont1_list:
        acont1_cut += p
    print('acont1条数：%d' % len(acont1_cut))

    acont2_data = [i[1][1][0] for i in sqlang_list]

    acont2_split_list = [acont2_data[i:i + split_num] for i in range(0, len(acont2_data), split_num)]
    pool = ThreadPool(10)
    acont2_list = pool.map(multipro_sqlang_context, acont2_split_list)
    pool.close()
    pool.join()
    acont2_cut = []
    for p in acont2_list:
        acont2_cut += p
    print('acont2条数：%d' % len(acont2_cut))

    query_data = [i[3][0] for i in sqlang_list]

    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multipro_sqlang_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[2][0][0] for i in sqlang_list]

    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multipro_sqlang_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))
    qids = [i[0] for i in sqlang_list]

    return acont1_cut, acont2_cut, query_cut, code_cut, qids


def main(lang_type, split_num, source_path, save_path):
    total_data = []
    with open(source_path, "rb") as f:
        #  存储为字典 有序
        corpus_lis = pickle.load(f)  # pickle

        # corpus_lis = eval(f.read()) #txt

        # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, [qcont] 查询上下文, 块长度，标签]

        if lang_type == 'python':

            parse_acont1, parse_acont2, parse_query, parse_code, qids = parse_python(corpus_lis, split_num)
            for i in range(0, len(qids)):
                total_data.append([qids[i], [parse_acont1[i], parse_acont2[i]], [parse_code[i]], parse_query[i]])

        if lang_type == 'sql':

            parse_acont1, parse_acont2, parse_query, parse_code, qids = parse_sqlang(corpus_lis, split_num)
            for i in range(0, len(qids)):
                total_data.append([qids[i], [parse_acont1[i], parse_acont2[i]], [parse_code[i]], parse_query[i]])

    f = open(save_path, "w")
    f.write(str(total_data))
    f.close()


python_type = 'python'
sqlang_type = 'sql'

words_top = 100

split_num = 1000


def test(path1, path2):
    with open(path1, "rb") as f:
        #  存储为字典 有序
        corpus_lis1 = pickle.load(f)  # pickle
    with open(path2, "rb") as f:
        corpus_lis2 = eval(f.read())  # txt

    print(corpus_lis1[10])
    print(corpus_lis2[10])


if __name__ == '__main__':
    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'

    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'

    # main(sqlang_type,split_num,staqc_sql_path,staqc_sql_save)
    # main(python_type, split_num, staqc_python_path, staqc_python_save)

    large_python_path = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'

    large_sql_path = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    # main(sqlang_type, split_num, large_sql_path, large_sql_save)
    main(python_type, split_num, large_python_path, large_python_save)
