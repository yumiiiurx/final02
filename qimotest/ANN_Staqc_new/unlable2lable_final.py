from __future__ import print_function
# 给语料打标签
import os
import tensorflow as tf
import numpy as np
import random
import pickle


random.seed(42)
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
from sklearn.metrics import *
from configs import *
import warnings

warnings.filterwarnings("ignore")

set_session = tf.compat.v1.keras.backend.set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # half of the memory
set_session(tf.compat.v1.Session(config=config))

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class StandoneCode:
    # dict.get(）：返回指定键的值，如果键不在字典中返回默认值 None 或者设置的默认值
    def __init__(self, conf=None):
        self.conf = dict() if conf is None else conf
        self._buckets = conf.get('buckets', [(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)])
        self._buckets_text_max = (max([i for i, _, _, _ in self._buckets]), max([j for _, j, _, _ in self._buckets]))
        self._buckets_code_max = (max([i for _, _, i, _ in self._buckets]), max([j for _, _, _, j in self._buckets]))
        self.path = self.conf.get('workdir', './data/')
        self.train_params = conf.get('training_params', dict())
        self.data_params = conf.get('data_params', dict())
        self.model_params = conf.get('model_params', dict())
        self._eval_sets = None

    def load_pickle(self, filename):
        with open(filename, 'rb') as f:
            word_dict = pickle.load(f)
        return word_dict

        ##### Data Set #####

    ##### Padding #####
    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Model Loading / saving #####
    def save_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
        if not os.path.exists(self.path + 'models/' + self.model_params['model_name'] + '/'):
            os.makedirs(self.path + 'models/' + self.model_params['model_name'] + '/')
        model.save("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path,
                                                                                                  self.model_params[
                                                                                                      'model_name'],
                                                                                                  d12, d3, d4, d5, r,
                                                                                                  epoch),
                   overwrite=True)

    import os

    class ModelLoader:
        def load_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
            """
            加载特定epoch的模型。

            参数:
                model: 要加载权重的模型对象。
                epoch (int): epoch号。
                d12, d3, d4, d5, r: 文件名中使用的参数。
            """
            # 打印模型检查点存储路径
            print(self.path)

            # 根据提供的参数和epoch号构建文件名
            filename = "{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
                self.path, d12, d3, d4, d5, r, epoch)

            # 检查文件是否存在
            assert os.path.exists(filename), "未找到第 {:d} 个epoch的权重".format(epoch)

            # 将权重加载到模型中
            model.load(filename)

        def del_pre_model(self, prepoch, d12, d3, d4, d5, r):
            """
            删除上一个模型检查点文件。

            参数:
                prepoch: 之前epoch号的列表。
                d12, d3, d4, d5, r: 文件名中使用的参数。
            """
            # 检查是否至少有两个之前的epoch号
            if len(prepoch) >= 2:
                # 获取倒数第二个epoch号
                epoch = prepoch[-2]

                # 根据提供的参数和epoch号构建文件名
                filename = "{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
                    self.path, self.model_params['model_name'], d12, d3, d4, d5, r, epoch)

                # 检查文件是否存在
                if os.path.exists(filename):
                    # 删除文件
                    os.remove(filename)

    import numpy as np

    class DataProcessor:
        def process_instance(self, instance, target, maxlen):
            """
            处理单个实例并将其添加到目标列表中。

            参数：
                instance: 要处理的实例。
                target: 目标列表。
                maxlen: 最大长度。
            """
            w = self.pad(instance, maxlen)
            target.append(w)

        def process_matrix(self, inputs, trans1_length, maxlen):
            """
            处理输入矩阵。

            参数：
                inputs: 输入矩阵。
                trans1_length: 转换维度1的长度。
                maxlen: 最大长度。

            返回：
                处理后的输入列表。
            """
            inputs_trans1 = np.split(inputs, trans1_length, axis=1)
            processed_inputs = []
            for item in inputs_trans1:
                item_trans2 = np.squeeze(item, axis=1).tolist()
                processed_inputs.append(item_trans2)
            return processed_inputs

        def get_data(self, path):
            """
            获取数据。

            参数：
                path: 数据路径。

            返回：
                包含文本、代码、查询、标签和ID的元组。
            """
            data = self.load_pickle(path)

            text_S1 = []
            text_S2 = []
            code = []
            queries = []
            labels = []
            ids = []

            text_block_length, text_word_length, query_word_length, code_token_length = 2, 100, 25, 350

            # 处理文本块
            text_blocks = self.process_matrix(np.array([samples_term[1] for samples_term in data]),
                                              text_block_length, 100)
            text_S1 = text_blocks[0]
            text_S2 = text_blocks[1]

            # 处理代码块
            code_blocks = self.process_matrix(np.array([samples_term[2] for samples_term in data]),
                                              text_block_length - 1, 350)
            code = code_blocks[0]

            queries = [samples_term[3] for samples_term in data]
            labels = [samples_term[5] for samples_term in data]
            ids = [samples_term[0] for samples_term in data]

            return text_S1, text_S2, code, queries, labels, ids

    import numpy as np

    class Evaluation:
        def eval(self, model, path):
            """
            在验证集上评估模型性能。

            参数：
                model: 要评估的模型。
                path: 数据路径。

            返回：
                准确率、F1值、召回率、精确度和损失。
            """
            text_S1, text_S2, code, queries, labels, ids = self.get_data(path)

            # 预测标签
            labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                                      batch_size=100)
            labelpred = np.argmax(labelpred, axis=1)

            # 计算损失和指标
            loss = log_loss(labels, labelpred)
            acc = accuracy_score(labels, labelpred)
            f1 = f1_score(labels, labelpred)
            recall = recall_score(labels, labelpred)
            precision = precision_score(labels, labelpred)
            print("测试性能：准确率=%.3f，召回率=%.3f，F1=%.3f，准确度=%.3f" % (
                precision, recall, f1, acc))
            return acc, f1, recall, precision, loss

        def u2l_codemf(self, model, path, save_path):
            """
            为语料打上codemf标签。

            参数：
                model: 模型用于打标签。
                path: 数据路径。
                save_path: 保存标签的路径。
            """
            total_label = []
            text_S1, text_S2, code, queries, labels, ids1 = self.get_data(path)

            # 预测标签
            labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                                      batch_size=100)
            labelpred1 = np.argmax(labelpred, axis=1)

            total_label.append(ids1)
            total_label.append(labelpred1.tolist())

            # 保存标签到文件
            with open(save_path, "w") as f:
                f.write(str(total_label))
            print("已为语料打上codemf标签")

        def u2l_textsa(self, model, path, save_path):
            """
            为语料打上textsa标签。

            参数：
                model: 模型用于打标签。
                path: 数据路径。
                save_path: 保存标签的路径。
            """
            with open(save_path, 'r') as f:
                pre = eval(f.read())

            my_pre1 = pre[1]  # codemf_label
            total_label = []
            text_S1, text_S2, code, queries, labels, ids1 = self.get_data(path)

            # 预测标签
            labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                                      batch_size=100)
            labelpred1 = np.argmax(labelpred, axis=1)

            total_label.append(ids1)
            total_label.append(my_pre1)
            total_label.append(labelpred1.tolist())

            # 保存标签到文件
            with open(save_path, "w") as f:
                f.write(str(total_label))
            print("已为语料打上textsa标签")

    # 给语料打codesa标签
    def u2l_codesa(self, model, path, save_path):

        with open(save_path, 'r') as f:
            pre = eval(f.read())
        f.close()

        my_pre1 = pre[1]  # codemf_label
        my_pre2 = pre[2]  # textsa_label

        total_label = []
        text_S1, text_S2, code, queries, labels, ids1 = self.get_data(path)
        labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                                  batch_size=100)
        labelpred1 = np.argmax(labelpred, axis=1)

        total_label.append(ids1)
        total_label.append(my_pre1)
        total_label.append(my_pre2)
        total_label.append(labelpred1.tolist())
        f = open(save_path, "w")
        f.write(str(total_label))
        f.close()
        print("codesa标签已打完")


# 分析组合不同模型打标签的结果
'''
这一步是已经确定了选择text_sa与code_sa中的模型，与codemf模型标签节后进行最后的标签过滤
'''

import argparse


def final_analay(path, hnn_path, save_path):
    """
    分析标签并生成最终结果。

    参数：
        path: 原始标签文件路径。
        hnn_path: HNN标签文件路径。
        save_path: 保存最终结果的路径。
    """
    with open(path, 'r') as f:
        pre = eval(f.read())
    ids = pre[0]
    codemf_lable = pre[1]
    textsa_lable = pre[2]
    codesa_lable = pre[3]
    hnn_lable_1 = []
    with open(hnn_path, 'r') as f:
        hnn = eval(f.read())
    for i in range(0, len(hnn[0])):
        if (hnn[1][i] == 1):
            hnn_lable_1.append(hnn[0][i])

    total_final = []
    count = 0
    for i in range(0, len(ids)):
        if (codesa_lable[i] == 1 and textsa_lable[i] == 1 and codemf_lable[i] == 1):
            if ids[i] in hnn[0]:
                continue
            else:
                total_final.append(ids[i])
                count += 1
    total_final = total_final + hnn_lable_1
    with open(save_path, "w") as f:
        for i in range(0, len(total_final)):
            f.writelines(str(total_final[i]))
            f.writelines('\n')


def final_analay_large(path, hnn_path, single_path, save_path):
    """
    分析标签并生成最终结果（包含单个文件）。

    参数：
        path: 原始标签文件路径。
        hnn_path: HNN标签文件路径。
        single_path: 单个文件路径。
        save_path: 保存最终结果的路径。
    """
    with open(path, 'r') as f:
        pre = eval(f.read())
    ids = pre[0]
    codemf_lable = pre[1]
    textsa_lable = pre[2]
    codesa_lable = pre[3]
    hnn_lable_1 = []
    with open(hnn_path, 'r') as f:
        hnn = eval(f.read())
    for i in range(0, len(hnn[0])):
        if (hnn[1][i] == 1):
            hnn_lable_1.append(hnn[0][i])

    total_final = []
    count = 0
    for i in range(0, len(ids)):
        if (codesa_lable[i] == 1 and textsa_lable[i] == 1 and codemf_lable[i] == 1):
            if ids[i] in hnn[0]:
                continue
            else:
                total_final.append(ids[i])
                count += 1
    with open(single_path, 'r') as f:
        single = eval(f.read())
    single_ids = []
    for i in range(0, len(single)):
        single_ids.append(single[i][0])
    total_final = total_final + hnn_lable_1
    with open(save_path, "w") as f:
        for i in range(0, len(total_final)):
            f.writelines(str(total_final[i]))
            f.writelines('\n')


def parse_args():
    """
    解析命令行参数。

    返回：
        包含解析后参数的对象。
    """
    parser = argparse.ArgumentParser("Train and Test Model")
    parser.add_argument("--train", choices=["python", "sql"], default="sql", help="train dataset set")
    parser.add_argument("--mode", choices=["train", "eval"], default='eval',
                        help="The mode to run. The `train` mode trains a model;"
                             " the `eval` mode evaluat models in a test set ")
    parser.add_argument("--verbose", action="store_true", default=True, help="Be verbose")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    conf = get_config_u2l(args.train)
    train_path = conf['data_params']['train_path']
    dev_path = conf['data_params']['valid_path']
    test_path = conf['data_params']['test_path']
    embding = conf['data_params']['code_pretrain_emb_path']

    ##### Define model ######
    logger.info('Build Model')

    model = eval(conf['model_params']['model_name'])(
        conf)  # initialize the model,  model== <models.CARLCS_CNN object at 0x7f1d9c2e2cc0>
    StandoneCode = StandoneCode(conf)

    # ====================================sql打标签====================================
    # 无标签的地址--包括单后选和多候选
    staqc_sql_f = '../data_processing/hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    large_sql_f = '../data_processing/hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'
    large_single_path = '../data_processing/hnn_process/ulabel_data/large_corpus/single/sql_large_single_label.txt'
    # ---------有标签的地址----------
    hnn_lable_sql_path = '../data_processing/hnn_process/ulabel_data/staqc/hnn_label_sql.txt'
    # staqc:存放only-code、only-text、codemf标签地址
    staqc_sql_final_label = '../data_processing/hnn_process/ulabel_data/staqc/sql_final_label.txt'
    # staqc:利用only-code、only-text、codemf中都为1筛选后的语料地址
    save_path_final_lable_staqc_sql = '../data_processing/hnn_process/ulabel_data/staqc/combine_final_sql_lable.txt'

    # large:存放only-code、only-text、codemf标签地址
    large_sql_fianl_lable = '../data_processing/hnn_process/ulabel_data/staqc/large_sql_final_label.txt'
    # large:利用only-code、only-text、codemf中都为1筛选后的语料地址
    save_path_final_lable_large_sql_mul = '../data_processing/hnn_process/ulabel_data/staqc/combine_codedb_sql_final_label_mul.txt'

    # ====================================python打标签====================================
    # 无标签的地址--包括单后选和多候选
    staqc_python_f = '../data_processing/hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'
    large_python_f = '../data_processing/hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl'
    large_single_python_path = '../data_processing/hnn_process/ulabel_data/large_corpus/single/python_large_single_label.txt'
    # ---------有标签的地址----------
    hnn_lable_python_path = '../data_processing/hnn_process/ulabel_data/staqc/hnn_label_python.txt'
    # staqc:存放only-code、only-text、codemf标签地址
    staqc_python_final_lable = '../data_processing/hnn_process/ulabel_data/staqc/python_final_label.txt'
    # staqc:利用only-code、only-text、codemf中都为1筛选后的语料地址
    save_path_final_lable_staqc_python = '../data_processing/hnn_process/ulabel_data/staqc/combine_final_python_lable.txt'

    # large:存放only-code、only-text、codemf标签地址
    large_python_fianl_lable = '../data_processing/hnn_process/ulabel_data/staqc/large_python_final_label.txt'
    # large:利用only-code、only-text、codemf中都为1筛选后的语料地址
    save_path_final_lable_large_python_mul = '../data_processing/hnn_process/ulabel_data/staqc/combine_codedb_python_final_label_mul.txt'

    drop1 = drop2 = drop3 = drop4 = drop5 = np.round(0.25, 2)
    model.params_adjust(dropout1=drop1, dropout2=drop2, dropout3=drop3, dropout4=drop4, dropout5=drop5,
                        Regularizer=round(0.0004, 4), num=8, seed=42)

    model.build()
    if args.mode == 'eval':
        '''--------------------------------sql打标签-----------------------------------'''
        # 第一次执行:codemf
        # StandoneCode.load_model_epoch(model, 86, 0.25, 0.25, 0.25, 0.25, 0.0004000000000000001)
        # 第二次执行:text_sa
        # StandoneCode.load_model_epoch(model, 1033, 0.1, 0.1, 0.1, 0.1, 1.0002)
        # 第三次执行:code_sa
        StandoneCode.load_model_epoch(model, 1111, 0.1, 0.1, 0.1, 0.1, 101)

        # -----------------staqc_sql------------------------
        # 第一次执行
        # StandoneCode.u2l_codemf(model, staqc_sql_f, staqc_sql_final_label)
        # 第二次执行
        # StandoneCode.u2l_textsa(model, staqc_sql_f, staqc_sql_final_label)
        # 第三次执行
        # StandoneCode.u2l_codesa(model, staqc_sql_f, staqc_sql_final_label)

        # -----------------large_sql------------------------
        # 第一次执行
        # StandoneCode.u2l_codemf(model, staqc_sql_f, large_sql_fianl_lable)
        # 第二次执行
        # StandoneCode.u2l_textsa(model, staqc_sql_f, large_sql_fianl_lable)
        # 第三次执行
        # StandoneCode.u2l_codesa(model, staqc_sql_f, large_sql_fianl_lable)

        # =====================分析最终标签==============================
        # staqc:抽取codemf、testsa、codesa里面标签都为1
        final_analay(staqc_sql_final_label, hnn_lable_sql_path, save_path_final_lable_staqc_sql)
        # large:抽取codemf、testsa、codesa里面标签都为1，并把之前抽出的单候选合并进去
        # final_analay_large(large_sql_fianl_lable,hnn_lable_sql_path,large_single_path,save_path_final_lable_large_sql_mul)

        '''--------------------------------python打标签-----------------------------------'''
        # 第一次执行：codemf
        # StandoneCode.load_model_epoch(model, 1166, 0.5, 0.45, 0.55, 0.45, 0.0006)
        # 第二次执行：test_sa
        # StandoneCode.load_model_epoch(model, 1079, 0.5, 0.5, 0.5, 0.5, 1.0002)
        # 第三次执行code_sa
        # StandoneCode.load_model_epoch(model, 138, 0.15, 0.15, 0.15, 0.15, 101)

        # -----------------staqc_python------------------------
        # 第一次执行
        # StandoneCode.u2l_codemf(model, staqc_python_f, staqc_python_final_lable)
        # 第二次执行
        # StandoneCode.u2l_textsa(model, staqc_python_f, staqc_python_final_lable)
        # 第三次执行
        # StandoneCode.u2l_codesa(model, staqc_python_f, staqc_python_final_lable)

        # -----------------large_python------------------------
        # 第一次执行
        # StandoneCode.u2l_codemf(model, large_python_f, large_python_fianl_lable)
        # 第二次执行
        # StandoneCode.u2l_textsa(model, large_python_f, large_python_fianl_lable)
        # 第三次执行
        # StandoneCode.u2l_codesa(model, large_python_f, large_python_fianl_lable)

        # =====================分析最终标签==============================
        # staqc:抽取codemf、testsa、codesa里面标签都为1
        # final_analay(staqc_python_final_lable, hnn_lable_python_path, save_path_final_lable_staqc_python)
        # large:抽取codemf、testsa、codesa里面标签都为1,并把之前抽出的单候选合并进去
        # final_analay_large(large_python_fianl_lable, hnn_lable_python_path, large_single_python_path, save_path_final_lable_large_python_mul)



