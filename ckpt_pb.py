import tensorflow as tf
import config as cfg
from tensorflow.python.framework import graph_util
WEIGHTS_INIT = "weights/yolo.ckpt-98-0.7907"
__weights_init = WEIGHTS_INIT

output_node_sbbox = "yolov3/pred_sbbox/concat_2"
output_node_mbbox = "yolov3/pred_mbbox/concat_2"
output_node_lbbox = "yolov3/pred_lbbox/concat_2"
output_node_name = "yolov3/pred_sbbox/concat_2,yolov3/pred_mbbox/concat_2,yolov3/pred_lbbox/concat_2"
moving_ave_decay = cfg.MOVING_AVE_DECAY

output_graph = "frozen_model.pb"

saver = tf.train.import_meta_graph(__weights_init + '.meta', clear_devices=True)
graph = tf.get_default_graph()  # 获得默认的图
input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图



with tf.Session() as sess:
    saver.restore(sess, __weights_init)  # 恢复图并得到数据
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
    output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
        sess=sess,
        input_graph_def=sess.graph_def,  # 等于:sess.graph_def
        output_node_names=output_node_name.split(","))  # 如果有多个输出节点，以逗号隔开
        #output_node_names=[output_node_sbbox, output_node_mbbox, output_node_lbbox])

    with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
        f.write(output_graph_def.SerializeToString())  # 序列化输出
    print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


"""

org_weights_mess = []

G = tf.Graph()
with G.as_default():
    load = tf.train.import_meta_graph(__weights_init + '.meta', clear_devices=True)
    with tf.Session(graph=G) as sess_G:
        load.restore(sess_G, __weights_init)
        for var in tf.global_variables():
            var_name = var.op.name
            var_name_mess = str(var_name).split('/')
            var_shape = var.shape
            #if var_name_mess[-1] != 'pred_sbbox':
                #continue
            # print(var_name_mess[0])
            org_weights_mess.append([var_name, var_shape])
            print(var_name)
            print("\n")
print(org_weights_mess)
"""