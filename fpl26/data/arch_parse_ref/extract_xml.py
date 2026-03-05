from xml.etree import ElementTree as ET
from torch_geometric.data import Data
from extract_pb import *
import pandas as pd
import torch
from get_switch import load_switch_dict
from get_segment import *
from get_utilis import *

def process_tile_to_nodes_dict(file_path, vib_name, nodes_dict):

    tree = ET.parse(file_path)
    root = tree.getroot()

    switch_data_dict = load_switch_dict('switch_info.csv')
    segment_dict = create_segment_data_dict(file_path)

    vib_xpath = f"vib_arch/vib[@name='{vib_name}']/multistage_muxs"
    multistage_muxs = root.find(vib_xpath)

    if multistage_muxs is None:
        raise ValueError(f"No 'multistage_muxs' element found under 'vib name={vib_name}'.")
    else:
        # 处理找到的 multistage_muxs
        add_mux_node(multistage_muxs, nodes_dict, switch_data_dict, segment_dict, vib_name)

    global_feature = get_global(segment_dict, nodes_dict)

    return nodes_dict, global_feature


def process_xml_to_data(file_path, label, node_labels):
    # 解析 XML 文件
    tree = ET.parse(file_path)
    root = tree.getroot()

    vib_archs = root.findall("vib_arch/vib")

    nodes_dict = {}
    global_features = []

    extract_pb_nodes(file_path, nodes_dict)

    # 遍历所有的 'vib' 元素
    for vib in vib_archs:
        # 提取 'vib_name'
        vib_name = vib.get('name') if vib.get('name') is not None else 'Unknown'
        nodes_dict, global_feature = process_tile_to_nodes_dict(file_path, vib_name, nodes_dict)
        global_features.extend(global_feature)
    
    # print(nodes_dict)
    data = process_dict_to_data(nodes_dict, global_features, label, node_labels)

    return data



def process_dict_to_data(nodes_dict, global_features, label, node_labels):

    nodes_dict['startpoint'] = {
        'type': 'startpoint',
        'inputs': [],
        'outputs': [],
        'vib_name':'none'
    }

    nodes_dict['endpoint'] = {
        'type': 'endpoint',
        'inputs': [],
        'outputs': [],
        'vib_name': 'none'
    }


    for name, attrs in nodes_dict.items():
    # 排除 startpoint 和 end_point
        if name in ['startpoint', 'endpoint']:
            continue

        # 如果当前节点的 inputs 为空
        if not attrs.get('inputs', []):
            # 将当前节点的名字加入 startpoint 的 outputs
            nodes_dict['startpoint'].setdefault('outputs', []).append(name)
            # 将 startpoint 加入当前节点的 inputs
            attrs.setdefault('inputs', []).append('startpoint')

        # 如果当前节点的 outputs 为空
        if not attrs.get('outputs', []):
            # 将 end_point 加入当前节点的 outputs
            attrs.setdefault('outputs', []).append('endpoint')
            # 将当前节点的名字加入 end_point 的 inputs
            nodes_dict['endpoint'].setdefault('inputs', []).append(name)

    start_nodes = []
    end_nodes = []

    # 直接从nodes_dict提取边信息
    for node, attrs in nodes_dict.items():
        for output_node in attrs.get('outputs', []):
            start_nodes.append(node)
            end_nodes.append(output_node)

    # 将节点名称映射到连续的整数索引
    node_to_idx = {node: idx for idx, node in enumerate(nodes_dict.keys())}

    # print("End Nodes:", end_nodes)
    # print("Node to Index Keys:", node_to_idx)


    # 转换节点名称为索引
    start_nodes_idx = [node_to_idx[node] for node in start_nodes]
    end_nodes_idx = [node_to_idx[node] for node in end_nodes]

    # 创建edge_index张量
    edge_index = torch.tensor([start_nodes_idx, end_nodes_idx], dtype=torch.long)

    one_hot_encoding(nodes_dict)

    type_list = ['clb.q', 'startpoint', 'ble.out', 'mult_36.out', 'mem_1024x32_dp.data2', 'divisible_mult_18x18.a', 
                 'divisible_mult_18x18.b', 'memory.addr1', 'mem_1024x32_dp.clk', 'io_top.inpad', 'io_bottom.inpad', 
                 'mem_1024x32_dp.addr1', 'mem_1024x32_dp.out1', 'clb.Ie', 'memory.out', 'mult_36.b', 'ff.D', 'mult_18x18.a', 
                 'ff.Q', 'mult_18x18.out', 'memory.data', 'mult_18x18_slice.A_cfg', 'memory.addr2', 'clb.Ib', 'segment_out', 
                 'io_top.outpad', 'clb.o', 'fle.clk', 'fle.in', 'mux', 'mem_1024x32_dp.data1', 'mult_18x18_slice.B_cfg', 
                 'clb.Ia', 'clb.Ic', 'memory.clk', 'io_right.inpad', 'fle.out', 'ff.clk', 'mem_1024x32_dp.we2', 
                 'mult_36.a', 'clb.Id', 'lut_6.out', 'mem_1024x32_dp.addr2', 'memory.we2', 'clb.clk', 'mult_18x18.b', 
                 'lut_6.in', 'clb.Ih', 'io_bottom.outpad', 'mem_1024x32_dp.we1', 'memory.we1', 'mem_1024x32_dp.out2', 
                 'clb.If', 'io_left.outpad', 'endpoint', 'io_left.inpad', 'io_right.outpad', 'mult_18x18_slice.OUT_cfg', 
                 'ble.clk', 'ble.in', 'divisible_mult_18x18.out', 'clb.Ig', 'segment_in']

    feature_names = ['delay']
    position_features = ['from_x', 'from_y', 'to_x', 'to_y']
    seg_features = ['l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'l11', 'l12', 'l13', 'l14', 'l15', 'l16']
    vib_position_features = ['vib_bram2', 'vib_dsp3', 'vib_bram4', 'vib_bram0', 'vib_dsp0', 'vib1', 'vib_dsp2', 'vib0', 'vib3',
                          'vib2', 'vib5', 'vib4', 'vib_bram1', 'vib_bram5', 'vib_dsp1', 'vib_bram3', 'none']
    freq_features = ['freq']

    feature_names.extend(type_list)
    feature_names.extend(seg_features)
    feature_names.extend(position_features)
    feature_names.extend(vib_position_features)
    feature_names.extend(freq_features)

    features_list = []
    labels = [0] * len(nodes_dict)

    # print(nodes_dict)

    for node, attrs in nodes_dict.items():
        node_features = []
        for feature in feature_names:
            if feature in position_features:
                node_features.append(attrs.get(feature, 0) / 2)
            elif feature == 'delay':
                node_features.append(attrs.get(feature, 0) * 10**11)
            elif feature in seg_features:
                node_features.append(attrs.get(feature, 0))
            elif feature in vib_position_features:
                node_features.append(attrs.get(feature, 0))
            elif feature in type_list:
                node_features.append(attrs.get(feature, 0))
            elif feature in freq_features:
                node_features.append(attrs.get(feature, 0) * 10)
            else:
                raise ValueError(f"Unknown feature: {feature}")
        features_list.append(node_features)
        labels[node_to_idx[node]] = node_labels.get(node, 0)

    # print(features_list)

    # 转换特征列表为张量
    features_tensor = torch.tensor(features_list, dtype=torch.float)

    # 将标签转换为张量
    labels_tensor = torch.tensor(labels, dtype=torch.int)

    global_features.append(label)
    # 将全局特征转换为张量
    global_features_tensor = torch.tensor(global_features, dtype=torch.float)

    # 使用字典组织特征
    data = Data(x=features_tensor, edge_index=edge_index, y=labels_tensor, other_attrs=global_features_tensor)

    return data


# file_path = "/home/wllpro/llwang/yfdai/HRAE_paper/raw_dataset_archs/vib_new0_10.xml"  # Replace with the actual file path
# data = process_xml_to_data(file_path, 0.5, {'mem_1024x32_dp.out2[30]':0})
# print(data)
