from xml.etree import ElementTree as ET
from get_switch import query_tdel
from get_segment import get_position
import re
import pandas as pd

def normal_delay(delay):
    return delay

def get_seg_direct(input_str: str) -> str:
    if '.' in input_str:
        parts = input_str.split('.')
        if len(parts) > 1 and len(parts[1]) > 0:
            return parts[1][0]  # 获取 '.' 后部分的第一个字母
        else:
            raise ValueError(f"No valid character found after '.' in input: {input_str}")
    else:
        raise ValueError(f"No '.' found in input: {input_str}")

def get_seg_length(input_str: str) -> str:
    match = re.search(r'l(\d+)\.', input_str)
    if match:
        return match.group(1)  # 返回匹配到的数字
    else:
        raise ValueError(f"No valid format found in input: {input_str}")

def get_seg_freq(segment_dict, length) -> float:
    return segment_dict[length]['freq']

def identify_to_type(input_str: str) -> str:
    if input_str.startswith("clb.I") and len(input_str) > len("clb.I"):
        return input_str[:len("clb.I") + 1]
    elif input_str.startswith("l"):
        return "segment_out"
    elif input_str.startswith("io"):
        return "io"
    elif input_str.startswith("memory"):
        return "memory"
    elif input_str.startswith("mult"):
        return "mult"
    else:
        raise TypeError(f"Unknown type for input: {input_str}")

def identify_from_type(input_str: str) -> str:
    if input_str.startswith("clb.") and len(input_str) > len("clb."):
        return input_str[:len("clb.") + 1]
    elif input_str.startswith("l"):
        return "segment_in"
    elif input_str.startswith("mux") or input_str.startswith("omux"):
        return "mux"
    elif input_str.startswith("io"):
        return "io"
    elif input_str.startswith("memory"):
        return "memory"
    elif input_str.startswith("mult"):
        return "mult"
    else:
        raise TypeError(f"Unknown type for input: {input_str}")

def add_prefix_name(original_names, prefix):
    new_names = []
    for name in original_names:
        if name.startswith("io") or name.startswith("clb") or name.startswith("memory") or name.startswith("mult"):
            new_names.append(name)
        else:
            new_names.append(prefix + name)
    return new_names, original_names

def add_prefix_name_single(original_name, prefix):
    if original_name.startswith("io") or original_name.startswith("clb") or original_name.startswith("memory") or original_name.startswith("mult"):
        return original_name
    else:
        return prefix + original_name

# Function to add nodes from 'mux' tags
def add_mux_node(root, nodes_dict, switch_dict, segment_dict, vib_name):
    prefix = f"{vib_name}_"
    first_stage = root.find("first_stage")
    if first_stage is None:
        raise ValueError("No 'first_stage' element found under 'multistage_muxs'.")

    # Iterate through each 'mux' in 'first_stage'
    for mux in first_stage.iter("mux"):
        # Extract the 'name' attribute
        mux_name = add_prefix_name_single(mux.get("name"), prefix)
        
        from_text = mux.find("from")
        from_elements = []

        # If the 'from' element exists, split its content by spaces
        if from_text is not None and from_text.text is not None:
            # Split by spaces and store in the array
            from_elements = from_text.text.strip().split()
        from_elements = add_from_suffix(from_elements)
        from_elements, from_elements_original = add_prefix_name(from_elements, prefix)
        
        if mux_name and mux_name not in nodes_dict:
            nodes_dict[mux_name] = {
                'name': mux_name,
                'type': 'mux',
                'delay': query_tdel(switch_dict, 'only_mux',len(from_elements)),
                'inputs': from_elements,
                'outputs': [],
                'inform_distance': 0,
                'level': 'clb',
                'to_x': 0,
                'to_y': 0,
                'vib_name': vib_name,
            }
        else:
            nodes_dict[mux_name]['inputs'].extend(from_elements)
            nodes_dict[mux_name]['delay'] = query_tdel(switch_dict, 'only_mux',len(nodes_dict[mux_name]['inputs']))
        
        for from_node in from_elements_original:
            add_from_node(from_node, nodes_dict, segment_dict, vib_name)
        
        for from_node in from_elements:
            nodes_dict[f"{from_node}"]['outputs'].append(mux_name)

    second_stage = root.find("second_stage")
    if second_stage is None:
        raise ValueError("No 'second_stage' element found under 'multistage_muxs'.")

    # Iterate through each 'mux' in 'second_stage'
    for mux in second_stage.iter("mux"):
        # Extract the 'name' attribute
        mux_name = mux.get("name")
        to_text = mux.find("to")
        from_text = mux.find("from")
        from_elements = []

        # If the 'from' element exists, split its content by spaces
        if from_text is not None and from_text.text is not None:
            # Split by spaces and store in the array
            from_elements = from_text.text.strip().split()
            from_elements = add_from_suffix(from_elements)
            from_elements, from_elements_original = add_prefix_name(from_elements, prefix)
        
        type = identify_to_type(to_text.text)

        if type == 'segment_out':
            direct = get_seg_direct(to_text.text)
            length = get_seg_length(to_text.text)
            to_x, to_y = get_position(segment_dict, length, direct, 'out')
            delay = query_tdel(switch_dict, length, len(from_elements))
            name = prefix + to_text.text + '_to'
            freq = get_seg_freq(segment_dict, length)
            if name not in nodes_dict:
                nodes_dict[name] = {
                    'name': name,
                    'type': type,
                    'delay': delay,
                    'inputs': from_elements,
                    'outputs': [],
                    'inform_distance': length,
                    'direct': direct,
                    'level': 'seg',
                    'to_x': to_x,
                    'to_y': to_y,
                    'freq': freq,
                    'vib_name': vib_name,
                }
            else:
                nodes_dict[name]['inputs'].extend(from_elements)
                nodes_dict[name]['delay'] = query_tdel(switch_dict, 'ipin', len(nodes_dict[name]['inputs']))
        
        else:
            name = add_prefix_name_single(to_text.text, prefix)
            delay = query_tdel(switch_dict, 'ipin',len(from_elements))
            if name not in nodes_dict:
                raise KeyError(f"The key '{name}' is not present in the dictionary.")
            else:
                nodes_dict[name]['inputs'].extend(from_elements)
                nodes_dict[name]['delay'] = query_tdel(switch_dict, 'ipin', len(nodes_dict[name]['inputs']))
                
        for from_node in from_elements_original:
            add_from_node(from_node, nodes_dict, segment_dict, vib_name)
        
        for from_node in from_elements:
            nodes_dict[f"{from_node}"]['outputs'].append(name)
    
def add_from_suffix(from_elements):
    # 创建一个新的列表来存储处理后的 from_node
    updated_from_elements = []
    
    for from_node in from_elements:
        # 判断 from_node 是否为 'segment_input' 类型
        if identify_from_type(from_node) == 'segment_in':
            # 如果满足条件，则加上 '_from' 后缀
            updated_from_elements.append(from_node + '_from')
        else:
            # 不满足条件的保持原样
            updated_from_elements.append(from_node)
    
    return updated_from_elements

def add_from_node(element, nodes_dict, segment_dict, vib_name):
    prefix = f"{vib_name}_"
    type = identify_from_type(element)
    name = element
    if type == 'segment_in':
        name = prefix + name
        if name in nodes_dict:
            return
        else:
            direct = get_seg_direct(name)
            length = get_seg_length(name)
            from_x, from_y = get_position(segment_dict, length, direct, 'in')
            freq = get_seg_freq(segment_dict, length)
            nodes_dict[name] = {
            'name': name,
            'type': type,
            'delay': 0,
            'inputs': [],
            'outputs': [],
            'inform_distance': length,
            'level': 'seg',
            'direct': direct,
            'from_x': from_x,
            'from_y': from_y,
            'freq': freq,
            'vib_name': vib_name,
        }
    
    elif type == 'mux':
        name = prefix + name
        if name in nodes_dict:
            return
        else:
            nodes_dict[name] = {
            'name': name,
            'type': type,
            'delay': 0,
            'inputs': [],
            'outputs': [],
            'inform_distance': 0,
            'level': 'mux',
            'vib_name': vib_name,
        }
            
    else:
        if name not in nodes_dict:
            raise ValueError('Unknown node type: ' + name)
        

# 添加种类特征
def one_hot_encoding(node_dict):
    # types = set()
    # for node in node_dict:
    #     types.add(node_dict[node]['type'])

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

    for name, attributes in node_dict.items():
        # 遍历 type_list 中的每个类型
        for type_name in type_list:
            # 检查节点的 'type' 是否与当前类型相匹配
            if attributes['type'] == type_name:
                attributes[type_name] = 1
            else:
                attributes[type_name] = 0
        if attributes['type'] not in type_list:
            print(f"Error: Node '{name}' has an unknown type: {attributes['type']}")
            raise ValueError(f"Node '{name}' has an unrecognized type '{attributes['type']}'")

    seg_features = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']

    vib_name_features = ['vib_bram2', 'vib_dsp3', 'vib_bram4', 'vib_bram0', 'vib_dsp0', 'vib1', 'vib_dsp2', 'vib0', 'vib3',
                          'vib2', 'vib5', 'vib4', 'vib_bram1', 'vib_bram5', 'vib_dsp1', 'vib_bram3', 'none']
    
    for name, attributes in node_dict.items():

        if 'inform_distance' not in attributes:
            attributes['inform_distance'] = '0'  # 如果不存在，设为 '0'
        else:
            attributes['inform_distance'] = str(attributes['inform_distance'])

        # 检查节点的 'inform_distance' 是否在 seg_features 中
        if attributes['inform_distance'] not in seg_features:
            print(f"Error: Node '{name}' has an unknown inform_distance: {attributes['inform_distance']}")
            raise ValueError(f"Node '{name}' has an unrecognized inform_distance '{attributes['inform_distance']}'")

        # 遍历 seg_features 中的每个类型，并在前面加 'l'
        for seg in seg_features:
            prefixed_seg = 'l' + seg  # 加上前缀 'l'
            if attributes['inform_distance'] == seg:
                attributes[prefixed_seg] = 1
            else:
                attributes[prefixed_seg] = 0  
                
        for vib in vib_name_features:
            if str(attributes['vib_name']) == vib:
                attributes[vib] = 1
            else:
                attributes[vib] = 0  
    
    return node_dict