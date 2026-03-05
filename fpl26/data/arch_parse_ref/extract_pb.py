import xml.etree.ElementTree as ET
from itertools import chain  

def extract_pb_nodes(file_path, nodes_dict):
    extract_clb_nodes(file_path, nodes_dict)
    extract_io_left_nodes(nodes_dict)
    extract_io_right_nodes(nodes_dict)
    extract_io_top_nodes(nodes_dict)
    extract_io_bottom_nodes(nodes_dict)
    extract_mult_36_nodes(file_path, nodes_dict)
    extract_memory_nodes(file_path, nodes_dict)


def construct_nodes_dict(root, root_name, nodes_dict, level, vib_name):
    for node_type in ['input', 'output', 'clock']:
        for element in root.findall(f"./{node_type}"):
            num_pins = int(element.get('num_pins'))
            name_prefix = element.get('name')  # Expected to be something like 'Ia'

            if num_pins == 1:
                node_name = f"{root_name}.{name_prefix}"
                nodes_dict[node_name] = {
                    'name': node_name,
                    'type': level + '.' + name_prefix,
                    'level':level,
                    'delay': 0,
                    'inputs': [],
                    'outputs': [],
                    'inform_distance': 0,
                    'vib_name': vib_name
                }
            else:
                # Create nodes based on num_pins
                for i in range(num_pins):
                    node_name = f"{root_name}.{name_prefix}[{i}]"
                    nodes_dict[node_name] = {
                        'name': node_name,
                        'type': level + '.' + name_prefix,
                        'level':level,
                        'delay': 0,
                        'inputs': [],
                        'outputs': [],
                        'inform_distance': 0,
                        'vib_name': vib_name
                    }

    return nodes_dict

def complete_connections(inputs, outputs, nodes_dict):
    for input_name in inputs:
        nodes_dict[input_name]['outputs'].extend(outputs)
    for output_name in outputs:
        nodes_dict[output_name]['inputs'].extend(inputs)


def extract_clb_nodes(xml_file, nodes_dict):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    pb_type_name='clb'
    pb_types = root.findall(".//pb_type[@name='" + pb_type_name + "']")

    # 创建一个新的XML树，仅包含找到的pb_type
    new_root = ET.Element("complexblocks")
    for pb in pb_types:
        new_root.append(pb)

    # 最外层clb
    construct_nodes_dict(new_root.find(".//pb_type[@name='" + 'clb' + "']"), 'clb', nodes_dict, 'clb', 'vib0')

    # 建立8个fle
    fle_root = new_root.find(".//pb_type[@name='" + 'fle' + "']")
    fle_num = int(fle_root.get('num_pb'))
    for num in range(fle_num):
        fle_name = f"fle[{num}]"
        construct_nodes_dict(fle_root, fle_name, nodes_dict, 'fle', 'vib0')

        ble_root = fle_root.find(".//pb_type[@name='" + 'ble' + "']")
        ble_name = fle_name + f".ble"
        construct_nodes_dict(ble_root, ble_name, nodes_dict, 'ble', 'vib0')
        
        lut_6_root = ble_root.find(".//pb_type[@name='" + 'lut_6' + "']")
        lut_6_name = ble_name + f".lut_6"
        construct_nodes_dict(lut_6_root, lut_6_name, nodes_dict, 'lut_6', 'vib0')

        ff_root = ble_root.find(".//pb_type[@name='" + 'ff' + "']")
        ff_name = ble_name + f".ff"
        construct_nodes_dict(ff_root, ff_name, nodes_dict, 'ff', 'vib0')

        #添加延时信息和块内连接信息
        lut_6_delay_matrix = 1.43259e-10
        for i in range(6):
            nodes_dict[lut_6_name+f'.in[{i}]']['delay'] = lut_6_delay_matrix
            nodes_dict[lut_6_name+f'.in[{i}]']['outputs'].append(lut_6_name+f'.out')
            nodes_dict[lut_6_name+f'.out']['inputs'].append(lut_6_name+f'.in[{i}]')
        nodes_dict[ff_name+'.D']['delay'] = 1.891e-11
        nodes_dict[ff_name+'.clk']['delay'] = 6.032e-11
        nodes_dict[ff_name+'.Q']['inputs'].append(ff_name+'.clk')
        nodes_dict[ff_name+'.clk']['outputs'].append(ff_name+'.Q')

        # 添加连接信息
        # interconnect_root = ble_root.find(".//interconnect")
        # directs = interconnect_root.findall(".//direct")
        # for direct in directs:
        #     src_node = direct.get('input')
        #     sink_node = direct.get('output')
        #     nodes_dict[sink_node]['inputs'].append(src_node)
        #     nodes_dict[src_node]['outputs'].append(sink_node)
        
        #specific connections

        #ble --lut6
        for i in range(6):
            nodes_dict[ble_name+f'.in[{i}]']['outputs'].append(lut_6_name+f'.in[{i}]')
            nodes_dict[lut_6_name+f'.in[{i}]']['inputs'].append(ble_name+f'.in[{i}]')

        #lut6 --ff
        nodes_dict[lut_6_name+'.out']['outputs'].append(ff_name+'.D')
        nodes_dict[ff_name+'.D']['inputs'].append(lut_6_name+'.out')

        #ble out --lut6, ff
        nodes_dict[lut_6_name+'.out']['outputs'].append(ble_name+'.out[0]')
        nodes_dict[ble_name+'.out[0]']['inputs'].append(lut_6_name+'.out')
        nodes_dict[ff_name+'.Q']['outputs'].append(ble_name+'.out[1]')
        nodes_dict[ble_name+'.out[1]']['inputs'].append(ff_name+'.Q')

        #ble--fle
        nodes_dict['clb.clk']['outputs'].extend([ff_name+'.clk', fle_name+'.clk', ble_name+'.clk'])
        for i in [ff_name+'.clk', fle_name+'.clk', ble_name+'.clk']:
            nodes_dict[i]['inputs'].append('clb.clk')
        for i in range(6):
            nodes_dict[fle_name+f'.in[{i}]']['outputs'].append(ble_name+f'.in[{i}]')
            nodes_dict[ble_name+f'.in[{i}]']['inputs'].append(fle_name+f'.in[{i}]')
        for i in range(2):
            nodes_dict[ble_name+f'.out[{i}]']['outputs'].append(fle_name+f'.out[{i}]')
            nodes_dict[fle_name+f'.out[{i}]']['inputs'].append(ble_name+f'.out[{i}]')
        

    #fle --clb
    #connect all clb input nodes' outputs to fle input nodes
    for i in range(6):
        nodes_dict[f'clb.Ia[{i}]']['outputs'].append(f'fle[0].in[{i}]')
        nodes_dict[f'clb.Ib[{i}]']['outputs'].append(f'fle[1].in[{i}]')
        nodes_dict[f'clb.Ic[{i}]']['outputs'].append(f'fle[2].in[{i}]')
        nodes_dict[f'clb.Id[{i}]']['outputs'].append(f'fle[3].in[{i}]')
        nodes_dict[f'clb.Ie[{i}]']['outputs'].append(f'fle[4].in[{i}]')
        nodes_dict[f'clb.If[{i}]']['outputs'].append(f'fle[5].in[{i}]')
        nodes_dict[f'clb.Ig[{i}]']['outputs'].append(f'fle[6].in[{i}]')
        nodes_dict[f'clb.Ih[{i}]']['outputs'].append(f'fle[7].in[{i}]')

        nodes_dict[f'fle[0].in[{i}]']['inputs'].append(f'clb.Ia[{i}]')
        nodes_dict[f'fle[1].in[{i}]']['inputs'].append(f'clb.Ib[{i}]')
        nodes_dict[f'fle[2].in[{i}]']['inputs'].append(f'clb.Ic[{i}]')
        nodes_dict[f'fle[3].in[{i}]']['inputs'].append(f'clb.Id[{i}]')
        nodes_dict[f'fle[4].in[{i}]']['inputs'].append(f'clb.Ie[{i}]')
        nodes_dict[f'fle[5].in[{i}]']['inputs'].append(f'clb.If[{i}]')
        nodes_dict[f'fle[6].in[{i}]']['inputs'].append(f'clb.Ig[{i}]')
        nodes_dict[f'fle[7].in[{i}]']['inputs'].append(f'clb.Ih[{i}]')

    #connect all clb outputs    
    for i in range(8):
        nodes_dict[f'clb.o[{i}]']['inputs'].append(f'fle[{i}].out[0]')
        nodes_dict[f'fle[{i}].out[0]']['outputs'].append(f'clb.o[{i}]')
        nodes_dict[f'clb.q[{i}]']['inputs'].append(f'fle[{i}].out[1]')
        nodes_dict[f'fle[{i}].out[1]']['outputs'].append(f'clb.q[{i}]')

def extract_io_left_nodes(nodes_dict):
    inputs = []
    outputs = []
    for i in range(8):
        node_name = f"io_left.inpad[{i}]"
        nodes_dict[node_name] = {
            'name': node_name,
            'type': 'io_left.inpad',
            'level':'io_left',
            'delay': 4.243e-11,
            'inputs': [],
            'outputs': [],
            'inform_distance': 0,
            'vib_name': 'vib1'
        }
        inputs.append(node_name)
        node_name = f"io_left.outpad[{i}]"
        nodes_dict[node_name] = {
            'name': node_name,
            'type': 'io_left.outpad',
            'level':'io_left',
            'delay': 1.394e-11,
            'inputs': [],
            'outputs': [],
            'inform_distance': 0,
            'vib_name': 'vib1'
        }
        outputs.append(node_name)
    
    complete_connections(inputs, outputs, nodes_dict)


def extract_io_right_nodes(nodes_dict):
    inputs = []
    outputs = []
    for i in range(8):
        node_name = f"io_right.inpad[{i}]"
        nodes_dict[node_name] = {
            'name': node_name,
            'type': 'io_right.inpad',
            'level':'io_right',
            'delay': 4.243e-11,
            'inputs': [],
            'outputs': [],
            'inform_distance': 0,
            'vib_name': 'vib2'
        }
        inputs.append(node_name)
        node_name = f"io_right.outpad[{i}]"
        nodes_dict[node_name] = {
            'name': node_name,
            'type': 'io_right.outpad',
            'level':'io_right',
            'delay': 1.394e-11,
            'inputs': [],
            'outputs': [],
            'inform_distance': 0,
            'vib_name': 'vib2'
        }
        outputs.append(node_name)
    
    complete_connections(inputs, outputs, nodes_dict)



def extract_io_top_nodes(nodes_dict):
    inputs = []
    outputs = []
    for i in range(8):
        node_name = f"io_top.inpad[{i}]"
        nodes_dict[node_name] = {
            'name': node_name,
            'type': 'io_top.inpad',
            'level':'io_top',
            'delay': 4.243e-11,
            'inputs': [],
            'outputs': [],
            'inform_distance': 0,
            'vib_name': 'vib3'
        }
        inputs.append(node_name)
        node_name = f"io_top.outpad[{i}]"
        nodes_dict[node_name] = {
            'name': node_name,
            'type': 'io_top.outpad',
            'level':'io_top',
            'delay': 1.394e-11,
            'inputs': [],
            'outputs': [],
            'inform_distance': 0,
            'vib_name': 'vib3'
        }
        outputs.append(node_name)
    
    complete_connections(inputs, outputs, nodes_dict)



def extract_io_bottom_nodes(nodes_dict):
    inputs = []
    outputs = []
    for i in range(8):
        node_name = f"io_bottom.inpad[{i}]"
        nodes_dict[node_name] = {
            'name': node_name,
            'type': 'io_bottom.inpad',
            'level':'io_bottom',
            'delay': 4.243e-11,
            'inputs': [],
            'outputs': [],
            'inform_distance': 0,
            'vib_name': 'vib4'
        }
        inputs.append(node_name)
        node_name = f"io_bottom.outpad[{i}]"
        nodes_dict[node_name] = {
            'name': node_name,
            'type': 'io_bottom.outpad',
            'level':'io_bottom',
            'delay': 1.394e-11,
            'inputs': [],
            'outputs': [],
            'inform_distance': 0,
            'vib_name': 'vib4'
        }
        outputs.append(node_name)

    complete_connections(inputs, outputs, nodes_dict)



def extract_mult_36_nodes(xml_file, nodes_dict):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    pb_type_name='mult_36'
    pb_types = root.findall(".//pb_type[@name='" + pb_type_name + "']")

    # 创建一个新的XML树，仅包含找到的pb_type
    new_root = ET.Element("complexblocks")
    for pb in pb_types:
        new_root.append(pb)

    # 最外层mult_36
    construct_nodes_dict(new_root.find(".//pb_type[@name='" + 'mult_36' + "']"), 'mult_36', nodes_dict, 'mult_36', 'vib_dsp0')

    # 建立2个divisible_mult_18x18
    divisible_mult_18x18_root = new_root.find(".//pb_type[@name='" + 'divisible_mult_18x18' + "']")
    divisible_mult_18x18_num = int(divisible_mult_18x18_root.get('num_pb'))
    for t in range(divisible_mult_18x18_num):
        divisible_mult_18x18_name = f"divisible_mult_18x18[{t}]"
        construct_nodes_dict(divisible_mult_18x18_root, divisible_mult_18x18_name, nodes_dict, 'divisible_mult_18x18', 'vib_dsp0')

        mult_18x18_slice_root = divisible_mult_18x18_root.find(".//pb_type[@name='" + 'mult_18x18_slice' + "']")
        mult_18x18_slice_name = divisible_mult_18x18_name + f".mult_18x18_slice"
        construct_nodes_dict(mult_18x18_slice_root, mult_18x18_slice_name, nodes_dict, 'mult_18x18_slice', 'vib_dsp0')
        
        mult_18x18_root = mult_18x18_slice_root.find(".//pb_type[@name='" + 'mult_18x18' + "']")
        mult_18x18_name = mult_18x18_slice_name + f".mult_18x18"
        construct_nodes_dict(mult_18x18_root, mult_18x18_name, nodes_dict, 'mult_18x18', 'vib_dsp0')


        for i in range(36):
            nodes_dict[mult_18x18_name+f'.out[{i}]']['delay'] = 1.523e-9
            for j in range(18):
                nodes_dict[mult_18x18_name+f'.a[{j}]']['outputs'].append(mult_18x18_name+f'.out[{i}]')
                nodes_dict[mult_18x18_name+f'.b[{j}]']['outputs'].append(mult_18x18_name+f'.out[{i}]')
                nodes_dict[mult_18x18_name+f'.out[{i}]']['inputs'].append(mult_18x18_name+f'.a[{j}]')
                nodes_dict[mult_18x18_name+f'.out[{i}]']['inputs'].append(mult_18x18_name+f'.b[{j}]')

        for i in range(18):
            nodes_dict[mult_18x18_name+f'.a[{i}]']['inputs'].append(mult_18x18_slice_name+f'.A_cfg[{i}]')
            nodes_dict[mult_18x18_name+f'.b[{i}]']['inputs'].append(mult_18x18_slice_name+f'.B_cfg[{i}]')
            nodes_dict[mult_18x18_slice_name+f'.A_cfg[{i}]']['outputs'].append(mult_18x18_name+f'.a[{i}]')
            nodes_dict[mult_18x18_slice_name+f'.B_cfg[{i}]']['outputs'].append(mult_18x18_name+f'.b[{i}]')

        for i in range(36):
            nodes_dict[mult_18x18_name+f'.out[{i}]']['outputs'].append(mult_18x18_slice_name+f'.OUT_cfg[{i}]')
            nodes_dict[mult_18x18_slice_name+f'.OUT_cfg[{i}]']['inputs'].append(mult_18x18_name+f'.out[{i}]')

        for i in range(18):
            nodes_dict[mult_18x18_slice_name+f'.A_cfg[{i}]']['inputs'].append(divisible_mult_18x18_name+f'.a[{i}]')
            nodes_dict[mult_18x18_slice_name+f'.B_cfg[{i}]']['inputs'].append(divisible_mult_18x18_name+f'.b[{i}]')
            nodes_dict[divisible_mult_18x18_name+f'.a[{i}]']['outputs'].append(mult_18x18_slice_name+f'.A_cfg[{i}]')
            nodes_dict[divisible_mult_18x18_name+f'.b[{i}]']['outputs'].append(mult_18x18_slice_name+f'.B_cfg[{i}]')
        
        for i in range(36):
            nodes_dict[divisible_mult_18x18_name+f'.out[{i}]']['outputs'].append(mult_18x18_slice_name+f'.OUT_cfg[{i}]')
            nodes_dict[mult_18x18_slice_name+f'.OUT_cfg[{i}]']['inputs'].append(divisible_mult_18x18_name+f'.out[{i}]')

        for i in range(18):
            nodes_dict[mult_18x18_slice_name+f'.A_cfg[{i}]']['inputs'].append('mult_36'+f'.a[{t * 18 + i}]')
            nodes_dict[mult_18x18_slice_name+f'.B_cfg[{i}]']['inputs'].append('mult_36'+f'.b[{t * 18 + i}]')
            nodes_dict['mult_36'+f'.a[{t * 18 + i}]']['outputs'].append(mult_18x18_slice_name+f'.A_cfg[{i}]')
            nodes_dict['mult_36'+f'.b[{t * 18 + i}]']['outputs'].append(mult_18x18_slice_name+f'.B_cfg[{i}]')
            nodes_dict[mult_18x18_slice_name+f'.A_cfg[{i}]']['delay'] = 134e-12
            nodes_dict[mult_18x18_slice_name+f'.B_cfg[{i}]']['delay'] = 134e-12
        
        for i in range(36):
            nodes_dict[mult_18x18_slice_name+f'.OUT_cfg[{i}]']['outputs'].append('mult_36'+f'.out[{t * 36 + i}]')
            nodes_dict['mult_36'+f'.out[{t * 36 + i}]']['inputs'].append(mult_18x18_slice_name+f'.OUT_cfg[{i}]')

    return nodes_dict

def extract_memory_nodes(xml_file, nodes_dict):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    pb_type_name='memory'
    pb_types = root.findall(".//pb_type[@name='" + pb_type_name + "']")

    # 创建一个新的XML树，仅包含找到的pb_type
    new_root = ET.Element("complexblocks")
    for pb in pb_types:
        new_root.append(pb)

    # 最外层memory
    construct_nodes_dict(new_root.find(".//pb_type[@name='" + 'memory' + "']"), 'memory', nodes_dict, 'memory', 'vib_bram0')

    # 建立1个mem_1024x32_dp
    mem_1024x32_dp_root = new_root.find(".//pb_type[@name='" + 'mem_1024x32_dp' + "']")
    mem_1024x32_dp_name = "mem_1024x32_dp"
    construct_nodes_dict(mem_1024x32_dp_root, mem_1024x32_dp_name, nodes_dict, 'mem_1024x32_dp', 'vib_bram0')

    for i in range(32):
        nodes_dict[mem_1024x32_dp_name+f'.out1[{i}]']['outputs'].append('memory'+f'.out[{i}]')
        nodes_dict['memory'+f'.out[{i}]']['inputs'].append(mem_1024x32_dp_name+f'.out1[{i}]')
        nodes_dict['memory'+f'.out[{i}]']['delay'] = 40e-12
        nodes_dict[mem_1024x32_dp_name+f'.out1[{i}]']['delay'] = 1.234e-9
        nodes_dict[mem_1024x32_dp_name+f'.out2[{i}]']['outputs'].append('memory'+f'.out[{i + 32}]')
        nodes_dict['memory'+f'.out[{i + 32}]']['inputs'].append(mem_1024x32_dp_name+f'.out2[{i}]')
        nodes_dict['memory'+f'.out[{i + 32}]']['delay'] = 40e-12
        nodes_dict[mem_1024x32_dp_name+f'.out2[{i}]']['delay'] = 1.234e-9

        nodes_dict['memory'+f'.data[{i}]']['outputs'].append(mem_1024x32_dp_name+f'.data1[{i}]')
        nodes_dict[mem_1024x32_dp_name+f'.data1[{i}]']['inputs'].append('memory'+f'.data[{i}]')
        nodes_dict['memory'+f'.data[{i + 32}]']['outputs'].append(mem_1024x32_dp_name+f'.data2[{i}]')
        nodes_dict[mem_1024x32_dp_name+f'.data2[{i}]']['inputs'].append('memory'+f'.data[{i + 32}]')
        nodes_dict[mem_1024x32_dp_name+f'.data1[{i}]']['delay'] = 132e-12
        nodes_dict[mem_1024x32_dp_name+f'.data2[{i}]']['delay'] = 132e-12

    for i in range(10):
        nodes_dict['memory'+f'.addr1[{i}]']['outputs'].append(mem_1024x32_dp_name+f'.addr1[{i}]')
        nodes_dict[mem_1024x32_dp_name+f'.addr1[{i}]']['inputs'].append('memory'+f'.addr1[{i}]')
        nodes_dict[mem_1024x32_dp_name+f'.addr1[{i}]']['delay'] = 132e-12
        nodes_dict['memory'+f'.addr2[{i}]']['outputs'].append(mem_1024x32_dp_name+f'.addr2[{i}]')
        nodes_dict[mem_1024x32_dp_name+f'.addr2[{i}]']['inputs'].append('memory'+f'.addr2[{i}]')
        nodes_dict[mem_1024x32_dp_name+f'.addr2[{i}]']['delay'] = 132e-12

    for i in range(32):
        nodes_dict[mem_1024x32_dp_name+f'.clk']['outputs'].append(mem_1024x32_dp_name+f'.out1[{i}]')
        nodes_dict[mem_1024x32_dp_name+f'.out1[{i}]']['inputs'].append(mem_1024x32_dp_name+f'.clk')
        nodes_dict[mem_1024x32_dp_name+f'.clk']['outputs'].append(mem_1024x32_dp_name+f'.out2[{i}]')
        nodes_dict[mem_1024x32_dp_name+f'.out2[{i}]']['inputs'].append(mem_1024x32_dp_name+f'.clk')

    nodes_dict[mem_1024x32_dp_name+f'.clk']['delay'] = 509e-12


    value = nodes_dict['memory'+f'.we1']
    del nodes_dict['memory'+f'.we1']
    nodes_dict['memory'+f'.we1[0]'] = value

    value = nodes_dict['memory'+f'.we2']
    del nodes_dict['memory'+f'.we2']
    nodes_dict['memory'+f'.we2[0]'] = value

    nodes_dict['memory'+f'.we1[0]']['outputs'].append(mem_1024x32_dp_name+f'.we1')
    nodes_dict[mem_1024x32_dp_name+f'.we1']['inputs'].append('memory'+f'.we1[0]')
    nodes_dict['memory'+f'.we2[0]']['outputs'].append(mem_1024x32_dp_name+f'.we2')
    nodes_dict[mem_1024x32_dp_name+f'.we2']['inputs'].append('memory'+f'.we2[0]')

    return nodes_dict

if __name__ == '__main__':
    nodes_dict = {}
    nodes_dict = extract_clb_nodes('/home/wllpro/llwang/yfdai/HRAE_paper/raw_dataset_archs/vib_new_999.xml')
    print(nodes_dict)
