import xml.etree.ElementTree as ET


def modify_xml_file_pedestrian(xml_file):
    # 解析 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 寻找并修改带有 allow="pedestrian" 属性的边
    for edge in root.findall('edge'):
        for lane in edge.findall('lane'):
            if 'allow' in lane.attrib and lane.attrib['allow'] == 'pedestrian':
                lane.attrib.pop('allow')  # 删除 allow 属性
                lane.set('disallow', 'tram rail_urban rail rail_electric rail_fast ship')


    # 将修改后的 XML 写回文件
    tree.write(xml_file)
def modify_xml_file_weight(xml_file):
    # 解析 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 删除带有 width="2.00" 属性的边
    edges_to_remove = []
    for edge in root.findall('edge'):
        for lane in edge.findall('lane'):
            if 'width' in lane.attrib and lane.attrib['width'] == '2.00':
                edges_to_remove.append(edge)
                break  # 只要找到一条带有指定属性的 lane，就可以删除对应的 edge
    for edge in edges_to_remove:
        root.remove(edge)

    # 将修改后的 XML 写回文件
    tree.write(xml_file)

# 测试函数
modify_xml_file_pedestrian('chengdu.net.xml')
