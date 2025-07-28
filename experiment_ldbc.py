import os,sys
import json
import pandas as pd
import networkx as nx
import pickle
import time
count = 0
# SafeBound
Safebound_dir = os.getcwd()
source_path = os.path.join(Safebound_dir, 'Source')
if source_path not in sys.path:
    sys.path.append(source_path)
from SafeBoundUtils import *
from JoinGraphUtils import *

def load_schema(schema_path):
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # vertex_labels和edge_labels映射
    vertex_labels = {v: str(k) for v, k in schema['vertex_labels'].items()}
    edge_labels = {v: str(k) for v, k in schema['edge_labels'].items()}
    
    return vertex_labels, edge_labels

def parse_edge_table_name(table_name):
    # 解析边表名，提取src&dst
    parts = table_name.split('_')
    if len(parts) >= 3:
        src_name = parts[0]
        dst_name = parts[-1]
        return src_name, dst_name
    
    print(f"边表名格式不合规: {table_name}")
    return None, None

# FKPK关系不能形成环
# 这里把FKPK关系当做有向图，检测环并打破环
def detect_and_break_cycles(FKtoKDict):
    def remove_cycles(graph):
        visited = set()
        stack = set()
        parent = {}

        def dfs(node):
            visited.add(node)
            stack.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    parent[neighbor] = node
                    dfs(neighbor)
                elif neighbor in stack and parent[node] != neighbor:
                    # 检测到环，移除环的最后一条边
                    graph[node].remove(neighbor)
                    graph[neighbor].remove(node)

                    # 在 FKtoKDict 中找到并删除这条边
                    removed = False
                    for u in list(FKtoKDict.keys()):
                        if u != node and u !=neighbor:
                            continue
                        relations = FKtoKDict[u]
                        for rel in relations:
                            _, v, _ = rel
                            if v == node or v == neighbor:
                                print(f"移除 FK-PK 关系: {u}: {rel}")
                                FKtoKDict[u].remove(rel)
                                removed = True
                                continue
                    if not removed:
                        print("error")

            stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                dfs(node)

        return graph

    # 构建图
    adjacency_list = {}
    for u, relations in FKtoKDict.items():
        for relation in relations:
            _, v, _ = relation
            if u not in adjacency_list:
                adjacency_list[u] = []
            if v not in adjacency_list:
                adjacency_list[v] = []
            adjacency_list[u].append(v)
            adjacency_list[v].append(u)

    print("原始图：", adjacency_list)
    adjacency_list = remove_cycles(adjacency_list)
    print("移除环后的图：", adjacency_list)  

    return FKtoKDict

def build_safebound(data_dir, schema_path,result_path):
    start_time = time.time() 
    # 加载schema
    vertex_labels, edge_labels = load_schema(schema_path)
    
    # 存着备用
    vertex_tables = {}
    edge_tables = {}
    
    # 准备输入
    tableDFs = []
    tableNames = []
    tableJoinCols = []
    filterColumns = []
    FKtoKDict = {}

    # 加载数据
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            table_name = file.replace('.csv', '')
            
            # 顶点表
            if '_' not in table_name:
                try:
                    df = pd.DataFrame()
                    tmp_df = pd.read_csv(os.path.join(data_dir, file))
                    
                    if vertex_labels.get(table_name) is None:
                        print(f"顶点表 {table_name} 未在schema中定义")
                        continue
                    
                    table_label = f"{str(vertex_labels[table_name])}"
                    df[table_label] = tmp_df["id"]
                    tableDFs.append(df)
                    tableNames.append(table_label)
                    col_label = str(vertex_labels[table_name])
                    tableJoinCols.append([col_label])
                    vertex_tables[table_label] = df
                    
                    filterColumns.append([])
                        
                    # print(f"已加载顶点表: {table_name} | 行数: {len(df)}")
                except Exception as e:
                    print(f"无法读取顶点文件 {file}: {e}")

            # 边表
            else:
                try:
                    df = pd.DataFrame()
                    tmp_df = pd.read_csv(os.path.join(data_dir, file))
                    src_name, dst_name = parse_edge_table_name(table_name)

                    if vertex_labels.get(src_name) is None or vertex_labels.get(dst_name) is None:
                        print(f"边表 {table_name} 中的src或dst标签未在schema中定义: {src_name}, {dst_name}")
                        continue
                    if edge_labels.get(table_name) is None:
                        # print(f"边表 {table_name} 未在schema中定义")
                        continue
                    print(f"table_name已加载{table_name}")

                    src_label = vertex_labels[src_name]
                    dst_label = vertex_labels[dst_name]
                    table_label = f"0{str(edge_labels[table_name])}"

                    df[src_label]=tmp_df["src"]
                    if src_label != dst_label:
                        df[dst_label]=tmp_df["dst"]
                    else:
                        df[f"{dst_label}_dup"]=tmp_df["dst"]
                        print(f"{df.columns}")
                    edge_tables[table_label] = df

                    tableDFs.append(df)
                    tableNames.append(table_label)
                    if src_label != dst_label:
                        tableJoinCols.append([src_label, dst_label])
                    else:
                        tableJoinCols.append([src_label, f"{dst_label}_dup"])

                    if src_label not in FKtoKDict:
                        FKtoKDict[src_label] = []
                    if dst_label not in FKtoKDict:
                        FKtoKDict[dst_label] = []

                    FKtoKDict[src_label].append([src_label, table_label , src_label])
                    if src_label != dst_label:# 不能有环，所以自环直接不处理
                        FKtoKDict[dst_label].append([dst_label, table_label, dst_label])
                    
                    # print(f"已加载边表: {table_name} | 行数: {len(df)}")
                except Exception as e:
                    print(f"无法读取边文件 {file}: {e}")
    
    # print(f"{tableDFs}")
    # print(f"tablenames: {tableNames}")
    # print(f"Vertex Table Join Columns: {tableJoinCols}")
    # print(f"{FKtoKDict}")

    # detect_and_break_cycles(FKtoKDict)

    safebound_instance = SafeBound(
            tableDFs=tableDFs,
            tableNames=tableNames,
            tableJoinCols=tableJoinCols,
            relativeErrorPerSegment=.01,
            FKtoKDict=FKtoKDict,
            numCores=12
        )
    
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"SafeBound构建耗时: {elapsed_time:.4f} 秒")  # 输出耗时

    # 准备输出
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_file = os.path.join(result_path, "results.csv")

    query_name = "safebound_init"

    result_df = pd.DataFrame([{
        "query_name": query_name,
        "execution_time_seconds": round(elapsed_time, 6),
        "cardinality_bound": "-"
    }])

    # 如果文件不存在则创建并写入 header，否则以追加模式写入
    file_exists = os.path.isfile(result_file)

    result_df.to_csv(
        result_file,
        mode='a',
        header=not file_exists,
        index=False
    )

    return safebound_instance, vertex_tables, edge_tables
def build_graph(pattern_path,safebound_instance):
    with open(pattern_path, 'r') as f:
        pattern = json.load(f)

    join_query = JoinQueryGraph()
    
    for vertex in pattern['vertices']:
        tag_id = str(vertex['tag_id'])
        label_id = str(vertex['label_id'])
        join_query.addAlias(label_id,f"tag_v{tag_id}")

        #print(f"处理顶点: tag_id={tag_id}, label_id={label_id}")

        for edge in pattern['edges']:
            if str(edge['src']) == tag_id:
                edge_label = str(edge['label_id'])
                edge_tag = str(edge['tag_id'])
                edge_tag = f"tag_e{edge_tag}"
                #print(f"处理边：tag_id={edge_tag}, label_id={edge_label}")
                join_query.addAlias(f"0{edge_label}",edge_tag)

                join_query.addJoin(f"tag_v{tag_id}", label_id, edge_tag, label_id)
            elif str(edge['dst']) == tag_id:

                edge_label = str(edge['label_id'])
                edge_tag = str(edge['tag_id'])
                edge_tag = f"tag_e{edge_tag}"
                #print(f"处理边：tag_id={edge_tag}, label_id={edge_label}")
                join_query.addAlias(f"0{edge_label}",edge_tag)

                join_query.addJoin(edge_tag, label_id, f"tag_v{tag_id}", label_id)

    join_query.buildJoinGraph()
    #join_query.printJoinGraph()

    bound = safebound_instance.functionalFrequencyBound(join_query)
    print("Cardinality Bound: " + str(bound))
    # print("SafeBound Memory (kB): " + str(safebound_instance.memory()/1000))

    return bound

def sort_results_csv(result_file='res/results.csv'):
    if not os.path.exists(result_file):
        print(f"{result_file} 不存在")
        return

    results_df = pd.read_csv(result_file)

    # 对query名排序
    results_df['query_name'] = pd.Categorical(
        results_df['query_name'],
        categories=sorted(results_df['query_name'].unique(), key=lambda x: x)
    )
    sorted_df = results_df.sort_values(by='query_name').reset_index(drop=True)

    sorted_df.to_csv(result_file, index=False)
    print(f"{result_file} 已排序")

if __name__ == "__main__":
    # 配置路径
    data_dir = '/home/phy/lab/executing/pathce/datasets/ldbc/sf1'
    schema_path = '/home/phy/lab/executing/pathce/schemas/ldbc/ldbc_gcard_schema.json'
    pattern_path = '/home/phy/lab/executing/pathce/patterns/lsqb'
    file_path = 'safebound_instance_lsqb_3path.pkl'
    result_path = 'res/3pathtest'

    start_time = time.time() 

    if not os.path.exists(file_path) or True:
        print("构建 SafeBound 实例...")
        safebound_instance,vertex_tables,edge_tables = build_safebound(data_dir, schema_path,result_path)

        # print("True Cardinality: " + str(compute_true_cardinality(edge_tables)))

        with open('safebound_instance_lsqb_3path.pkl', 'wb') as f:
            pickle.dump(safebound_instance, f)
        print("SafeBound 实例已保存至 safebound_instance.pkl")
    else:
        with open(file_path, 'rb') as f:
            safebound_instance = pickle.load(f)
        print("SafeBound 实例已从 safebound_instance.pkl 加载")        

    for file in os.listdir(pattern_path):
        if file.endswith('q6.json'):
            print(f"running:{file}")
            query_name = file.replace('.json', '')

            start_time1 = time.time()
            for i in range(0,5):
                bound = build_graph(f"{pattern_path}/{query_name}.json",safebound_instance)
            
            # 写入结果
            end_time1 = time.time()
            elapsed_time1 = (end_time1 - start_time1)/5
            result_df = pd.DataFrame([{
                "query_name": query_name,
                "execution_time_seconds": round(elapsed_time1, 6),
                "cardinality_bound": bound
            }])

            result_file = os.path.join(result_path, "results.csv")
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            # 如果文件不存在则创建并写入 header，否则以追加模式写入
            file_exists = os.path.isfile(result_file)

            result_df.to_csv(
                result_file,
                mode='a',
                header=not file_exists,
                index=False
            )

    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"总耗时: {elapsed_time:.4f} 秒")

    # 把结果排序
    sort_results_csv(os.path.join(result_path, "results.csv"))
