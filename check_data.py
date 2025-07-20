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
    # 使用DFS检测环
    def find_cycle(graph, start):
        visited = set()
        stack = []
        path = []

        def dfs(node):
            if node in path:
                return path[path.index(node):]  # 找到环
            if node in visited:
                return None
            visited.add(node)
            path.append(node)
            for neighbor in graph.get(node, []):
                cycle = dfs(neighbor)
                if cycle is not None:
                    return cycle
            path.pop()
            return None

        return dfs(start)

    # 构建图
    graph = {}
    for src, relations in FKtoKDict.items():
        for relation in relations:
            _, target, _ = relation
            if src not in graph:
                graph[src] = []
            graph[src].append(target)

    # 检测环
    for node in list(graph.keys()):
        cycle = find_cycle(graph, node)
        if cycle:
            print("检测到环:", cycle)

            # 打破环：移除环中的第一条边
            first_edge_from = cycle[0]
            first_edge_to = cycle[1]

            print(f"移除边: {first_edge_from} -> {first_edge_to}")

            # 在 FKtoKDict 中找到并删除这条边
            removed = False
            for table_name in list(FKtoKDict.keys()):
                relations = FKtoKDict[table_name]
                new_relations = []
                for rel in relations:
                    fk_col, target_table, pk_col = rel
                    if fk_col == first_edge_from and target_table == first_edge_to:
                        print(f"移除 FK-PK 关系: {rel}")
                        removed = True
                        continue
                    new_relations.append(rel)
                FKtoKDict[table_name] = new_relations

            if not removed:
                print("error")

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
                        print(f"边表 {table_name} 未在schema中定义")
                        continue

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

    detect_and_break_cycles(FKtoKDict)

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
def build_graph(pattern_path,safebound_instance,result_path='res'):
    start_time = time.time()

    with open(pattern_path, 'r') as f:
        pattern = json.load(f)

    join_query = JoinQueryGraph()
    
    for vertex in pattern['vertices']:
        tag_id = str(vertex['tag_id'])
        label_id = str(vertex['label_id'])
        join_query.addAlias(label_id,f"tag_v{tag_id}")

        print(f"处理顶点: tag_id={tag_id}, label_id={label_id}")

        for edge in pattern['edges']:
            if str(edge['dst']) == tag_id or str(edge['src']) == tag_id:
                
                edge_label = str(edge['label_id'])
                edge_tag = str(edge['tag_id'])
                edge_tag = f"tag_e{edge_tag}"
                print(f"处理边：tag_id={edge_tag}, label_id={edge_label}")
                join_query.addAlias(f"0{edge_label}",edge_tag)
                join_query.addJoin(f"tag_v{tag_id}", label_id, edge_tag, label_id)

    join_query.buildJoinGraph()
    join_query.printJoinGraph()

    bound = safebound_instance.functionalFrequencyBound(join_query)
    print("Cardinality Bound: " + str(bound))
    # print("SafeBound Memory (kB): " + str(safebound_instance.memory()/1000))

    # 写入结果
    end_time = time.time()
    elapsed_time = end_time - start_time

    query_name = os.path.basename(pattern_path).replace('.json', '')

    result_df = pd.DataFrame([{
        "query_name": query_name,
        "execution_time_seconds": round(elapsed_time, 6),
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

    return safebound_instance, join_query
# 没加vertex的版本
def compute_true_cardinality(edge_tables):

    print(f"{list(edge_tables.keys())}")
    visited_edges = set()
    edge_names = list(edge_tables.keys())

    result = edge_tables[edge_names[0]].copy()
    visited_edges.add(edge_names[0])
    edge_names.remove(edge_names[0])
    print(f"add{edge_names[0]}")

    while edge_names:
        for e_name in edge_names:
            if e_name in visited_edges:
                continue

            current_table = edge_tables[e_name]

            for column in current_table.columns:
                if column in result.columns:
                    # 检测自环,单独处理
                    columns_list = current_table.columns.tolist()
                    parts = columns_list[1].split('_')
                    if len(parts) >= 2:
                        column_name = parts[0]
                        suffix = parts[-1]
                        if suffix.startswith("dup"):
                            result = result.merge(current_table, on = column_name)
                            result.drop(columns=[f"{column_name}_dup"], inplace=True)
                            tmp = pd.DataFrame()
                            tmp[column_name] = current_table[f"{column_name}_dup"]
                            tmp[f"{column_name}_dup"] = current_table[column_name]
                            result = result.merge(tmp, on = column_name)
                            result.drop(columns=[f"{column_name}_dup"], inplace=True)
                            visited_edges.add(e_name)
                            edge_names.remove(e_name)
                            break

                    result = result.merge(current_table)
                    print(f"add{e_name}")
                    visited_edges.add(e_name)
                    edge_names.remove(e_name)
                    break

    return len(result)

# def compute_true_cardinality(vertex_tables,edge_tables):

#     print(f"{list(edge_tables.keys())}")
#     visited_edges = set()
#     visited_vertexs = set()
#     edge_names = list(edge_tables.keys())
#     vertex_names = list(vertex_tables.keys())

#     stage = 1

#     current_stage_vertexs = set()
#     current_stage_edges = set()

#     result = vertex_tables[vertex_names[0]].copy()
#     visited_vertexs.add(vertex_names[0])
#     current_stage_vertexs.add(vertex_names[0])
#     vertex_names.remove(vertex_names[0])
#     print(f"add{vertex_names[0]}")

#     while edge_names or vertex_names:
#         print("111")
#         if stage == 1:
#             for e_name in edge_names:
#                 if e_name in visited_edges:
#                     continue

#                 current_table = edge_tables[e_name]
#                 print("A")
#                 for column in current_table.columns.tolist():
#                     if column in current_stage_vertexs:
#                         # 检测自环,单独处理
#                         columns_list = current_table.columns.tolist()
#                         parts = columns_list[1].split('_')
#                         if len(parts) >= 2:
#                             column_name = parts[0]
#                             suffix = parts[-1]
#                             if suffix.startswith("dup"):
#                                 result = result.merge(current_table, on = column_name)
#                                 result.drop(columns=[f"{column_name}_dup"], inplace=True)
#                                 tmp = pd.DataFrame()
#                                 tmp[column_name] = current_table[f"{column_name}_dup"]
#                                 tmp[f"{column_name}_dup"] = current_table[column_name]
#                                 result = result.merge(tmp, on = column_name)
#                                 result.drop(columns=[f"{column_name}_dup"], inplace=True)
#                                 visited_edges.add(e_name)
#                                 edge_names.remove(e_name)
#                                 break

#                         result = result.merge(current_table)
#                         print(f"add{e_name}")
#                         visited_edges.add(e_name)
#                         current_stage_edges.add(e_name)
#                         edge_names.remove(e_name)
#                         break
#             if vertex_names:
#                 current_stage_vertexs = set()
#                 stage = 2
#         else:
#             for edge_name in current_stage_edges:
#                 print("B")
#                 dst_column = edge_tables[edge_name].columns.tolist()[1]
#                 if dst_column in vertex_names and dst_column not in visited_vertexs:
#                     result = result.merge(vertex_tables[dst_column])
#                     visited_vertexs.add(dst_column)
#                     current_stage_vertexs.add(dst_column)
#                     vertex_names.remove(dst_column)

#             if edge_names:
#                 current_stage_vertexs = set()
#                 stage = 1

#     return len(result)



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
    data_dir = '/home/phy/lab/executing/pathce/datasets/ldbc/sf0.003'
    schema_path = '/home/phy/lab/executing/pathce/schemas/ldbc/ldbc_gcard_schema.json'
    pattern_path = '/home/phy/lab/executing/pathce/patterns/glogs'
    file_path = 'safebound_instance_lsqb.pkl'
    result_path = './res'

    start_time = time.time() 

    if not os.path.exists(file_path):
        print("构建 SafeBound 实例...")
        safebound_instance,vertex_tables,edge_tables = build_safebound(data_dir, schema_path,result_path)

        # print("True Cardinality: " + str(compute_true_cardinality(edge_tables)))

        with open('safebound_instance_lsqb.pkl', 'wb') as f:
            pickle.dump(safebound_instance, f)
        print("SafeBound 实例已保存至 safebound_instance.pkl")
    else:
        with open(file_path, 'rb') as f:
            safebound_instance = pickle.load(f)
        print("SafeBound 实例已从 safebound_instance.pkl 加载")        

    for file in os.listdir(pattern_path):
        if file.endswith('.json'):
            if file != "p2.json":
                continue
            print(f"running:{file}")
            query_name = file.replace('.json', '')
            build_graph(f"{pattern_path}/{query_name}.json",safebound_instance,result_path)

    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"总耗时: {elapsed_time:.4f} 秒")

    # 把结果排序
    sort_results_csv(os.path.join(result_path, "results.csv"))
