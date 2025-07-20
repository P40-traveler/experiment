import networkx as nx
import matplotlib.pyplot as plt

# 定义邻接表
adjacency_list = {
    'Company': ['Company_isLocatedIn_Country', 'Person_workAt_Company'],
    'Company_isLocatedIn_Country': ['Company', 'Country'],
    'Person_workAt_Company': ['Company', 'Person'],
    'Country': ['Company_isLocatedIn_Country', 'City_isPartOf_Country', 'Comment_isLocatedIn_Country', 'Country_isPartOf_Continent', 'Post_isLocatedIn_Country'],
    'City_isPartOf_Country': ['Country', 'City'],
    'Comment_isLocatedIn_Country': ['Country', 'Comment'],
    'Country_isPartOf_Continent': ['Country', 'Continent'],
    'Post_isLocatedIn_Country': ['Country', 'Post'],
    'Tag': ['Tag_hasType_TagClass', 'Forum_hasTag_Tag', 'Post_hasTag_Tag', 'Comment_hasTag_Tag', 'Person_hasInterest_Tag'],
    'Tag_hasType_TagClass': ['Tag', 'TagClass'],
    'Forum_hasTag_Tag': ['Tag', 'Forum'],
    'Post_hasTag_Tag': ['Tag', 'Post'],
    'Comment_hasTag_Tag': ['Tag', 'Comment'],
    'Person_hasInterest_Tag': ['Tag', 'Person'],
    'TagClass': ['Tag_hasType_TagClass', 'TagClass_isSubclassOf_TagClass'],
    'TagClass_isSubclassOf_TagClass': ['TagClass'],
    'Forum': ['Forum_hasModerator_Person', 'Forum_hasTag_Tag', 'Forum_containerOf_Post', 'Forum_hasMember_Person'],
    'Forum_hasModerator_Person': ['Forum', 'Person'],
    'Forum_containerOf_Post': ['Forum', 'Post'],
    'Forum_hasMember_Person': ['Forum', 'Person'],
    'Person': ['Forum_hasModerator_Person', 'Person_knows_Person', 'Person_workAt_Company', 'Person_likes_Post', 'Comment_hasCreator_Person', 'Forum_hasMember_Person', 'Person_isLocatedIn_City', 'Person_hasInterest_Tag', 'Person_likes_Comment', 'Post_hasCreator_Person', 'Person_studyAt_University'],
    'Person_knows_Person': ['Person'],
    'Person_likes_Post': ['Person', 'Post'],
    'Comment_hasCreator_Person': ['Person', 'Comment'],
    'Person_isLocatedIn_City': ['Person', 'City'],
    'Person_likes_Comment': ['Person', 'Comment'],
    'Post_hasCreator_Person': ['Person', 'Post'],
    'Person_studyAt_University': ['Person', 'University'],
    'University': ['University_isLocatedIn_City', 'Person_studyAt_University'],
    'University_isLocatedIn_City': ['University', 'City'],
    'City': ['University_isLocatedIn_City', 'City_isPartOf_Country', 'Person_isLocatedIn_City'],
    'Comment': ['Comment_replyOf_Comment', 'Comment_hasTag_Tag', 'Comment_hasCreator_Person', 'Comment_isLocatedIn_Country', 'Person_likes_Comment', 'Comment_replyOf_Post'],
    'Comment_replyOf_Comment': ['Comment'],
    'Comment_replyOf_Post': ['Comment', 'Post'],
    'Post': ['Post_hasTag_Tag', 'Person_likes_Post', 'Forum_containerOf_Post', 'Post_isLocatedIn_Country', 'Post_hasCreator_Person', 'Comment_replyOf_Post'],
    'Continent': ['Country_isPartOf_Continent']
}

# 创建图
G = nx.Graph()

# 添加节点和边
for node, neighbors in adjacency_list.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# 绘制图
plt.figure(figsize=(20, 15))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=15, font_weight='bold')
plt.title("Graph Visualization")
plt.show()