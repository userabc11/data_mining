import json
import time
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

# 修改为你的 folder path
folder_path = r"C:\develop\data\10G_data_new\part-00000.parquet"
parquet_file = pq.ParquetFile(folder_path)
data = parquet_file.read().to_pandas()
data = data["purchase_history"]
data = data.head(500)
if isinstance(data.iloc[0], str):
    data = data.apply(json.loads)

# 展开成新的 DataFrame
df_flat = pd.json_normalize(data)
df_flat["item_ids"] = df_flat["items"].apply(
    lambda x: ",".join(str(item["id"]) for item in x) if isinstance(x, list) else None
)
df_flat.drop(columns=["items"], inplace=True)
print(df_flat.head(5))

# 任务一
# df_orders = df_flat.copy()  # 如果每行是一个商品（有多个 item_ids 展开了）
# df_orders["order_id"] = df_orders.index  # 若没有订单号，用 index 模拟
#
# # 聚合成事务（每笔订单包含的类别集合）
# order_category_map = df_orders.groupby("order_id")["categories"].apply(lambda x: list(set(x))).tolist()
# # 转换为布尔矩阵
# te = TransactionEncoder()
# te_ary = te.fit(order_category_map).transform(order_category_map)
# df_trans = pd.DataFrame(te_ary, columns=te.columns_)
# frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
#
#
# # 打印类别之间的共购规则
# print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values(by="confidence", ascending=False).head(10))
# # 转换为布尔矩阵
# te = TransactionEncoder()
# te_ary = te.fit(order_category_map).transform(order_category_map)
# df_trans = pd.DataFrame(te_ary, columns=te.columns_)


# 任务2
# # 准备每条记录的 [支付方式, 商品类别 ] 作为一个“事务”
# transactions = df_flat[["payment_method", "categories"]].dropna().values.tolist()
# transactions = [[str(i) for i in row] for row in transactions]  # 转换为字符串列表
#
# # 转为布尔矩阵
# te = TransactionEncoder()
# te_ary = te.fit(transactions).transform(transactions)
# df_trans = pd.DataFrame(te_ary, columns=te.columns_)
# frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
#
# # 只保留“支付方式” → “商品类别”的规则
# rules_filtered = rules[
#     rules["antecedents"].apply(lambda x: any(pm in x for pm in df_flat["payment_method"].unique()))
#     & rules["consequents"].apply(lambda x: any(cat in x for cat in df_flat["categories"].unique()))
# ]
#
# # 打印结果
# rules_sorted = rules_filtered.sort_values(by=["support", "confidence"], ascending=False)
# print(rules_filtered[["antecedents", "consequents"]])

# 任务三
# def preprocess_data(df):
#     # 转换日期为季度
#     df['quarter'] = pd.to_datetime(df['purchase_date']).dt.quarter
#     df['quarter'] = 'Q' + df['quarter'].astype(str)
#
#     return df
#
#
# df_processed = preprocess_data(df_flat)
#
# # 准备Apriori算法的输入数据
# # 按季度分组，收集所有item_ids
# quarter_items = df_processed.groupby('quarter')['item_ids'].sum()
#
# # 转换为交易列表格式
# transactions = quarter_items.tolist()
#
# # 使用TransactionEncoder进行编码
# te = TransactionEncoder()
# te_ary = te.fit(transactions).transform(transactions)
# df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
#
# # 应用Apriori算法
# frequent_itemsets = apriori(df_encoded, min_support=0.00001, use_colnames=True)
#
# # 生成关联规则
# rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.00001)
#
# # 筛选出包含季度的规则
# quarter_rules = rules[
#     rules['antecedents'].apply(lambda x: any(item.startswith('Q') for item in x)) |
#     rules['consequents'].apply(lambda x: any(item.startswith('Q') for item in x))
#     ]
#
# # 打印结果
# print("频繁项集:")
# print(frequent_itemsets)
# print("\n关联规则(包含季度的):")
# print(quarter_rules.sort_values('lift', ascending=False))

# 筛选已退款或部分退款的订单
refund_df = df_flat['order_status'].isin(['已退款', '部分退款'])

# 拆分 item_ids 成为 list
refund_df['item_list'] = refund_df['item_ids'].apply(lambda x: x.split(','))

# 构建事务列表
transactions = refund_df['item_list'].tolist()

# 使用 TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 使用 Apriori 找出频繁项集
frequent_itemsets = apriori(df_encoded, min_support=0.0005, use_colnames=True)

# 生成关联规则（可选）
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.0005)

# 查看频繁组合
print(frequent_itemsets.sort_values('support', ascending=False))

