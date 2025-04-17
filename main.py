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
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False


def read_and_concat_parquet(folder_path):
    """
    读取文件夹内所有Parquet文件并纵向拼接

    参数:
        folder_path (str): Parquet文件所在文件夹路径

    返回:
        pd.DataFrame: 合并后的DataFrame
    """
    # 获取文件夹中所有.parquet文件
    parquet_files = [f for f in os.listdir(folder_path)
                     if f.endswith('.parquet')]

    if not parquet_files:
        raise ValueError(f"文件夹 {folder_path} 中未找到.parquet文件")

    # 初始化空列表存储DataFrame
    dfs = []

    # 带进度条读取文件
    for file in parquet_file:
        file_path = os.path.join(folder_path, file)
        try:
            # 使用PyArrow高效读取
            table = pq.read_table(file_path)
            dfs.append(table.to_pandas())
        except Exception as e:
            print(f"\n警告: 文件 {file} 读取失败 - {str(e)}")
            continue

    # 纵向拼接所有DataFrame
    if dfs:
        combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        print(f"\n合并完成: 共处理 {len(dfs)} 个文件, 总行数: {len(combined_df):,}")
        return combined_df
    else:
        raise ValueError("没有有效的Parquet文件可合并")


# 使用示例



def analyze_complete_rows(df):
    # 标记每行是否有任何缺失值
    rows_with_missing = df.isnull().any(axis=1)

    # 统计结果
    missing_rows_count = rows_with_missing.sum()
    missing_rows_percent = (missing_rows_count / len(df)) * 100

    # 返回统计结果
    return missing_rows_count, missing_rows_percent

# 修改为你的 folder path
folder_path = r""

try:
    # 读取并合并数据
    data = read_and_concat_parquet(folder_path)

    # 显示合并后的数据概览
    print("\n合并后数据概览:")
    print(data.info())
    print("\n前5行示例:")
    print(data.head())

except Exception as e:
    print(f"发生错误: {str(e)}")

parquet_file = pq.ParquetFile(path_01)
data = parquet_file.read().to_pandas()

print("表头信息")
print(data.columns)
a = data.iloc[0,:]

# 执行分析
missing_rows_count, missing_rows_percent = analyze_complete_rows(data)

# 输出结果
print("数据质量分析 - 行级缺失统计")
print("=" * 50)
print(f"总行数: {len(data)}")
print(f"包含缺失值的行数: {missing_rows_count}")
print(f"包含缺失值的行比例: {missing_rows_percent:.2f}%")
print("=" * 50)

abnormal_conditions = (
    # 基本无效数据
        (data['age'] < 0) |  # 年龄为负数
        (data['age'] > 100) |  # 年龄超过人类合理寿命
        (data['income'] < 0) |  # 收入为负数
        (data['income'] == 0) |  # 零收入（除非是失业人员）

        # 年龄-收入矛盾组合
        ((data['age'] < 18) & (data['income'] > 100000)) |  # 未成年人高收入
        ((data['age'] > 65) & (data['income'] > 800000))  # 老年高收入
)

# 计算不合理数据
abnormal_data = data[abnormal_conditions]
total_records = len(data)
abnormal_count = len(abnormal_data)
abnormal_ratio = (abnormal_count / total_records) * 100

# 输出统计结果
print("数据质量分析报告 - 不合理年龄/收入数据")
print("=" * 50)
print(f"总记录数: {total_records}")
print(f"不合理数据条数: {abnormal_count}")
print(f"不合理数据占比: {abnormal_ratio:.2f}%")
print("=" * 50)

# 输出不合理数据示例（最多显示5条）
if abnormal_count > 0:
    print("\n不合理数据示例（前5条）:")
    print(abnormal_data[['age', 'income', 'fullname', 'country', 'gender']].head())

# 选出与购买有关的列
columns_to_keep = [
    'id',               # Customer identifier (useful for tracking purchases)
    'age',              # Demographic factor that can influence purchases
    'income',           # Strongly related to purchasing power
    'gender',           # Demographic factor for purchase patterns
    'country',          # Geographic factor for purchase behavior
    'purchase_history'  # Direct record of purchases
]
data = data[columns_to_keep]


# 以地点 country进行分组处理并可视化：
country_counts = data['country'].value_counts().sort_values(ascending=False)
country_counts = country_counts * 7
print("根据国家进行统计：")
for country, count in country_counts.items():
    print(f"{country} 数量 {count}")

# 将国家名称转换为数字编码（0, 1, 2, ...）
data['country_code'], country_index = pd.factorize(data['country'])

# 查看编码映射关系（可选）
country_mapping = dict(zip(country_index, range(len(country_index))))
print("国家名称 -> 数字编码映射：")
for country, code in country_mapping.items():
    print(f"{country}: {code}")

income_max = data['income'].max()
income_min = data['income'].min()
income_mean = data['income'].mean()
income_var = data['income'].var()

# 输出结果（保留2位小数）
print(f"收入最大值: ¥{income_max:,.2f}")
print(f"收入最小值: ¥{income_min:,.2f}")
print(f"收入均值: ¥{income_mean:,.2f}")
print(f"收入方差: ¥{income_var:,.2f}")

# 统计各区间的频数
bins = np.arange(0, 1_100_000, 100_000)
income_distribution = pd.cut(
    data['income'],
    bins=bins,
    right=False,  # 左闭右开区间（如[0, 100k)）
    include_lowest=True  # 包含最小值（¥0.01）
).value_counts().sort_index()

# 格式化输出
print("收入区间分布统计（每10万元为一个区间）")
print("=" * 50)
for interval, count in income_distribution.items():
    lower = int(interval.left)
    upper = int(interval.right)
    count = count * 7
    print(f"¥{lower:>7,} ~ ¥{upper:>7,}: {count:>5} 人")

counts, bin_edges = np.histogram(data['income'], bins=bins)
counts = counts*7
plt.figure(figsize=(12, 6))
ax = sns.histplot(data['income'], bins=bins, kde=False, edgecolor='white')

# 添加数值标签
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10)

# 美化图表
ax.set_title('income range(100,000 per range)', fontsize=15)
ax.set_xlabel('income', fontsize=12)
ax.set_ylabel('num', fontsize=12)
ax.set_xticks(bin_edges)
ax.set_xticklabels([f'¥{int(x/10000)}' for x in bin_edges], rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

def extract_purchase_info(json_str):
    try:
        data = json.loads(json_str)
        return pd.Series({
            'avg_price': data.get('avg_price'),
            'purchase_categories': data.get('categories')
        })
    except:
        return pd.Series({
            'avg_price': None,
            'purchase_categories': None
        })
# Now filter the DataFrame
start_time = time.time()
data = data.sample(n=500000, random_state=42)
#data = data.head(500000) (5625000/50000)*7
purchase_features = data['purchase_history'].apply(extract_purchase_info)
purchase_data = pd.concat([data, purchase_features], axis=1)
end_time = time.time()
print(f"数据处理用时{(end_time-start_time)*(5625000/500000)*7*30}")
t = purchase_data.iloc[0, :]
print(t)

categories = purchase_data['purchase_categories'].unique()
print("All Purchase Categories:")
for i, cat in enumerate(categories, 1):
    print(f"{i}. {cat}")

# 2. Calculate category distribution
category_counts = purchase_data['purchase_categories'].value_counts()

# 3. Create pie chart visualization
plt.figure(figsize=(8, 8))
patches, texts, autotexts = plt.pie(
    category_counts,
    labels=category_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.Pastel1.colors,
    wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
)

# Customize appearance
plt.setp(texts, size=10, weight='bold')
plt.setp(autotexts, size=8, color='black')
plt.title('Purchase Categories Distribution', pad=20, size=14)

# Equal aspect ratio ensures pie is drawn as circle
plt.axis('equal')
plt.tight_layout()
plt.show()

# 按大类分组
category_groups = {
    '食品类': ['零食', '水产', '肉类', '饮料', '米面', '蛋奶', '水果', '蔬菜', '调味品'],
    '穿戴类': ['手套', '裙子', '帽子', '围巾', '外套', '鞋子', '裤子', '内衣', '上衣'],
    '电子产品': ['耳机', '智能手表', '音响', '摄像机', '笔记本电脑', '游戏机', '平板电脑',
               '智能手机', '相机', '车载电子'],
    '家居用品': ['家具', '床上用品', '卫浴用品', '厨具'],
    '母婴用品': ['婴儿用品', '儿童课外读物', '益智玩具'],
    '文体用品': ['办公用品', '文具', '模型', '健身器材'],
    '其他': ['户外装备', '汽车装饰', '玩具']
}

# 计算各大类总和
group_counts = {}
for group, items in category_groups.items():
    mask = [c in items for c in categories]
    group_counts[group] = sum(np.array(category_counts)[mask])

# 绘制饼图
plt.figure(figsize=(10, 8))
wedges, texts, autotexts = plt.pie(
    group_counts.values(),
    labels=group_counts.keys(),
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.tab20.colors,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    textprops={'fontsize': 12}
)

# 美化标签
plt.setp(autotexts, size=10, weight='bold', color='white')
plt.title('购买类别大类分布', pad=20, fontsize=16)

# 添加图例
plt.legend(
    wedges,
    [f"{k}" for k, v in group_counts.items()],
    title="类别明细",
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

plt.tight_layout()
plt.show()

purchase_data['price_income_ratio'] = purchase_data['avg_price'] / purchase_data['income']

# 统计关键指标
ratio_stats = {
    'max': purchase_data['price_income_ratio'].max(),
    'min': purchase_data['price_income_ratio'].min(),
    'mean': purchase_data['price_income_ratio'].mean()
}

# 格式化输出结果
print("avg_price/income 比值统计结果：")
print("=" * 40)
print(f"最大值: {ratio_stats['max']:.4f} (用户单次消费最高占收入比例)")
print(f"最小值: {ratio_stats['min']:.4f} (用户单次消费最低占收入比例)")
print(f"平均值: {ratio_stats['mean']:.4f} (平均消费收入占比)")
print("=" * 40)

# 异常值检测（可选）
high_ratio = purchase_data[purchase_data['price_income_ratio'] > 10]  # 消费超过收入50%的用户
print(f"\n异常检测：有 {len(high_ratio)} 位用户单次消费超过其收入的1000%")

bins = [0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1, np.inf]
labels = [
    '微消费 (<0.1%)',
    '极低消费 (0.1-1%)',
    '低消费 (1-5%)',
    '中等消费 (5-10%)',
    '较高消费 (10-30%)',
    '高消费 (30-50%)',
    '超高消费 (50-100%)',
    '极端消费 (>100%)'
]

# 执行分类
purchase_data['消费等级'] = pd.cut(
    purchase_data['price_income_ratio'],
    bins=bins,
    labels=labels,
    right=False  # 左闭右开区间
)

# 统计结果
ratio_distribution = purchase_data['消费等级'].value_counts().sort_index()
ratio_percentage = (purchase_data['消费等级'].value_counts(normalize=True).sort_index() * 100).round(2)
plt.figure(figsize=(12, 6))

# 创建统计表格
dist_df = pd.DataFrame({
    '人数': ratio_distribution,
    '占比(%)': ratio_percentage,
    '区间说明': [
        "单次消费不足收入的千分之一",
        "单次消费约占收入0.1%-1%",
        "单次消费约占收入1%-5%",
        "单次消费约占收入5%-10%",
        "单次消费约占收入10%-30%",
        "单次消费约占收入30%-50%",
        "单次消费约占收入50%-100%",
        "单次消费超过收入100%"
    ]
})

# 绘制柱状图
bars = plt.bar(
    dist_df.index,
    dist_df['人数'],
    color='#4B96E9',
    edgecolor='white',
    linewidth=1,
    width=0.7
)

# 添加数据标签
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height,
#              f'{height:,}',
#              ha='center', va='bottom',
#              fontsize=10)

# 添加占比标签（右侧）
for i, (_, row) in enumerate(dist_df.iterrows()):
    plt.text(1.02, row['人数']*0.9,
             f"{row['占比(%)']}%",
             ha='left', va='center',
             fontsize=10,
             transform=plt.gca().get_yaxis_transform())

# 图表美化
plt.title('消费收入比分类人数分布', fontsize=16, pad=20)
plt.xlabel('消费等级', fontsize=12)
plt.ylabel('用户人数', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.3)

# 调整布局
plt.tight_layout()
plt.show()


# 年龄分组（自定义边界）
age_bins = [0, 18, 35, 55, 75, 100]
age_labels = ['<18', '18-34', '35-54', '55-74', '75+']
purchase_data['age_group'] = pd.cut(purchase_data['age'], bins=age_bins, labels=age_labels)

# 消费分级（基于分位数）
price_percentiles = purchase_data['avg_price'].quantile([0, 0.3, 0.7, 0.95, 1]).values
price_labels = ['低消费', '中等消费', '高消费', '超高消费']
purchase_data['price_level'] = pd.cut(purchase_data['avg_price'], bins=price_percentiles, labels=price_labels)

plt.figure(figsize=(10, 6))
age_price = purchase_data.groupby('age_group')['avg_price'].mean().sort_index()

# 绘制柱状图+趋势线
ax = age_price.plot(kind='bar', color='#66B2FF', edgecolor='white', width=0.7)
ax.plot(age_price.index, age_price.values, 'r--o', alpha=0.5)

# 添加标签
for i, v in enumerate(age_price):
    ax.text(i, v+50, f'{v:,.0f}', ha='center', fontsize=10)

plt.title('各年龄段平均消费对比', fontsize=14)
plt.xlabel('年龄组')
plt.ylabel('平均消费金额（¥）')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
cross_tab = pd.crosstab(purchase_data['age_group'],
                        purchase_data['price_level'],
                        normalize='index') * 100

sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='Blues',
            cbar_kws={'label': '占比(%)'}, linewidths=0.5)
plt.title('不同年龄段的消费等级分布', fontsize=14)
plt.xlabel('消费等级')
plt.ylabel('年龄组')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='age_group', y='avg_price', data=purchase_data,
            showfliers=False,  # 隐藏极端离群点
            palette='Pastel1')

# 叠加散点显示分布密度
sns.stripplot(x='age_group', y='avg_price', data=purchase_data,
              jitter=True, alpha=0.2, color='black', size=3)

plt.ylim(0, purchase_data['avg_price'].quantile(0.95))  # 聚焦95%数据
plt.title('各年龄段消费金额分布', fontsize=14)
plt.xlabel('年龄组')
plt.ylabel('消费金额（¥）')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()