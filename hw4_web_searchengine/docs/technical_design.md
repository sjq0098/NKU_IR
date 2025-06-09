# 南翎搜索技术设计文档

## 1. 系统架构

### 1.1 前端技术栈
- 原生HTML、CSS、JavaScript构建用户界面
- 基于模板引擎Jinja2进行页面渲染
- 采用AJAX技术实现搜索建议和历史记录的异步加载

### 1.2 后端技术栈
- Python Flask框架作为Web服务器
- Flask-Login实现用户认证系统
- SQLite作为主数据库，存储用户信息、搜索历史等
- Whoosh搜索引擎框架用于构建全文索引
- Jieba中文分词器用于中文分词处理

```
[前端层]
    HTML/CSS/JS → Jinja2模板引擎 → AJAX异步加载
        ↓
[应用层]
    Flask Web服务 → 查询处理 → 结果呈现
        ↓
[核心层]
    爬虫模块 → Whoosh搜索引擎 → 混合排名模块（自然语言处理+PageRank等） → 个性化推荐
        ↓
[存储层]
    SQLite数据库 → Whoosh索引 → 网页快照
```

## 2. 爬虫模块

### 2.1 实现方案
- Scrapy框架构建分布式爬虫
- 布隆过滤器实现高效URL去重
- 支持增量爬取和断点续传
- 遵循robots.txt协议的礼貌爬取

### 2.2 数据处理流程
- 网页提取：标题、正文、发布时间、元数据
- 链接关系提取：构建网页链接图
- 文档解析：支持PDF、Word、Excel等格式
- 数据清洗：HTML标签过滤、特殊字符处理
- 数据存储：将处理后的数据存入SQLite和Whoosh索引

## 3. Whoosh索引模块

### 3.1 索引设计
```python
# 自定义中文分析器
class ChineseAnalyzer(Analyzer):
    def __call__(self, text, **kargs):
        words = jieba.cut(text)
        token = Token()
        for i, word in enumerate(words):
            if word.strip():
                token.text = word
                token.pos = i
                token.positions = i
                token.startchar = 0
                token.endchar = len(word)
                yield token

# 索引结构定义
schema = Schema(
    url=ID(stored=True, unique=True),
    title=TEXT(stored=True, analyzer=ChineseAnalyzer()),
    content=TEXT(stored=True, analyzer=ChineseAnalyzer()),
    anchor_text=TEXT(stored=True, analyzer=ChineseAnalyzer()),
    publish_time=DATETIME(stored=True),
    file_type=KEYWORD(stored=True, commas=True),
    page_rank=NUMERIC(stored=True, type=float),
    outlinks=KEYWORD(stored=True, commas=True),
    inlinks=KEYWORD(stored=True, commas=True),
    snapshot_path=ID(stored=True)
)
```

### 3.2 索引优化
- 索引分片：根据网站域名或内容类型分割索引
- 增量更新：支持实时添加新文档
- 中文分词：集成jieba分词器处理中文特性
- 内存管理：平衡索引大小和性能

## 4. 搜索模块

### 4.1 查询类型
- 全文检索：基于相关性评分
- 精确匹配：短语和关键词精确查询
- 通配符查询：支持*和?通配符
- 文档类型过滤：针对不同文件格式（PDF、DOC、DOCX、XLS、XLSX等）
- 时间范围查询：基于发布时间

### 4.2 查询实现
```python
# 基本搜索实现
def search(query_text, fields=None, filter_dict=None, page=1, page_size=10):
    """
    执行搜索查询
    
    参数:
        query_text: 查询文本
        fields: 搜索字段权重，如 {"title": 3.0, "content": 1.0}
        filter_dict: 过滤条件，如 {"file_type": "pdf", "publish_time": (start_date, end_date)}
        page: 当前页码
        page_size: 每页结果数
    """
    ix = open_dir(INDEX_DIR)
    
    with ix.searcher() as searcher:
        # 构建查询对象
        query_fields = fields or {"title": 3.0, "content": 1.0, "anchor_text": 2.0}
        og = OrGroup.factory(0.9)  # 允许部分匹配
        
        # 创建多字段查询解析器
        parser = MultifieldParser(list(query_fields.keys()), ix.schema, 
                                 weights=query_fields, group=og)
        
        # 解析查询文本
        query = parser.parse(query_text)
        
        # 添加过滤器
        filter_queries = []
        if filter_dict:
            for field, value in filter_dict.items():
                if field == 'file_type':
                    filter_queries.append(Term('file_type', value))
                elif field == 'publish_time' and isinstance(value, tuple):
                    start, end = value
                    filter_queries.append(DateRange('publish_time', start, end))
        
        # 执行查询
        results = searcher.search_page(
            query, page, pagelen=page_size, 
            filter=And(filter_queries) if filter_queries else None,
            terms=True  # 返回匹配的词条，用于高亮显示
        )
        
        # 处理结果
        hits = []
        for hit in results:
            # 获取高亮片段
            highlights = hit.highlights("content", top=3)
            
            hits.append({
                "url": hit["url"],
                "title": hit["title"],
                "snippet": highlights or hit["content"][:200] + "...",
                "publish_time": hit["publish_time"],
                "file_type": hit["file_type"],
                "score": hit.score
            })
        
        return {
            "total": len(results),
            "page": page,
            "page_size": page_size,
            "hits": hits
        }
```

## 5. 排名算法

### 5.1 混合排名框架
- 多因子混合排名：融合内容相关性、链接分析和用户行为

### 5.2 内容相关性（Whoosh Score）
- 基于BM25F算法：考虑词频、字段权重和文档长度
- 字段权重：标题>锚文本>正文
- 自定义评分：整合其他因素

### 5.3 链接分析（PageRank）
```python
def compute_pagerank(link_graph, d=0.85, iterations=50):
    """
    计算网页PageRank值
    
    参数:
        link_graph: 链接关系图
        d: 阻尼系数
        iterations: 迭代次数
    """
    nodes = len(link_graph)
    M = np.zeros((nodes, nodes))
    
    # 构建转移矩阵
    for i, node in enumerate(link_graph):
        outlinks = len(link_graph[node])
        if outlinks > 0:
            for outlink in link_graph[node]:
                j = list(link_graph.keys()).index(outlink)
                M[j, i] = 1.0 / outlinks
    
    # 迭代计算
    v = np.ones(nodes) / nodes
    for _ in range(iterations):
        v_new = (1-d)/nodes + d * M.dot(v)
        # 检查收敛
        if np.linalg.norm(v_new - v) < 1e-6:
            break
        v = v_new
    
    # 构建结果字典
    pagerank = {}
    for i, node in enumerate(link_graph):
        pagerank[node] = v[i]
    
    return pagerank
```

## 6. SQLite数据库设计

### 6.1 数据库表设计

#### users 表（用户基本信息）
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_login TEXT,
    status TEXT DEFAULT 'active',
    avatar_path TEXT
);
```

#### user_profiles 表（用户身份信息）
```sql
CREATE TABLE user_profiles (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    role TEXT,                 -- 角色（本科生/研究生/教师等）
    department TEXT,           -- 院系
    major TEXT,                -- 专业
    interests TEXT,            -- 兴趣标签（逗号分隔）
    preferences TEXT,          -- 偏好设置（JSON格式）
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

#### search_history 表（搜索历史）
```sql
CREATE TABLE search_history (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,           -- 关联users表，匿名用户为NULL
    query TEXT NOT NULL,       -- 查询文本
    filters TEXT,              -- 过滤条件（JSON格式）
    timestamp TEXT NOT NULL,   -- 搜索时间
    results_count INTEGER,     -- 结果数量
    session_id TEXT,           -- 会话ID
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

#### click_events 表（点击记录）
```sql
CREATE TABLE click_events (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    search_id INTEGER,         -- 关联search_history表
    url TEXT NOT NULL,         -- 点击的URL
    position INTEGER,          -- 结果位置
    timestamp TEXT NOT NULL,   -- 点击时间
    dwell_time INTEGER,        -- 停留时间（秒）
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (search_id) REFERENCES search_history(id)
);
```

#### news 表（新闻内容）
```sql
CREATE TABLE news (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    publish_time TEXT,
    source TEXT,
    url TEXT UNIQUE,
    category TEXT,
    keywords TEXT,           -- 关键词（逗号分隔）
    snapshot_id INTEGER,     -- 关联网页快照
    FOREIGN KEY (snapshot_id) REFERENCES web_snapshots(id)
);
```

#### documents 表（各类文档数据）
```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    file_type TEXT,          -- 文件类型（PDF/DOC/XLS等）
    file_path TEXT,          -- 文件存储路径
    upload_time TEXT,
    source_url TEXT,
    author TEXT,
    created_date TEXT,
    modified_date TEXT,
    size INTEGER
);
```

#### web_snapshots 表（网页快照）
```sql
CREATE TABLE web_snapshots (
    id INTEGER PRIMARY KEY,
    url TEXT,                -- 原始URL
    html_content TEXT,       -- HTML内容
    capture_time TEXT,       -- 捕获时间
    version INTEGER          -- 版本号
);
```

### 6.2 数据库索引设计
```sql
-- users表索引
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- search_history表索引
CREATE INDEX idx_search_history_user ON search_history(user_id);
CREATE INDEX idx_search_history_query ON search_history(query);
CREATE INDEX idx_search_history_time ON search_history(timestamp);

-- news表索引
CREATE VIRTUAL TABLE news_fts USING fts5(title, content);
CREATE INDEX idx_news_publish_time ON news(publish_time);
CREATE INDEX idx_news_category ON news(category);

-- documents表索引
CREATE VIRTUAL TABLE documents_fts USING fts5(title, content);
CREATE INDEX idx_documents_file_type ON documents(file_type);
```

## 7. 个性化推荐系统

### 7.1 用户模型
- 基于SQLite存储的用户数据构建用户画像
- 主要特征：
  - 基本特征：角色（本科生/研究生/教师）、院系、专业
  - 行为特征：搜索历史、点击行为、停留时间
  - 兴趣特征：从搜索内容提取的主题兴趣

### 7.2 推荐算法
```python
def personalize_results(results, user_id, conn):
    """
    基于用户画像个性化排序搜索结果
    
    参数:
        results: 原始搜索结果
        user_id: 用户ID
        conn: SQLite连接
    """
    if not user_id:
        return results  # 匿名用户使用原始排序
    
    cursor = conn.cursor()
    
    # 获取用户信息
    cursor.execute(
        "SELECT u.role, p.department FROM users u JOIN user_profiles p ON u.id = p.user_id WHERE u.id = ?", 
        (user_id,)
    )
    user_info = cursor.fetchone()
    
    if not user_info:
        return results  # 用户不存在或没有个人资料
        
    role, department = user_info
    
    # 用户角色权重
    role_weights = {
        "undergraduate": {"title": 3.0, "content": 1.0, "publish_time": 2.0},
        "graduate": {"title": 2.0, "content": 2.0, "publish_time": 1.0},
        "teacher": {"title": 1.0, "content": 3.0, "publish_time": 0.5}
    }
    
    weights = role_weights.get(role, role_weights["undergraduate"])
    
    # 获取用户历史兴趣
    cursor.execute(
        "SELECT query FROM search_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10", 
        (user_id,)
    )
    search_queries = cursor.fetchall()
    search_history = [query[0] for query in search_queries]
    
    # 提取用户历史兴趣关键词
    interest_keywords = extract_interests_from_history(search_history)
    
    # 计算个性化分数
    for result in results["hits"]:
        # 基础分数
        base_score = result["score"]
        
        # 1. 角色调整因子
        if "教学" in result["title"] and role == "teacher":
            base_score *= 1.2
        elif "课程" in result["title"] and role in ["undergraduate", "graduate"]:
            base_score *= 1.1
            
        # 2. 院系相关性
        if department and department in result["title"]:
            base_score *= 1.3
            
        # 3. 兴趣匹配度
        interest_score = 0.0
        for keyword in interest_keywords:
            if keyword in result["title"]:
                interest_score += 0.05
            if keyword in result["snippet"]:
                interest_score += 0.02
                
        # 更新最终分数
        result["personalized_score"] = base_score * (1 + interest_score)
    
    # 按个性化分数重新排序
    results["hits"].sort(key=lambda x: x.get("personalized_score", 0.0), reverse=True)
    return results
```

### 7.3 实时推荐功能
- 搜索联想：基于用户历史搜索和热门查询
- 热门推荐：校内热点内容推荐
- 相关文档：基于当前文档特征推荐相似内容

### 7.4 推荐数据流
- 数据收集：用户交互数据存入SQLite
- 特征提取：定期分析用户行为形成特征向量
- 模型更新：实时更新用户兴趣模型
- 结果过滤：根据用户模型过滤和排序搜索结果

## 8. 系统性能优化

### 8.1 缓存策略
- 使用基于内存的缓存存储热门搜索结果
- LRU (Least Recently Used) 缓存置换策略
- 定期更新缓存保持数据新鲜度

### 8.2 并发处理
- 异步索引更新
- 查询请求的并行处理
- 数据库连接池优化

### 8.3 前端优化
- 懒加载减少初始加载时间
- 前端分页减少数据传输量
- AJAX异步加载提升响应速度

## 9. 系统实现路线图

1. 环境搭建与依赖安装
2. SQLite数据库设计与实现
3. Whoosh索引模块实现
4. 爬虫系统开发与数据采集
5. 搜索功能实现与优化
6. Hummingbird算法实现
7. 利用知识图谱进行优化一些搜索
8. 用户系统与登录认证开发
9. 个性化推荐算法实现
10. Web界面设计与开发
11. 系统集成测试
12. 性能优化与部署上线 

   混合爬虫(mixed_crawler.py) 
   → 存储原始数据(webpages.db) 
   → 数据清洗(data_cleaning.py) 
   → 存储清洗数据(cleaned_data.db)
   → PageRank计算(pagerank_calculator.py) 
   → 存储PageRank值(pagerank表)
   → 索引构建(index_builder.py) 
   → 搜索服务(search_engine.py)





![alt text](<屏幕截图 2025-05-12 125735.png>)

![alt text](<屏幕截图 2025-05-13 014630.png>)

![alt text](<屏幕截图 2025-05-19 132957.png>)