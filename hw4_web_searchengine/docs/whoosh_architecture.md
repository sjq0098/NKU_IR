# Whoosh搜索引擎架构与集成方案

## 1. Whoosh简介

Whoosh是一个用纯Python实现的全文搜索引擎库，具有以下特点：

- **纯Python实现**：无需外部依赖，易于安装和使用
- **快速索引**：高效的索引创建和更新机制
- **灵活查询**：支持布尔操作、短语查询、通配符等
- **结果高亮**：支持查询结果高亮显示
- **可扩展**：允许自定义分析器、评分函数等

## 2. 核心组件

### 2.1 索引结构

Whoosh索引由以下核心组件组成：

- **Schema**：定义文档字段结构和类型
- **Storage**：负责索引数据的存储（默认为文件系统）
- **Analyzer**：将文本分解为tokens用于索引和搜索
- **Writer**：添加、更新和删除文档
- **Searcher**：执行搜索并返回结果

### 2.2 主要字段类型

Whoosh提供多种字段类型适应不同需求：

- **TEXT**：全文索引字段，支持分析器
- **ID**：不分词的标识符字段
- **KEYWORD**：简单分词的关键词字段
- **NUMERIC**：数值字段（整数、浮点数）
- **DATETIME**：日期时间字段
- **BOOLEAN**：布尔值字段
- **STORED**：仅存储不索引的字段

## 3. 中文分词集成

### 3.1 jieba分词器集成

```python
from whoosh.analysis import Analyzer, Token
import jieba

class ChineseAnalyzer(Analyzer):
    """基于jieba的中文分析器"""
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
```

### 3.2 分词优化

- **自定义词典**：使用jieba的自定义词典功能增强分词效果
- **停用词过滤**：去除常见停用词提高检索精度
- **同义词扩展**：支持同义词查询扩展

## 4. 索引管理

### 4.1 索引设计

```python
from whoosh.fields import Schema, TEXT, ID, KEYWORD, DATETIME, NUMERIC

# 定义索引结构
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

### 4.2 索引创建与更新

```python
from whoosh.index import create_in, open_dir
import os

# 创建索引目录
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

# 创建新索引
if not os.listdir(INDEX_DIR):
    ix = create_in(INDEX_DIR, schema)
else:
    ix = open_dir(INDEX_DIR)

# 添加文档
writer = ix.writer()
writer.add_document(
    url="http://example.com/page1",
    title="示例页面标题",
    content="这是一个示例页面的内容...",
    publish_time=datetime.now(),
    file_type="html"
)
writer.commit()

# 更新文档
writer = ix.writer()
writer.update_document(
    url="http://example.com/page1",  # 使用唯一字段更新
    title="更新后的标题",
    content="更新后的内容...",
    publish_time=datetime.now(),
    file_type="html"
)
writer.commit()
```

### 4.3 增量索引策略

```python
def incremental_index(document_source, existing_index):
    """增量索引策略"""
    writer = existing_index.writer()
    
    # 获取新文档
    for doc in document_source.get_new_documents():
        # 检查文档是否已存在
        with existing_index.searcher() as searcher:
            result = searcher.document(url=doc["url"])
            
            if result:
                # 文档已存在，检查是否需要更新
                if doc["last_modified"] > result["last_modified"]:
                    writer.update_document(**doc)
            else:
                # 新文档，直接添加
                writer.add_document(**doc)
    
    # 提交更改
    writer.commit()
```

## 5. 查询功能实现

### 5.1 基本搜索

```python
from whoosh.qparser import QueryParser

with ix.searcher() as searcher:
    query_parser = QueryParser("content", ix.schema)
    query = query_parser.parse("搜索关键词")
    results = searcher.search(query, limit=20)
    
    for hit in results:
        print(f"{hit['title']} - {hit.score}")
```

### 5.2 高级查询功能

```python
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.query import Term, And, DateRange

# 多字段查询
field_boosts = {"title": 3.0, "content": 1.0, "anchor_text": 2.0}
parser = MultifieldParser(list(field_boosts.keys()), ix.schema, weights=field_boosts)
query = parser.parse("南开大学")

# 过滤器
filter_query = And([
    Term("file_type", "pdf"),
    DateRange("publish_time", datetime(2023, 1, 1), datetime(2023, 12, 31))
])

# 执行搜索
results = searcher.search(query, filter=filter_query, limit=20)
```

### 5.3 通配符查询

```python
from whoosh.qparser import WildcardPlugin

# 创建支持通配符的解析器
parser = QueryParser("content", ix.schema)
parser.add_plugin(WildcardPlugin())

# 执行通配符查询
query = parser.parse("南开*")  # 匹配以"南开"开头的词
```

### 5.4 结果高亮

```python
from whoosh.highlight import Highlighter, ContextFragmenter

# 创建高亮器
highlighter = Highlighter(fragmenter=ContextFragmenter(maxchars=200, surround=50))

# 在搜索结果中高亮匹配词
for hit in results:
    highlighted_text = hit.highlights("content", highlighter=highlighter)
    print(highlighted_text)
```

## 6. 性能优化

### 6.1 内存管理

```python
from whoosh.filedb.filestore import RamStorage

# 使用内存存储提高性能(适用于小型索引)
storage = RamStorage()
temp_ix = storage.create_index(schema)

# 或使用内存缓存来加速
from whoosh.filedb.filestore import FileStorage
storage = FileStorage(INDEX_DIR)
ix = storage.open_index(schema=schema)
```

### 6.2 批量索引

```python
# 批量添加文档
writer = ix.writer()
for doc in document_batch:
    writer.add_document(**doc)
writer.commit()
```

### 6.3 索引压缩

```python
# 优化索引(合并段)
writer = ix.writer()
writer.commit(optimize=True)
```

## 7. 多索引管理

### 7.1 索引分片

```python
from whoosh.filedb.multiproc import MultiSegmentWriter

# 使用多进程写入器
writer = MultiSegmentWriter(ix, procs=4)
```

### 7.2 多索引搜索

```python
from whoosh.searching import IndexSearcher

# 创建多个索引的搜索器
searcher1 = ix1.searcher()
searcher2 = ix2.searcher()

# 合并搜索器
from whoosh.searching import MultiSearcher
msearcher = MultiSearcher([searcher1, searcher2])

# 执行搜索
results = msearcher.search(query)
```

## 8. 与FastAPI集成

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import os
from datetime import datetime

app = FastAPI(title="南翎搜索API")

# 数据模型
class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    file_type: str
    score: float
    publish_time: Optional[datetime] = None

class SearchResponse(BaseModel):
    total: int
    page: int
    page_size: int
    hits: List[SearchResult]

# 搜索端点
@app.get("/api/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="搜索查询"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(10, ge=1, le=50, description="每页结果数"),
    file_type: Optional[str] = Query(None, description="文件类型过滤")
):
    # 构建过滤条件
    filter_dict = {}
    if file_type:
        filter_dict["file_type"] = file_type
    
    # 执行搜索
    results = perform_search(
        query_text=q,
        page=page,
        page_size=page_size,
        filter_dict=filter_dict
    )
    
    return results
```

## 9. 与Scrapy集成

```python
# Scrapy管道，用于将爬取结果直接写入Whoosh索引
class WhooshIndexPipeline:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.schema = create_schema()
        
    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            index_dir=crawler.settings.get('WHOOSH_INDEX_DIR', 'data/index')
        )
    
    def open_spider(self, spider):
        # 确保索引目录存在
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        
        # 打开或创建索引
        if not os.listdir(self.index_dir):
            self.index = create_in(self.index_dir, self.schema)
        else:
            self.index = open_dir(self.index_dir)
        
        # 创建writer
        self.writer = self.index.writer()
        
    def process_item(self, item, spider):
        # 解析日期
        try:
            publish_time = datetime.strptime(item.get('publish_time', ''), "%Y-%m-%d %H:%M:%S")
        except:
            publish_time = datetime.now()
            
        # 添加或更新文档
        self.writer.update_document(
            url=item['url'],
            title=item.get('title', ''),
            content=item.get('content', ''),
            anchor_text=item.get('anchor_text', ''),
            publish_time=publish_time,
            file_type=item.get('file_type', 'html'),
            outlinks=','.join(item.get('outlinks', [])),
            inlinks=','.join(item.get('inlinks', [])),
            snapshot_path=item.get('snapshot_path', '')
        )
        
        return item
    
    def close_spider(self, spider):
        # 提交更改并关闭writer
        self.writer.commit()
``` 