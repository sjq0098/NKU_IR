#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sqlite3
from datetime import datetime
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID, DATETIME, STORED, KEYWORD
from jieba.analyse import ChineseAnalyzer
from tqdm import tqdm
import os.path

def get_db_data():
    """从SQLite数据库获取数据"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 首先打印出当前目录中的所有数据库文件，帮助调试
    print("当前目录下的数据库文件:")
    for file in os.listdir(current_dir):
        if file.endswith('.db'):
            print(f" - {file}")
    
    # 连接网页数据库
    webpages_db_path = os.path.join(current_dir, "clean_webpages.db")
    if not os.path.exists(webpages_db_path):
        print(f"错误: 数据库文件不存在: {webpages_db_path}")
        webpages_db_path = os.path.join(current_dir, "webpages.db")
        if os.path.exists(webpages_db_path):
            print(f"尝试使用替代数据库: {webpages_db_path}")
        else:
            print(f"错误: 替代数据库文件不存在: {webpages_db_path}")
            return [], {}, []
    
    webpages_conn = sqlite3.connect(webpages_db_path)
    webpages_cursor = webpages_conn.cursor()
    
    # 检查数据库中的表
    webpages_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = webpages_cursor.fetchall()
    print(f"网页数据库中的表: {[table[0] for table in tables]}")
    
    # 连接文档数据库
    documents_db_path = os.path.join(current_dir, "clean_documents.db")
    if not os.path.exists(documents_db_path):
        print(f"错误: 数据库文件不存在: {documents_db_path}")
        documents_db_path = os.path.join(current_dir, "documents.db")
        if os.path.exists(documents_db_path):
            print(f"尝试使用替代数据库: {documents_db_path}")
        else:
            print(f"错误: 替代数据库文件不存在: {documents_db_path}")
            return [], {}, []
    
    documents_conn = sqlite3.connect(documents_db_path)
    documents_cursor = documents_conn.cursor()
    
    # 检查数据库中的表
    documents_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = documents_cursor.fetchall()
    print(f"文档数据库中的表: {[table[0] for table in tables]}")
    
    # 获取网页数据 (确保表名正确)
    try:
        webpages_cursor.execute("""
            SELECT id, url, title, content, html, publish_time, crawl_time, 
                clean_time, domain, metadata, is_valid
            FROM pages
            WHERE is_valid = 1
        """)
        pages = webpages_cursor.fetchall()
        print(f"获取到 {len(pages)} 条网页数据")
    except sqlite3.OperationalError as e:
        print(f"SQL错误: {e}")
        pages = []
    
    # 获取链接数据（用于提取锚文本）
    try:
        webpages_cursor.execute("""
            SELECT source_url, target_url, anchor_text
            FROM links
        """)
        links = webpages_cursor.fetchall()
        print(f"获取到 {len(links)} 条链接数据")
    except sqlite3.OperationalError as e:
        print(f"SQL错误: {e}")
        links = []
    
    # 创建URL到锚文本的映射
    url_to_anchors = {}
    for source, target, anchor in links:
        if target not in url_to_anchors:
            url_to_anchors[target] = []
        if anchor and anchor.strip():
            url_to_anchors[target].append(anchor.strip())
    
    # 获取文档数据
    try:
        documents_cursor.execute("""
            SELECT id, url, title, file_name, file_path, file_type, file_category, 
                file_size, crawl_time, clean_time, source_url, domain, anchor_text, 
                download_id, is_valid
            FROM documents
            WHERE is_valid = 1
        """)
        documents = documents_cursor.fetchall()
        print(f"获取到 {len(documents)} 条文档数据")
    except sqlite3.OperationalError as e:
        print(f"SQL错误: {e}")
        documents = []
    
    webpages_conn.close()
    documents_conn.close()
    
    return pages, url_to_anchors, documents

def create_webpage_schema():
    """创建网页索引结构"""
    analyzer = ChineseAnalyzer()
    return Schema(
        id=ID(stored=True, unique=True),
        url=ID(stored=True),
        title=TEXT(stored=True, analyzer=analyzer),
        content=TEXT(stored=True, analyzer=analyzer),
        domain=ID(stored=True),
        publish_time=DATETIME(stored=True),
        crawl_time=DATETIME(stored=True),
        clean_time=DATETIME(stored=True),
        anchor_text=TEXT(stored=True, analyzer=analyzer)
    )

def create_document_schema():
    """创建文档索引结构"""
    analyzer = ChineseAnalyzer()
    return Schema(
        id=ID(stored=True, unique=True),
        url=ID(stored=True),
        title=TEXT(stored=True, analyzer=analyzer),
        file_name=TEXT(stored=True, analyzer=analyzer),
        file_path=ID(stored=True),
        file_type=ID(stored=True),
        file_category=ID(stored=True),
        file_size=STORED,
        crawl_time=DATETIME(stored=True),
        clean_time=DATETIME(stored=True),
        domain=ID(stored=True),
        source_url=ID(stored=True),
        anchor_text=TEXT(stored=True, analyzer=analyzer)
    )

def parse_datetime(dt_str):
    """解析日期时间字符串"""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except:
        try:
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except:
            try:
                return datetime.strptime(dt_str, "%Y-%m-%d")
            except:
                return None

def initialize_index():
    """初始化索引"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建索引目录
    index_dir = os.path.join(current_dir, "..", "index")
    webpages_index_dir = os.path.join(index_dir, "webpages")
    documents_index_dir = os.path.join(index_dir, "documents") 
    
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    if not os.path.exists(webpages_index_dir):
        os.makedirs(webpages_index_dir)
    if not os.path.exists(documents_index_dir):
        os.makedirs(documents_index_dir)
    
    print(f"索引将保存在: {index_dir}")
    
    # 创建索引
    webpages_ix = create_in(webpages_index_dir, create_webpage_schema())
    documents_ix = create_in(documents_index_dir, create_document_schema())
    
    # 获取数据
    pages, url_to_anchors, documents = get_db_data()
    
    # 为网页添加索引
    if pages:
        with webpages_ix.writer() as writer:
            print("正在为网页数据创建索引...")
            count = 0
            # 使用tqdm创建进度条
            for page in tqdm(pages, desc="网页索引进度"):
                id_val, url, title, content, html, publish_time, crawl_time, clean_time, domain, metadata, is_valid = page
                
                # 获取该URL的所有锚文本
                anchor_texts = " ".join(url_to_anchors.get(url, []))
                
                document = {
                    'id': str(id_val),
                    'url': url,
                    'title': title if title else "",
                    'content': content if content else "",
                    'domain': domain if domain else "",
                    'publish_time': parse_datetime(publish_time),
                    'crawl_time': parse_datetime(crawl_time),
                    'clean_time': parse_datetime(clean_time),
                    'anchor_text': anchor_texts
                }
                
                # 过滤掉None值
                document = {k: v for k, v in document.items() if v is not None}
                
                try:
                    writer.add_document(**document)
                    count += 1
                except Exception as e:
                    print(f"\n添加网页索引时出错 (ID: {id_val}): {str(e)}")
            
            print(f"网页索引创建完成！共处理 {count} 条数据")
    else:
        print("没有网页数据可索引")
    
    # 为文档添加索引
    if documents:
        with documents_ix.writer() as writer:
            print("正在为文档数据创建索引...")
            count = 0
            # 使用tqdm创建进度条
            for doc in tqdm(documents, desc="文档索引进度"):
                id_val, url, title, file_name, file_path, file_type, file_category, file_size, \
                crawl_time, clean_time, source_url, domain, anchor_text, download_id, is_valid = doc
                
                document = {
                    'id': str(id_val),
                    'url': url if url else "",
                    'title': title if title else "",
                    'file_name': file_name if file_name else "",
                    'file_path': file_path if file_path else "",
                    'file_type': file_type if file_type else "",
                    'file_category': file_category if file_category else "",
                    'file_size': file_size,
                    'crawl_time': parse_datetime(crawl_time),
                    'clean_time': parse_datetime(clean_time),
                    'domain': domain if domain else "",
                    'source_url': source_url if source_url else "",
                    'anchor_text': anchor_text if anchor_text else ""
                }
                
                # 过滤掉None值
                document = {k: v for k, v in document.items() if v is not None}
                
                try:
                    writer.add_document(**document)
                    count += 1
                except Exception as e:
                    print(f"\n添加文档索引时出错 (ID: {id_val}): {str(e)}")
            
            print(f"文档索引创建完成！共处理 {count} 条数据")
    else:
        print("没有文档数据可索引")
    
    return webpages_ix, documents_ix

if __name__ == "__main__":
    webpages_ix, documents_ix = initialize_index() 