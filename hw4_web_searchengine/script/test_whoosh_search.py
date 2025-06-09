#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from whoosh.index import open_dir
from whoosh.qparser import QueryParser, MultifieldParser, OrGroup
from whoosh.query import *
from jieba.analyse import ChineseAnalyzer
import jieba

class WhooshSearchTester:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_dir = os.path.join(current_dir, "whoosh_index")
        self.news_index_dir = os.path.join(self.index_dir, "news")
        self.documents_index_dir = os.path.join(self.index_dir, "documents")
        
        # 初始化分析器
        self.analyzer = ChineseAnalyzer()
        
    def test_news_search(self, query_text, limit=10):
        """测试新闻搜索"""
        print(f"\n=== 搜索新闻: '{query_text}' ===")
        
        if not os.path.exists(self.news_index_dir):
            print("❌ 新闻索引不存在，请先运行 build_whoosh_index.py")
            return
        
        try:
            ix = open_dir(self.news_index_dir)
            with ix.searcher() as searcher:
                # 创建多字段查询解析器
                parser = MultifieldParser(["title", "content", "source"], ix.schema, group=OrGroup)
                
                # 解析查询
                query = parser.parse(query_text)
                print(f"查询解析结果: {query}")
                
                # 执行搜索
                results = searcher.search(query, limit=limit)
                
                print(f"找到 {len(results)} 条结果:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. 标题: {result['title']}")
                    print(f"   URL: {result['url']}")
                    print(f"   来源: {result.get('source', 'N/A')}")
                    print(f"   PageRank: {result.get('pagerank', 0.0):.6f}")
                    
                    # 显示内容摘要
                    content = result.get('content', '')
                    if content:
                        summary = content[:100] + "..." if len(content) > 100 else content
                        print(f"   摘要: {summary}")
                        
        except Exception as e:
            print(f"❌ 搜索出错: {e}")
    
    def test_document_search(self, query_text, limit=10):
        """测试文档搜索"""
        print(f"\n=== 搜索文档: '{query_text}' ===")
        
        if not os.path.exists(self.documents_index_dir):
            print("❌ 文档索引不存在，请先运行 build_whoosh_index.py")
            return
        
        try:
            ix = open_dir(self.documents_index_dir)
            with ix.searcher() as searcher:
                # 创建多字段查询解析器（主要搜索文件名）
                parser = MultifieldParser(["title", "file_name"], ix.schema, group=OrGroup)
                
                # 解析查询
                query = parser.parse(query_text)
                print(f"查询解析结果: {query}")
                
                # 执行搜索
                results = searcher.search(query, limit=limit)
                
                print(f"找到 {len(results)} 条结果:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. 文件名: {result['file_name']}")
                    print(f"   标题: {result.get('title', 'N/A')}")
                    print(f"   类型: {result.get('file_type', 'N/A')}")
                    print(f"   大小: {result.get('file_size', 0)} bytes")
                    print(f"   URL: {result['url']}")
                    
        except Exception as e:
            print(f"❌ 搜索出错: {e}")
    
    def show_index_stats(self):
        """显示索引统计信息"""
        print("\n=== 索引统计信息 ===")
        
        # 新闻索引统计
        if os.path.exists(self.news_index_dir):
            try:
                ix = open_dir(self.news_index_dir)
                with ix.searcher() as searcher:
                    news_count = searcher.doc_count()
                    print(f"新闻索引: {news_count} 条记录")
            except Exception as e:
                print(f"新闻索引错误: {e}")
        else:
            print("新闻索引: 不存在")
        
        # 文档索引统计
        if os.path.exists(self.documents_index_dir):
            try:
                ix = open_dir(self.documents_index_dir)
                with ix.searcher() as searcher:
                    docs_count = searcher.doc_count()
                    print(f"文档索引: {docs_count} 条记录")
            except Exception as e:
                print(f"文档索引错误: {e}")
        else:
            print("文档索引: 不存在")
    
    def interactive_test(self):
        """交互式测试"""
        print("=== Whoosh 索引测试工具 ===")
        self.show_index_stats()
        
        while True:
            print("\n请选择操作:")
            print("1. 搜索新闻")
            print("2. 搜索文档")
            print("3. 显示索引统计")
            print("4. 退出")
            
            choice = input("\n请输入选项 (1-4): ").strip()
            
            if choice == '1':
                query = input("请输入搜索关键词: ").strip()
                if query:
                    self.test_news_search(query)
                else:
                    print("搜索关键词不能为空")
                    
            elif choice == '2':
                query = input("请输入搜索关键词: ").strip()
                if query:
                    self.test_document_search(query)
                else:
                    print("搜索关键词不能为空")
                    
            elif choice == '3':
                self.show_index_stats()
                
            elif choice == '4':
                print("退出程序")
                break
                
            else:
                print("无效选项，请重新选择")

def main():
    tester = WhooshSearchTester()
    
    # 可以直接运行一些测试查询
    if len(os.sys.argv) > 1:
        query = " ".join(os.sys.argv[1:])
        print("=== 快速搜索测试 ===")
        tester.test_news_search(query)
        tester.test_document_search(query)
    else:
        # 交互式测试
        tester.interactive_test()

if __name__ == "__main__":
    main() 