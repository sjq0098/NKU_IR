#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
南开新闻搜索引擎启动脚本
"""

import os
import sys
import logging
from search_engine import app

def check_dependencies():
    """检查系统依赖"""
    try:
        import flask
        import pymongo
        import whoosh
        import jieba
        print("✅ 所有依赖包检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def check_mongodb():
    """检查MongoDB连接"""
    try:
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
        client.admin.command('ismaster')
        print("✅ MongoDB连接正常")
        return True
    except Exception as e:
        print(f"❌ MongoDB连接失败: {e}")
        print("请确保MongoDB服务正在运行")
        return False

def check_index():
    """检查搜索索引"""
    index_dir = os.path.join(os.path.dirname(__file__), "whoosh_index")
    news_index = os.path.join(index_dir, "news")
    docs_index = os.path.join(index_dir, "documents")
    
    if os.path.exists(news_index) and os.path.exists(docs_index):
        print("✅ 搜索索引存在")
        return True
    else:
        print("⚠️  搜索索引不存在")
        print("请先运行: python build_whoosh_index.py")
        return False

def print_banner():
    """打印启动横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════╗
    ║             南开新闻搜索引擎 (NKUNEWS)                ║
    ║         基于Whoosh和BM25的智能搜索系统                ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """主函数"""
    print_banner()
    
    print("🔍 系统检查中...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要3.8或更高版本")
        return
    
    print(f"✅ Python版本: {sys.version}")
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查MongoDB
    if not check_mongodb():
        return
    
    # 检查索引
    check_index()
    
    print("\n🚀 启动Web服务器...")
    print("📱 访问地址: http://localhost:5000")
    print("🛑 按 Ctrl+C 停止服务\n")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 启动Flask应用
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # 避免重复加载
        )
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")

if __name__ == "__main__":
    main() 