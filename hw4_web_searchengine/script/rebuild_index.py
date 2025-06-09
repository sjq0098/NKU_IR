#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import logging
from build_whoosh_index import WhooshIndexBuilder

def rebuild_indexes():
    """重建所有索引以支持短语搜索"""
    print("=" * 60)
    print("重建Whoosh索引以支持短语搜索")
    print("=" * 60)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 创建索引构建器
        builder = WhooshIndexBuilder()
        
        # 检查并备份现有索引
        current_dir = os.path.dirname(os.path.abspath(__file__))
        index_dir = os.path.join(current_dir, "whoosh_index")
        backup_dir = os.path.join(current_dir, "whoosh_index_backup")
        
        if os.path.exists(index_dir):
            print(f"备份现有索引到: {backup_dir}")
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(index_dir, backup_dir)
            
            print("删除旧索引...")
            shutil.rmtree(index_dir)
        
        # 重新创建索引目录
        os.makedirs(index_dir, exist_ok=True)
        os.makedirs(os.path.join(index_dir, "news"), exist_ok=True)
        os.makedirs(os.path.join(index_dir, "documents"), exist_ok=True)
        
        print("\n开始重建索引...")
        
        # 重建新闻索引
        print("\n1. 重建新闻索引...")
        news_ix = builder.build_news_index()
        print("✅ 新闻索引重建完成")
        
        # 重建文档索引
        print("\n2. 重建文档索引...")
        docs_ix = builder.build_documents_index()
        print("✅ 文档索引重建完成")
        
        # 显示索引统计
        print("\n索引重建完成！")
        stats = builder.get_index_stats()
        print(f"新闻索引文档数: {stats.get('news_docs', 0)}")
        print(f"文档索引文档数: {stats.get('documents_docs', 0)}")
        
        print("\n现在可以进行短语搜索了！")
        print("使用方法: 在搜索框中输入关键词，然后选择'短语搜索'按钮")
        print("例如: 搜索 '南开大学' 会查找包含完整短语'南开大学'的文档")
        
    except Exception as e:
        print(f"❌ 重建索引失败: {e}")
        logging.error(f"重建索引失败: {e}", exc_info=True)
        
        # 尝试恢复备份
        if os.path.exists(backup_dir):
            print("尝试恢复备份索引...")
            if os.path.exists(index_dir):
                shutil.rmtree(index_dir)
            shutil.copytree(backup_dir, index_dir)
            print("✅ 索引已恢复到重建前状态")
        
        return False
    
    return True

if __name__ == "__main__":
    success = rebuild_indexes()
    if success:
        print("\n🎉 索引重建成功！短语搜索功能现在可以正常使用了。")
    else:
        print("\n❌ 索引重建失败，请检查错误信息。") 