#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from datetime import datetime
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, DATETIME, STORED, NUMERIC
from jieba.analyse import ChineseAnalyzer
from tqdm import tqdm
from pymongo import MongoClient
import gridfs

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_index.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class WhooshIndexBuilder:
    def __init__(self, mongo_url='mongodb://localhost:27017/'):
        """初始化索引构建器"""
        self.mongo_client = MongoClient(mongo_url)
        self.db = self.mongo_client['nankai_news_datasets']
        self.news_collection = self.db['NEWS']
        self.files_collection = self.db['FILES']
        self.pagerank_collection = self.db['PAGERANK']
        self.fs = gridfs.GridFS(self.db)
        
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_dir = os.path.join(current_dir, "whoosh_index")
        self.news_index_dir = os.path.join(self.index_dir, "news")
        self.documents_index_dir = os.path.join(self.index_dir, "documents")
        
        # 创建索引目录
        self._create_index_directories()
        
        logging.info(f"索引将保存在: {self.index_dir}")

    def _create_index_directories(self):
        """创建索引目录"""
        for dir_path in [self.index_dir, self.news_index_dir, self.documents_index_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"创建目录: {dir_path}")

    def create_news_schema(self):
        """创建新闻索引结构"""
        analyzer = ChineseAnalyzer()
        return Schema(
            id=ID(stored=True, unique=True),
            url=ID(stored=True),
            title=TEXT(stored=True, analyzer=analyzer, phrase=True),
            content=TEXT(stored=True, analyzer=analyzer, phrase=True),
            source=TEXT(stored=True, analyzer=analyzer, phrase=True),
            date=DATETIME(stored=True),
            created_at=DATETIME(stored=True),
            pagerank=NUMERIC(stored=True, decimal_places=8),
            snapshot_hash=ID(stored=True),
            attachment_count=NUMERIC(stored=True)
        )

    def create_documents_schema(self):
        """创建文档索引结构"""
        analyzer = ChineseAnalyzer()
        return Schema(
            id=ID(stored=True, unique=True),
            url=ID(stored=True),
            title=TEXT(stored=True, analyzer=analyzer, phrase=True),
            file_name=TEXT(stored=True, analyzer=analyzer, phrase=True),
            file_type=ID(stored=True),
            file_size=NUMERIC(stored=True),
            fetched_at=DATETIME(stored=True),
            gridfs_id=ID(stored=True)
        )

    def get_pagerank_scores(self):
        """获取PageRank分数"""
        pagerank_scores = {}
        try:
            for doc in self.pagerank_collection.find({}):
                pagerank_scores[doc['url']] = doc.get('pagerank', 0.0)
            logging.info(f"获取到 {len(pagerank_scores)} 个PageRank分数")
        except Exception as e:
            logging.warning(f"获取PageRank分数失败: {e}")
        return pagerank_scores

    def parse_datetime(self, dt_obj):
        """解析日期时间"""
        if dt_obj is None:
            return None
        if isinstance(dt_obj, datetime):
            return dt_obj
        if isinstance(dt_obj, str):
            try:
                return datetime.fromisoformat(dt_obj.replace('Z', '+00:00'))
            except:
                try:
                    return datetime.strptime(dt_obj, "%Y-%m-%d")
                except:
                    return None
        return None

    def build_news_index(self):
        """构建新闻索引"""
        logging.info("开始构建新闻索引...")
        
        # 创建或打开索引
        news_ix = create_in(self.news_index_dir, self.create_news_schema())
        
        # 获取PageRank分数
        pagerank_scores = self.get_pagerank_scores()
        
        # 获取新闻总数
        total_news = self.news_collection.count_documents({})
        logging.info(f"找到 {total_news} 条新闻数据")
        
        if total_news == 0:
            logging.warning("没有新闻数据可索引")
            return news_ix
        
        # 构建索引
        with news_ix.writer() as writer:
            count = 0
            for news in tqdm(self.news_collection.find({}), total=total_news, desc="新闻索引进度"):
                try:
                    # 解析日期
                    date = self.parse_datetime(news.get('date'))
                    created_at = self.parse_datetime(news.get('created_at'))
                    
                    # 获取PageRank分数
                    pagerank = pagerank_scores.get(news.get('url'), 0.0)
                    
                    # 计算附件数量
                    attachment_count = len(news.get('attachments', []))
                    
                    # 准备文档数据
                    document = {
                        'id': str(news.get('_id')),
                        'url': news.get('url', ''),
                        'title': news.get('title', ''),
                        'content': news.get('content', ''),
                        'source': news.get('source', ''),
                        'pagerank': pagerank,
                        'snapshot_hash': news.get('snapshot_hash', ''),
                        'attachment_count': attachment_count
                    }
                    
                    # 添加日期字段（如果存在）
                    if date:
                        document['date'] = date
                    if created_at:
                        document['created_at'] = created_at
                    
                    # 过滤掉None值
                    document = {k: v for k, v in document.items() if v is not None}
                    
                    writer.add_document(**document)
                    count += 1
                    
                except Exception as e:
                    logging.error(f"添加新闻索引时出错 (ID: {news.get('_id')}): {str(e)}")
                    continue
            
            logging.info(f"新闻索引创建完成！共处理 {count} 条数据")
        
        return news_ix

    def build_documents_index(self):
        """构建文档索引"""
        logging.info("开始构建文档索引...")
        
        # 创建或打开索引
        documents_ix = create_in(self.documents_index_dir, self.create_documents_schema())
        
        # 获取文档总数
        total_files = self.files_collection.count_documents({})
        logging.info(f"找到 {total_files} 条文档数据")
        
        if total_files == 0:
            logging.warning("没有文档数据可索引")
            return documents_ix
        
        # 构建索引
        with documents_ix.writer() as writer:
            count = 0
            for file_doc in tqdm(self.files_collection.find({}), total=total_files, desc="文档索引进度"):
                try:
                    # 解析日期
                    fetched_at = self.parse_datetime(file_doc.get('fetched_at'))
                    
                    # 准备文档数据
                    document = {
                        'id': str(file_doc.get('_id')),
                        'url': file_doc.get('url', ''),
                        'title': file_doc.get('title', ''),
                        'file_name': file_doc.get('file_name', ''),
                        'file_type': file_doc.get('file_type', ''),
                        'file_size': file_doc.get('file_size', 0),
                        'gridfs_id': str(file_doc.get('gridfs_id', ''))
                    }
                    
                    # 添加日期字段（如果存在）
                    if fetched_at:
                        document['fetched_at'] = fetched_at
                    
                    # 过滤掉None值和空字符串
                    document = {k: v for k, v in document.items() if v is not None and v != ''}
                    
                    writer.add_document(**document)
                    count += 1
                    
                except Exception as e:
                    logging.error(f"添加文档索引时出错 (ID: {file_doc.get('_id')}): {str(e)}")
                    continue
            
            logging.info(f"文档索引创建完成！共处理 {count} 条数据")
        
        return documents_ix

    def build_all_indexes(self):
        """构建所有索引"""
        logging.info("开始构建所有索引...")
        
        try:
            # 测试MongoDB连接
            self.mongo_client.admin.command('ismaster')
            logging.info("MongoDB连接成功")
        except Exception as e:
            logging.error(f"MongoDB连接失败: {e}")
            return None, None
        
        # 打印数据库统计信息
        news_count = self.news_collection.count_documents({})
        files_count = self.files_collection.count_documents({})
        pagerank_count = self.pagerank_collection.count_documents({})
        
        logging.info(f"数据库统计信息:")
        logging.info(f"  新闻数据: {news_count} 条")
        logging.info(f"  文件数据: {files_count} 条")
        logging.info(f"  PageRank数据: {pagerank_count} 条")
        
        # 构建新闻索引
        news_ix = self.build_news_index()
        
        # 构建文档索引
        documents_ix = self.build_documents_index()
        
        logging.info("所有索引构建完成！")
        return news_ix, documents_ix

    def rebuild_indexes(self):
        """重建所有索引（删除旧索引）"""
        logging.info("重建索引：删除现有索引...")
        
        # 删除现有索引目录
        import shutil
        if os.path.exists(self.index_dir):
            shutil.rmtree(self.index_dir)
            logging.info("已删除现有索引")
        
        # 重新创建目录
        self._create_index_directories()
        
        # 构建新索引
        return self.build_all_indexes()

    def get_index_stats(self):
        """获取索引统计信息"""
        stats = {}
        
        try:
            if os.path.exists(self.news_index_dir):
                news_ix = open_dir(self.news_index_dir)
                with news_ix.searcher() as searcher:
                    stats['news_count'] = searcher.doc_count()
            else:
                stats['news_count'] = 0
        except Exception as e:
            logging.error(f"获取新闻索引统计失败: {e}")
            stats['news_count'] = 0
        
        try:
            if os.path.exists(self.documents_index_dir):
                documents_ix = open_dir(self.documents_index_dir)
                with documents_ix.searcher() as searcher:
                    stats['documents_count'] = searcher.doc_count()
            else:
                stats['documents_count'] = 0
        except Exception as e:
            logging.error(f"获取文档索引统计失败: {e}")
            stats['documents_count'] = 0
        
        return stats

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()
            logging.info("MongoDB连接已关闭")


def main():
    """主函数"""
    builder = None
    try:
        # 创建索引构建器
        builder = WhooshIndexBuilder()
        
        # 获取现有索引统计
        existing_stats = builder.get_index_stats()
        if existing_stats['news_count'] > 0 or existing_stats['documents_count'] > 0:
            logging.info(f"发现现有索引 - 新闻: {existing_stats['news_count']}, 文档: {existing_stats['documents_count']}")
            
            # 询问是否重建
            rebuild = input("发现现有索引，是否重建？(y/N): ").lower().strip()
            if rebuild == 'y':
                news_ix, documents_ix = builder.rebuild_indexes()
            else:
                logging.info("跳过重建，使用现有索引")
                return
        else:
            # 构建新索引
            news_ix, documents_ix = builder.build_all_indexes()
        
        # 显示最终统计信息
        final_stats = builder.get_index_stats()
        logging.info(f"索引构建完成！")
        logging.info(f"  新闻索引: {final_stats['news_count']} 条记录")
        logging.info(f"  文档索引: {final_stats['documents_count']} 条记录")
        logging.info(f"  索引位置: {builder.index_dir}")
        
    except KeyboardInterrupt:
        logging.info("用户中断操作")
    except Exception as e:
        logging.error(f"构建索引时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if builder:
            builder.cleanup()


if __name__ == "__main__":
    main() 