#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import hashlib
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from whoosh.index import open_dir
from whoosh.qparser import QueryParser, MultifieldParser, OrGroup, WildcardPlugin, PhrasePlugin
from whoosh.query import *
from whoosh.scoring import BM25F
from jieba.analyse import ChineseAnalyzer
from pymongo import MongoClient
import gridfs
import jieba
import re
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # 在生产环境中请使用更安全的密钥

# 配置日志
logging.basicConfig(level=logging.INFO)

class NankaiSearchEngine:
    def __init__(self, mongo_url='mongodb://localhost:27017/'):
        """初始化搜索引擎"""
        self.mongo_client = MongoClient(mongo_url)
        self.db = self.mongo_client['nankai_news_datasets']
        self.news_collection = self.db['NEWS']
        self.files_collection = self.db['FILES']
        self.users_collection = self.db['USERS']
        self.search_logs_collection = self.db['SEARCH_LOGS']
        self.snapshots_collection = self.db['WEB_snapshot']
        self.fs = gridfs.GridFS(self.db)
        
        # 索引目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_dir = os.path.join(current_dir, "whoosh_index")
        self.news_index_dir = os.path.join(self.index_dir, "news")
        self.documents_index_dir = os.path.join(self.index_dir, "documents")
        
        # 初始化分析器
        self.analyzer = ChineseAnalyzer()
        
        # 学院兴趣关键词映射
        self.college_interests = {
            '计算机': ['计算机', '软件', '人工智能', '算法', '编程', '数据', '网络', '技术', 'AI', '机器学习'],
            '金融': ['金融', '经济', '银行', '投资', '股票', '基金', '证券', '货币', '财经', '市场'],
            '数学': ['数学', '统计', '概率', '函数', '方程', '几何', '代数', '微积分', '算术'],
            '物理': ['物理', '力学', '电磁', '光学', '量子', '相对论', '实验', '能量', '粒子'],
            '化学': ['化学', '有机', '无机', '分析', '物化', '实验', '分子', '反应', '元素'],
            '文学': ['文学', '诗歌', '小说', '散文', '戏剧', '作家', '文化', '语言', '阅读'],
            '历史': ['历史', '古代', '近代', '现代', '文物', '考古', '史学', '传统', '文明'],
            '生物': ['生物', '细胞', '基因', 'DNA', '生态', '进化', '医学', '植物', '动物']
        }

    def get_user_college(self, user_id):
        """获取用户所属学院"""
        user = self.users_collection.find_one({'_id': user_id})
        return user.get('college', '通用') if user else '通用'

    def calculate_personalized_score(self, result, user_college):
        """计算个性化BM25分数"""
        base_score = result.score
        
        if user_college == '通用' or user_college not in self.college_interests:
            return base_score
        
        # 获取用户学院的兴趣关键词
        interest_keywords = self.college_interests[user_college]
        
        # 检查标题和内容中是否包含兴趣关键词
        title = result.get('title', '').lower()
        content = result.get('content', '').lower()
        source = result.get('source', '').lower()
        
        interest_boost = 0
        for keyword in interest_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in title:
                interest_boost += 0.3  # 标题中出现给予更高权重
            if keyword_lower in content:
                interest_boost += 0.1  # 内容中出现给予较低权重
            if keyword_lower in source:
                interest_boost += 0.2  # 来源中出现给予中等权重
        
        # 应用个性化加权
        personalized_score = base_score * (1 + interest_boost)
        return personalized_score

    def calculate_time_score(self, result):
        """计算时效性分数"""
        date = result.get('date')
        if not date:
            return 0.5  # 没有日期的给予中等分数
        
        if isinstance(date, str):
            try:
                date = datetime.fromisoformat(date.replace('Z', '+00:00'))
            except:
                return 0.5
        
        # 计算时间差（天数）
        days_diff = (datetime.now() - date).days
        
        # 时效性评分：越新的新闻分数越高
        if days_diff <= 1:
            return 1.0  # 1天内
        elif days_diff <= 7:
            return 0.9  # 1周内
        elif days_diff <= 30:
            return 0.8  # 1月内
        elif days_diff <= 90:
            return 0.6  # 3月内
        elif days_diff <= 365:
            return 0.4  # 1年内
        else:
            return 0.2  # 1年以上

    def search_news(self, query_text, user_id=None, search_type='basic', limit=20, page=1):
        """搜索新闻"""
        if not os.path.exists(self.news_index_dir):
            return {'error': '新闻索引不存在'}
        
        try:
            ix = open_dir(self.news_index_dir)
            with ix.searcher(weighting=BM25F()) as searcher:
                # 根据搜索类型构建查询
                if search_type == 'phrase':
                    # 短语搜索 - 改用And查询来模拟短语搜索
                    try:
                        from whoosh.query import And, Term, Or
                        import jieba
                        
                        # 使用jieba进行分词
                        tokens = [token.strip() for token in jieba.cut(query_text) if token.strip()]
                        
                        if len(tokens) == 1:
                            # 单词，直接使用基本搜索
                            parser = MultifieldParser(["title", "content", "source"], ix.schema, group=OrGroup)
                            query = parser.parse(query_text)
                        else:
                            # 多词短语，为每个字段创建And查询，要求所有词都出现
                            field_queries = []
                            for field_name in ["title", "content", "source"]:
                                if field_name in ix.schema:
                                    # 为每个字段创建包含所有token的And查询
                                    token_queries = [Term(field_name, token) for token in tokens]
                                    field_and_query = And(token_queries)
                                    field_queries.append(field_and_query)
                            
                            # 组合所有字段的查询（任意字段匹配即可）
                            if field_queries:
                                query = Or(field_queries)
                            else:
                                # 回退到基本搜索
                                parser = MultifieldParser(["title", "content", "source"], ix.schema, group=OrGroup)
                                query = parser.parse(query_text)
                        
                    except Exception as phrase_error:
                        logging.warning(f"短语搜索失败，回退到基本搜索: {phrase_error}")
                        # 回退到基本搜索
                        parser = MultifieldParser(["title", "content", "source"], ix.schema, group=OrGroup)
                        query = parser.parse(query_text)
                        
                elif search_type == 'wildcard':
                    # 通配符搜索
                    parser = MultifieldParser(["title", "content", "source"], ix.schema, group=OrGroup)
                    parser.add_plugin(WildcardPlugin())
                    query = parser.parse(query_text)
                else:
                    # 基本搜索
                    parser = MultifieldParser(["title", "content", "source"], ix.schema, group=OrGroup)
                    query = parser.parse(query_text)
                
                # 执行搜索
                results = searcher.search(query, limit=limit * 5)  # 获取更多结果用于重排序
                
                # 获取用户学院信息
                user_college = self.get_user_college(user_id) if user_id else '通用'
                
                # 重新计算分数并排序
                scored_results = []
                for result in results:
                    # 计算个性化分数
                    personalized_score = self.calculate_personalized_score(result, user_college)
                    # 计算时效性分数
                    time_score = self.calculate_time_score(result)
                    # 获取PageRank分数
                    pagerank = result.get('pagerank', 0.0)
                    
                    # 综合分数：BM25 + 个性化 + 时效性 + PageRank
                    final_score = (personalized_score * 0.4 + 
                                 time_score * 0.3 + 
                                 pagerank * 10 * 0.2 +
                                 result.score * 0.1)
                    
                    scored_results.append({
                        'id': result.get('id'),
                        'title': result.get('title'),
                        'content': result.get('content', '')[:200] + '...' if len(result.get('content', '')) > 200 else result.get('content', ''),
                        'url': result.get('url'),
                        'source': result.get('source'),
                        'date': result.get('date'),
                        'pagerank': pagerank,
                        'final_score': final_score,
                        'snapshot_hash': result.get('snapshot_hash')
                    })
                
                # 按综合分数排序
                scored_results.sort(key=lambda x: x['final_score'], reverse=True)
                
                # 分页
                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                paginated_results = scored_results[start_idx:end_idx]
                
                # 记录搜索日志
                if user_id:
                    self.log_search(user_id, query_text, search_type, len(scored_results))
                
                return {
                    'results': paginated_results,
                    'total': len(scored_results),
                    'query': query_text,
                    'user_college': user_college,
                    'page': page,
                    'per_page': limit
                }
                
        except Exception as e:
            logging.error(f"搜索出错: {e}")
            return {'error': str(e)}

    def search_documents(self, query_text, limit=20):
        """搜索文档"""
        if not os.path.exists(self.documents_index_dir):
            return {'error': '文档索引不存在'}
        
        try:
            ix = open_dir(self.documents_index_dir)
            with ix.searcher() as searcher:
                parser = MultifieldParser(["title", "file_name"], ix.schema, group=OrGroup)
                query = parser.parse(query_text)
                results = searcher.search(query, limit=limit)
                
                documents = []
                for result in results:
                    documents.append({
                        'id': result.get('id'),
                        'title': result.get('title'),
                        'file_name': result.get('file_name'),
                        'file_type': result.get('file_type'),
                        'file_size': result.get('file_size'),
                        'url': result.get('url'),
                        'gridfs_id': result.get('gridfs_id')
                    })
                
                return {
                    'results': documents,
                    'total': len(documents),
                    'query': query_text
                }
                
        except Exception as e:
            logging.error(f"文档搜索出错: {e}")
            return {'error': str(e)}

    def log_search(self, user_id, query, search_type, result_count):
        """记录搜索日志"""
        try:
            self.search_logs_collection.insert_one({
                'user_id': user_id,
                'query': query,
                'search_type': search_type,
                'result_count': result_count,
                'timestamp': datetime.now()
            })
        except Exception as e:
            logging.error(f"记录搜索日志失败: {e}")

    def get_search_history(self, user_id, limit=20):
        """获取用户搜索历史"""
        try:
            history = list(self.search_logs_collection.find(
                {'user_id': user_id}
            ).sort('timestamp', -1).limit(limit))
            
            return [{'query': log['query'], 'timestamp': log['timestamp']} for log in history]
        except Exception as e:
            logging.error(f"获取搜索历史失败: {e}")
            return []

    def get_snapshot(self, snapshot_hash):
        """获取网页快照"""
        try:
            # 从WEB_snapshot集合获取快照内容，字段为content_hash和html_content
            snapshot = self.db['WEB_snapshot'].find_one({'content_hash': snapshot_hash})
            return snapshot.get('html_content', '') if snapshot else None
        except Exception as e:
            logging.error(f"获取快照失败: {e}")
            return None

    def register_user(self, username, password, email, college):
        """用户注册"""
        try:
            # 检查用户名是否已存在
            if self.users_collection.find_one({'username': username}):
                return {'error': '用户名已存在'}
            
            # 检查邮箱是否已存在
            if self.users_collection.find_one({'email': email}):
                return {'error': '邮箱已被注册'}
            
            # 创建用户
            user_id = self.users_collection.insert_one({
                'username': username,
                'password_hash': generate_password_hash(password),
                'email': email,
                'college': college,
                'created_at': datetime.now()
            }).inserted_id
            
            return {'success': True, 'user_id': str(user_id)}
        except Exception as e:
            logging.error(f"用户注册失败: {e}")
            return {'error': '注册失败'}

    def login_user(self, username, password):
        """用户登录"""
        try:
            user = self.users_collection.find_one({'username': username})
            if user and check_password_hash(user['password_hash'], password):
                return {
                    'success': True,
                    'user_id': str(user['_id']),
                    'username': user['username'],
                    'college': user['college']
                }
            else:
                return {'error': '用户名或密码错误'}
        except Exception as e:
            logging.error(f"用户登录失败: {e}")
            return {'error': '登录失败'}

    def get_global_hotwords(self, limit=10):
        """获取全站热搜词"""
        try:
            pipeline = [
                {"$group": {"_id": "$query", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": limit}
            ]
            hotwords = list(self.search_logs_collection.aggregate(pipeline))
            return [item['_id'] for item in hotwords]
        except Exception as e:
            logging.error(f"获取全局热词失败: {e}")
            return []

    def get_user_frequent_queries(self, user_id, limit=10):
        """获取用户高频搜索词"""
        try:
            pipeline = [
                {"$match": {"user_id": user_id}},
                {"$group": {"_id": "$query", "count": {"$sum": 1}, "last_search": {"$max": "$timestamp"}}},
                {"$sort": {"count": -1, "last_search": -1}},
                {"$limit": limit}
            ]
            frequent_queries = list(self.search_logs_collection.aggregate(pipeline))
            return [item['_id'] for item in frequent_queries]
        except Exception as e:
            logging.error(f"获取用户高频搜索词失败: {e}")
            return []

    def get_smart_suggestions(self, query, user_id=None, limit=10):
        """智能搜索建议"""
        suggestions = []
        query = query.lower().strip()
        
        if len(query) < 1:
            return []
        
        try:
            # 1. 用户历史搜索（个性化权重最高）
            user_history = []
            user_frequent = []
            if user_id:
                # 最近搜索历史
                recent_logs = list(self.search_logs_collection.find(
                    {'user_id': user_id}
                ).sort('timestamp', -1).limit(50))
                
                user_history = []
                seen_queries = set()
                for log in recent_logs:
                    q = log['query'].lower()
                    if query in q and q not in seen_queries and len(user_history) < 5:
                        user_history.append(log['query'])
                        seen_queries.add(q)
                
                # 用户高频搜索
                user_frequent = self.get_user_frequent_queries(user_id, 10)
                user_frequent = [q for q in user_frequent if query in q.lower() and q not in user_history][:3]

            # 2. 基于用户学院的智能建议
            college_suggestions = []
            if user_id:
                user_college = self.get_user_college(user_id)
                if user_college in self.college_interests:
                    interest_words = self.college_interests[user_college]
                    
                    # 匹配兴趣关键词
                    for word in interest_words:
                        if query in word.lower():
                            college_suggestions.append(word)
                    
                    # 组合建议（查询词 + 学院兴趣词）
                    for word in interest_words[:3]:
                        combined = f"{query} {word}"
                        if len(combined) <= 20:  # 限制长度
                            college_suggestions.append(combined)
                    
                    college_suggestions = college_suggestions[:4]

            # 3. 全局热搜词匹配
            global_hotwords = self.get_global_hotwords(20)
            global_matches = [q for q in global_hotwords if query in q.lower()][:3]

            # 4. 基于索引的智能补全
            index_suggestions = []
            try:
                if os.path.exists(self.news_index_dir):
                    ix = open_dir(self.news_index_dir)
                    with ix.searcher() as searcher:
                        # 使用Whoosh的拼写建议功能
                        corrector = searcher.corrector("title")
                        suggestions_from_index = corrector.suggest(query, limit=5)
                        index_suggestions.extend(suggestions_from_index)
                        
                        # 基于标题字段的前缀匹配
                        from whoosh.query import Prefix
                        prefix_query = Prefix("title", query)
                        results = searcher.search(prefix_query, limit=10)
                        for result in results:
                            title_words = jieba.cut(result['title'])
                            for word in title_words:
                                if len(word) > 1 and query in word.lower() and word not in index_suggestions:
                                    index_suggestions.append(word)
                                    if len(index_suggestions) >= 5:
                                        break
            except Exception as e:
                logging.error(f"从索引获取建议失败: {e}")

            # 5. 智能组合和排序
            result = {
                'recent': user_history[:3],           # 最近搜索
                'frequent': user_frequent[:3],        # 高频搜索  
                'college': college_suggestions[:3],   # 学院相关
                'hot': global_matches[:3],           # 全站热搜
                'smart': index_suggestions[:3]       # 智能补全
            }
            
            # 去重并生成最终建议列表
            final_suggestions = []
            seen = set()
            
            # 优先级：最近搜索 > 高频搜索 > 学院相关 > 智能补全 > 全站热搜
            for category in ['recent', 'frequent', 'college', 'smart', 'hot']:
                for suggestion in result[category]:
                    if suggestion.lower() not in seen and len(final_suggestions) < limit:
                        final_suggestions.append(suggestion)
                        seen.add(suggestion.lower())
            
            return {
                'suggestions': final_suggestions,
                'categorized': result
            }
            
        except Exception as e:
            logging.error(f"获取智能建议失败: {e}")
            return {'suggestions': [], 'categorized': {}}

    def get_spell_suggestions(self, query):
        """拼写建议和纠错"""
        try:
            suggestions = []
            if os.path.exists(self.news_index_dir):
                ix = open_dir(self.news_index_dir)
                with ix.searcher() as searcher:
                    # 对标题和内容字段进行拼写建议
                    for field in ['title', 'content']:
                        corrector = searcher.corrector(field)
                        field_suggestions = corrector.suggest(query, limit=3)
                        suggestions.extend(field_suggestions)
            
            # 去重
            return list(set(suggestions))
        except Exception as e:
            logging.error(f"拼写建议失败: {e}")
            return []

# 创建搜索引擎实例
search_engine = NankaiSearchEngine()

def login_required(f):
    """登录装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/search')
def search():
    """搜索页面"""
    query = request.args.get('q', '').strip()
    search_type = request.args.get('type', 'basic')
    page = int(request.args.get('page', 1))
    
    if not query:
        return render_template('search.html', error='请输入搜索关键词', results={'query': '', 'total': 0, 'results': []}, search_type=search_type)
    
    user_id = session.get('user_id')
    
    if search_type == 'documents':
        results = search_engine.search_documents(query)
        return render_template('documents.html', results=results, query=query)
    else:
        results = search_engine.search_news(query, user_id, search_type, page=page)
        return render_template('search.html', results=results, query=query, search_type=search_type)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """用户注册"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        college = request.form.get('college')
        
        result = search_engine.register_user(username, password, email, college)
        if 'error' in result:
            flash(result['error'], 'error')
        else:
            flash('注册成功，请登录', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """用户登录"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        result = search_engine.login_user(username, password)
        if 'error' in result:
            flash(result['error'], 'error')
        else:
            session['user_id'] = result['user_id']
            session['username'] = result['username']
            session['college'] = result['college']
            return redirect(url_for('index'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """用户登出"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/history')
@login_required
def search_history():
    """搜索历史"""
    user_id = session.get('user_id')
    history = search_engine.get_search_history(user_id)
    return render_template('history.html', history=history)

@app.route('/snapshot/<snapshot_hash>')
def snapshot(snapshot_hash):
    """网页快照"""
    content = search_engine.get_snapshot(snapshot_hash)
    if content:
        return render_template('snapshot.html', content=content, snapshot_hash=snapshot_hash)
    else:
        return render_template('error.html', message='快照不存在'), 404

@app.route('/api/suggestions')
def suggestions():
    """智能搜索建议API"""
    query = request.args.get('q', '').strip()
    user_id = session.get('user_id')
    
    if len(query) < 1:
        return jsonify({'suggestions': [], 'categorized': {}})
    
    # 使用新的智能建议系统
    result = search_engine.get_smart_suggestions(query, user_id, limit=10)
    
    # 添加拼写建议
    if len(result['suggestions']) < 5:
        spell_suggestions = search_engine.get_spell_suggestions(query)
        for suggestion in spell_suggestions:
            if suggestion not in result['suggestions'] and len(result['suggestions']) < 10:
                result['suggestions'].append(suggestion)
    
    return jsonify(result)

@app.route('/api/clear-history', methods=['POST'])
@login_required
def clear_search_history():
    """清空搜索历史"""
    try:
        user_id = session.get('user_id')
        search_engine.search_logs_collection.delete_many({'user_id': user_id})
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"清空搜索历史失败: {e}")
        return jsonify({'success': False})

@app.route('/api/trending')
def trending_searches():
    """获取实时热门搜索"""
    try:
        # 获取今日热搜
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_hotwords = list(search_engine.search_logs_collection.aggregate([
            {"$match": {"timestamp": {"$gte": today_start}}},
            {"$group": {"_id": "$query", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]))
        
        # 获取本周热搜
        week_start = datetime.now() - timedelta(days=7)
        week_hotwords = list(search_engine.search_logs_collection.aggregate([
            {"$match": {"timestamp": {"$gte": week_start}}},
            {"$group": {"_id": "$query", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 8}
        ]))
        
        # 获取用户个性化推荐
        personalized = []
        user_id = session.get('user_id')
        if user_id:
            college = session.get('college', '通用')
            if college in search_engine.college_interests:
                personalized = search_engine.college_interests[college][:6]
        
        result = {
            'today': [item['_id'] for item in today_hotwords],
            'week': [item['_id'] for item in week_hotwords],
            'personalized': personalized
        }
        
        return jsonify(result)
    except Exception as e:
        logging.error(f"获取热门搜索失败: {e}")
        return jsonify({'today': [], 'week': [], 'personalized': []})

@app.route('/sw.js')
def service_worker():
    """Service Worker (可选的PWA功能)"""
    return '', 404

@app.route('/hybridaction/zybTrackerStatisticsAction')
def tracker_stats():
    """处理跟踪统计请求"""
    return '', 204  # No Content

@app.errorhandler(404)
def not_found_error(error):
    """处理404错误"""
    return render_template('error.html', 
                         message='抱歉，您访问的页面不存在或已被移除。',
                         error_code='404'), 404

@app.errorhandler(500)
def internal_error(error):
    """处理500错误"""
    return render_template('error.html', 
                         message='服务器遇到了一个错误，无法完成您的请求。',
                         error_code='500'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 