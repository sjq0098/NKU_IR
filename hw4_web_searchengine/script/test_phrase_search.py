#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from search_engine import NankaiSearchEngine

def test_phrase_search():
    """测试短语搜索功能"""
    print("=" * 50)
    print("测试短语搜索功能")
    print("=" * 50)
    
    try:
        # 创建搜索引擎实例
        engine = NankaiSearchEngine()
        
        # 测试用的短语查询
        test_queries = [
            "南开大学",
            "计算机学院", 
            "金融学院",
            "人工智能",
            "数据科学"
        ]
        
        for query in test_queries:
            print(f"\n测试短语搜索: '{query}'")
            print("-" * 30)
            
            try:
                # 执行短语搜索
                results = engine.search_news(
                    query_text=query,
                    user_id=None,
                    search_type='phrase',
                    limit=5,
                    page=1
                )
                
                if 'error' in results:
                    print(f"❌ 搜索失败: {results['error']}")
                else:
                    print(f"✅ 搜索成功")
                    print(f"找到 {results.get('total', 0)} 条结果")
                    
                    # 显示前几条结果
                    for i, result in enumerate(results.get('results', [])[:3], 1):
                        print(f"  {i}. {result.get('title', 'N/A')}")
                        if result.get('content'):
                            content = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                            print(f"     {content}")
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                
        # 对比测试：基础搜索 vs 短语搜索
        print(f"\n" + "=" * 50)
        print("对比测试：基础搜索 vs 短语搜索")
        print("=" * 50)
        
        test_query = "南开大学"
        
        print(f"\n测试查询: '{test_query}'")
        
        # 基础搜索
        print("\n1. 基础搜索结果:")
        basic_results = engine.search_news(test_query, search_type='basic', limit=3)
        if 'error' not in basic_results:
            for i, result in enumerate(basic_results.get('results', []), 1):
                print(f"  {i}. {result.get('title', 'N/A')}")
        
        # 短语搜索
        print("\n2. 短语搜索结果:")
        phrase_results = engine.search_news(test_query, search_type='phrase', limit=3)
        if 'error' not in phrase_results:
            for i, result in enumerate(phrase_results.get('results', []), 1):
                print(f"  {i}. {result.get('title', 'N/A')}")
        else:
            print(f"  ❌ 短语搜索失败: {phrase_results['error']}")
        
    except Exception as e:
        print(f"❌ 测试程序出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_phrase_search()
    print(f"\n" + "=" * 50)
    print("测试完成")
    print("如果短语搜索失败，建议运行: python rebuild_index.py")
    print("=" * 50) 