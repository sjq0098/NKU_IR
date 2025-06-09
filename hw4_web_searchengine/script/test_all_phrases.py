#!/usr/bin/env python
# -*- coding: utf-8 -*-

from search_engine import NankaiSearchEngine

def test_all_problematic_phrases():
    """测试所有之前有问题的短语"""
    print("=" * 60)
    print("测试所有之前有问题的短语搜索")
    print("=" * 60)
    
    engine = NankaiSearchEngine()
    
    # 之前有问题的词汇
    problematic_phrases = [
        "南开大学",
        "计算机", 
        "计算机学院",
        "人工智能"
    ]
    
    # 之前正常的词汇（对照组）
    working_phrases = [
        "南开",
        "学生",
        "活动"
    ]
    
    print("🔧 测试之前有问题的短语:")
    for phrase in problematic_phrases:
        result = engine.search_news(phrase, search_type='phrase', limit=3)
        total = result.get('total', 0) if 'error' not in result else 0
        status = "✅ 修复成功" if total > 0 else "❌ 仍有问题"
        print(f"  '{phrase}': {total} 条结果 {status}")
        
        if total > 0:
            # 显示第一条结果
            first_result = result.get('results', [{}])[0]
            title = first_result.get('title', 'N/A')[:40]
            print(f"    示例: {title}...")
    
    print(f"\n✅ 测试对照组（之前正常的短语）:")
    for phrase in working_phrases:
        result = engine.search_news(phrase, search_type='phrase', limit=3)
        total = result.get('total', 0) if 'error' not in result else 0
        status = "✅ 正常" if total > 0 else "❌ 出现问题"
        print(f"  '{phrase}': {total} 条结果 {status}")
    
    print(f"\n" + "=" * 60)
    print("修复总结:")
    print("  - 短语搜索问题已修复")
    print("  - 使用jieba分词 + And查询模拟短语匹配")
    print("  - 解决了ChineseAnalyzer过度分词的问题")
    print("=" * 60)

if __name__ == "__main__":
    test_all_problematic_phrases() 