// 南开新闻搜索引擎 - 主JavaScript文件

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // 初始化搜索建议
    initSearchSuggestions();
    
    // 初始化工具提示
    initTooltips();
    
    // 初始化快捷键
    initKeyboardShortcuts();
    
    // 初始化滚动效果
    initScrollEffects();
    
    // 初始化表单验证
    initFormValidation();
    
    // 初始化主题切换
    initThemeToggle();
}

// 搜索建议功能
function initSearchSuggestions() {
    const searchInputs = document.querySelectorAll('input[name="q"]');
    
    searchInputs.forEach(input => {
        let suggestionTimer;
        
        input.addEventListener('input', function() {
            const query = this.value.trim();
            const suggestionsContainer = this.parentElement.querySelector('.suggestions');
            
            if (!suggestionsContainer) return;
            
            clearTimeout(suggestionTimer);
            
            if (query.length < 2) {
                suggestionsContainer.style.display = 'none';
                return;
            }
            
            suggestionTimer = setTimeout(() => {
                fetchSuggestions(query, suggestionsContainer);
            }, 300);
        });
        
        // 键盘导航
        input.addEventListener('keydown', function(e) {
            const suggestionsContainer = this.parentElement.querySelector('.suggestions');
            if (!suggestionsContainer || suggestionsContainer.style.display === 'none') return;
            
            const suggestions = suggestionsContainer.querySelectorAll('.suggestion-item');
            let activeIndex = Array.from(suggestions).findIndex(item => 
                item.classList.contains('active')
            );
            
            switch(e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    activeIndex = (activeIndex + 1) % suggestions.length;
                    updateActiveSuggestion(suggestions, activeIndex);
                    break;
                    
                case 'ArrowUp':
                    e.preventDefault();
                    activeIndex = activeIndex <= 0 ? suggestions.length - 1 : activeIndex - 1;
                    updateActiveSuggestion(suggestions, activeIndex);
                    break;
                    
                case 'Enter':
                    if (activeIndex >= 0) {
                        e.preventDefault();
                        const selectedItem = suggestions[activeIndex];
                        const suggestionText = selectedItem.textContent.trim();
                        selectSuggestion(suggestionText, selectedItem);
                    } else {
                        // 直接搜索当前输入
                        this.form.submit();
                    }
                    break;
                    
                case 'Escape':
                    suggestionsContainer.style.display = 'none';
                    break;
            }
        });
    });
}

// 获取搜索建议
async function fetchSuggestions(query, container) {
    try {
        const response = await fetch(`/api/suggestions?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        if (data.suggestions && data.suggestions.length > 0) {
            displaySuggestions(data, container, query);
        } else {
            container.style.display = 'none';
        }
    } catch (error) {
        console.error('获取搜索建议失败:', error);
        container.style.display = 'none';
    }
}

// 显示搜索建议
function displaySuggestions(data, container, query) {
    const categorized = data.categorized || {};
    const suggestions = data.suggestions || [];
    
    if (suggestions.length === 0) {
        container.style.display = 'none';
        return;
    }
    
    let html = '';
    
    // 分组显示建议
    const groups = [
        { key: 'recent', title: '最近搜索', icon: 'fas fa-clock' },
        { key: 'frequent', title: '常用搜索', icon: 'fas fa-star' },
        { key: 'college', title: '专业相关', icon: 'fas fa-graduation-cap' },
        { key: 'smart', title: '智能建议', icon: 'fas fa-lightbulb' },
        { key: 'hot', title: '热门搜索', icon: 'fas fa-fire' }
    ];
    
    groups.forEach(group => {
        const items = categorized[group.key] || [];
        if (items.length > 0) {
            html += `<div class="suggestion-group">`;
            html += `<div class="suggestion-group-title">
                <i class="${group.icon}"></i> ${group.title}
            </div>`;
            
            items.forEach((item, index) => {
                const highlightedText = highlightQuery(item, query);
                html += `<div class="suggestion-item" onclick="selectSuggestion('${escapeHtml(item)}', this)">
                    <i class="${group.icon} suggestion-icon"></i>
                    <span class="suggestion-text">${highlightedText}</span>
                </div>`;
            });
            
            html += `</div>`;
        }
    });
    
    // 如果没有分组数据，显示简单列表
    if (!html && suggestions.length > 0) {
        suggestions.forEach(item => {
            const highlightedText = highlightQuery(item, query);
            html += `<div class="suggestion-item" onclick="selectSuggestion('${escapeHtml(item)}', this)">
                <i class="fas fa-search suggestion-icon"></i>
                <span class="suggestion-text">${highlightedText}</span>
            </div>`;
        });
    }
    
    container.innerHTML = html;
    container.style.display = 'block';
}

function highlightQuery(text, query) {
    const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
    return text.replace(regex, '<span class="suggestion-query">$1</span>');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// 更新活跃建议
function updateActiveSuggestion(suggestions, activeIndex) {
    suggestions.forEach((item, index) => {
        item.classList.toggle('active', index === activeIndex);
    });
}

// 选择建议
function selectSuggestion(text, element) {
    const container = element.closest('.position-relative') || element.parentElement.parentElement;
    const input = container ? container.querySelector('input[name="q"]') : document.querySelector('input[name="q"]');
    const suggestions = container ? container.querySelector('.suggestions') : document.querySelector('.suggestions');
    
    if (input) {
        input.value = text;
        if (suggestions) {
            suggestions.style.display = 'none';
        }
        
        // 可选：自动提交搜索 (暂时注释掉，让用户决定何时搜索)
        // const form = input.closest('form');
        // if (form) {
        //     form.submit();
        // }
        
        // 将焦点放回输入框
        input.focus();
    }
}

// 初始化工具提示
function initTooltips() {
    // 使用Bootstrap的tooltip组件
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// 键盘快捷键
function initKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K: 聚焦搜索框
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('input[name="q"]');
            if (searchInput) {
                searchInput.focus();
                searchInput.select();
            }
        }
        
        // Ctrl/Cmd + Enter: 在新标签页搜索
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const form = document.querySelector('form[action*="search"]');
            if (form) {
                form.target = '_blank';
                form.submit();
                form.target = '';
            }
        }
        
        // ESC: 关闭搜索建议
        if (e.key === 'Escape') {
            const suggestions = document.querySelectorAll('.suggestions');
            suggestions.forEach(suggestion => {
                suggestion.style.display = 'none';
            });
        }
    });
}

// 滚动效果
function initScrollEffects() {
    // 返回顶部按钮
    const backToTop = document.createElement('button');
    backToTop.innerHTML = '<i class="fas fa-arrow-up"></i>';
    backToTop.className = 'btn btn-primary position-fixed';
    backToTop.style.cssText = `
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        display: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    `;
    backToTop.onclick = () => window.scrollTo({ top: 0, behavior: 'smooth' });
    document.body.appendChild(backToTop);
    
    // 监听滚动
    window.addEventListener('scroll', function() {
        if (window.scrollY > 300) {
            backToTop.style.display = 'block';
        } else {
            backToTop.style.display = 'none';
        }
        
        // 添加滚动视差效果
        const parallaxElements = document.querySelectorAll('.parallax');
        parallaxElements.forEach(element => {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;
            element.style.transform = `translateY(${rate}px)`;
        });
    });
}

// 表单验证
function initFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!form.checkValidity()) {
                e.preventDefault();
                e.stopPropagation();
                
                // 聚焦到第一个无效字段
                const invalidField = form.querySelector(':invalid');
                if (invalidField) {
                    invalidField.focus();
                }
            }
            
            form.classList.add('was-validated');
        });
        
        // 实时验证
        const inputs = form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                if (this.value.trim() !== '') {
                    if (this.checkValidity()) {
                        this.classList.remove('is-invalid');
                        this.classList.add('is-valid');
                    } else {
                        this.classList.remove('is-valid');
                        this.classList.add('is-invalid');
                    }
                }
            });
            
            input.addEventListener('input', function() {
                if (this.classList.contains('is-invalid') && this.checkValidity()) {
                    this.classList.remove('is-invalid');
                    this.classList.add('is-valid');
                }
            });
        });
    });
}

// 主题切换
function initThemeToggle() {
    const themeToggle = document.createElement('button');
    themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    themeToggle.className = 'btn btn-outline-secondary btn-sm position-fixed';
    themeToggle.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 1000;
        border-radius: 50%;
        width: 40px;
        height: 40px;
    `;
    themeToggle.title = '切换主题';
    
    // 检查当前主题
    const currentTheme = localStorage.getItem('theme') || 'light';
    if (currentTheme === 'dark') {
        document.body.classList.add('dark-theme');
        themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    }
    
    themeToggle.addEventListener('click', function() {
        document.body.classList.toggle('dark-theme');
        const isDark = document.body.classList.contains('dark-theme');
        
        this.innerHTML = isDark ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    });
    
    document.body.appendChild(themeToggle);
}

// 搜索统计
function trackSearch(query, type, results) {
    // 发送搜索统计到服务器
    fetch('/api/track-search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: query,
            type: type,
            results: results,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent
        })
    }).catch(error => {
        console.error('搜索统计发送失败:', error);
    });
}

// 工具函数
const utils = {
    // 防抖函数
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // 节流函数
    throttle: function(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },
    
    // 格式化文件大小
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    // 高亮关键词
    highlightKeywords: function(text, keywords) {
        if (!keywords || !text) return text;
        const keywordArray = keywords.split(' ').filter(k => k.length > 0);
        let highlightedText = text;
        
        keywordArray.forEach(keyword => {
            const regex = new RegExp(`(${keyword})`, 'gi');
            highlightedText = highlightedText.replace(regex, '<mark>$1</mark>');
        });
        
        return highlightedText;
    },
    
    // 复制到剪贴板
    copyToClipboard: function(text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => {
                this.showToast('已复制到剪贴板');
            });
        } else {
            // 降级方案
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            this.showToast('已复制到剪贴板');
        }
    },
    
    // 显示提示消息
    showToast: function(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `alert alert-${type} position-fixed`;
        toast.style.cssText = `
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            min-width: 200px;
            text-align: center;
        `;
        toast.textContent = message;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
};

// 暴露全局变量
window.searchApp = {
    utils: utils,
    selectSuggestion: selectSuggestion,
    trackSearch: trackSearch
};

// PWA支持
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker 注册成功:', registration.scope);
            })
            .catch(function(error) {
                console.log('ServiceWorker 注册失败:', error);
            });
    });
} 