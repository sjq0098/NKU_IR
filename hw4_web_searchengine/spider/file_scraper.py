import os
import time
import random
import logging
import requests
from datetime import datetime
from urllib.parse import urljoin, urlparse, unquote
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo import MongoClient, errors
import gridfs

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# 全局关闭 InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("multi_domain_file_scraper.log", encoding='utf-8')
    ]
)

# 可下载文件后缀
DOWNLOAD_SUFFIXES = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
                     ".mp3", ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv",
                     ".zip", ".rar", ".tar", ".gz", ".bz2", ".7z",
                     ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
                     ".csv", ".txt", ".rtf"}

# 请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# 最大下载文件数
MAX_FILES = 1000
# 并发下载线程数
WORKERS = 8  # 降低并发数避免被封
# 请求延迟范围
DELAY_RANGE = (1, 2)  # 增加延迟避免被反爬虫

class MultiDomainFileScraper:
    def __init__(self, seeds):
        self.seeds = seeds
        self.visited = set()
        self.to_crawl = list(seeds)
        self.downloaded_count = 0

        # 测试并初始化MongoDB连接
        try:
            self.client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            # 测试连接
            self.client.admin.command('ismaster')
            self.db = self.client['nankai_news_datasets']
            self.fs = gridfs.GridFS(self.db)
            self.files_col = self.db['FILES']
            self.files_col.create_index('url', unique=True)
            logging.info("✅ MongoDB连接成功")
        except errors.ServerSelectionTimeoutError:
            logging.error("❌ MongoDB连接失败！请确保MongoDB服务已启动")
            raise Exception("MongoDB连接失败")
        except Exception as e:
            logging.error(f"❌ MongoDB初始化失败: {e}")
            raise

        # 允许的域名集合
        self.allowed_domains = {urlparse(u).netloc for u in seeds}
        logging.info(f"🎯 将爬取以下域名: {self.allowed_domains}")

    def fetch_soup(self, url):
        try:
            time.sleep(random.uniform(*DELAY_RANGE))
            response = requests.get(url, headers=HEADERS, timeout=15, verify=False)
            response.encoding = response.apparent_encoding
            
            if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                logging.info(f"✅ 成功获取页面: {url}")
                return BeautifulSoup(response.text, 'html.parser')
            else:
                logging.warning(f"⚠️ 页面响应异常 {response.status_code}: {url}")
        except requests.exceptions.Timeout:
            logging.warning(f"⏰ 请求超时: {url}")
        except requests.exceptions.ConnectionError:
            logging.warning(f"🔌 连接错误: {url}")
        except Exception as e:
            logging.warning(f"❌ 获取页面失败 {url}: {e}")
        return None

    def extract_links(self, soup, base_url):
        """提取页面中所有合法链接及其标题"""
        links = []
        if not soup:
            return links
            
        for a in soup.find_all('a', href=True):
            try:
                href = a['href'].strip()
                if not href:
                    continue
                    
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)
                
                # 只处理允许的域名
                if parsed.netloc not in self.allowed_domains:
                    continue
                
                # 获取链接标题
                title = a.get('title', '').strip()
                if not title:
                    title = a.get_text(strip=True)
                if not title:
                    title = unquote(os.path.basename(parsed.path))
                
                title = title.split('?')[0].split('#')[0].strip()
                if title:
                    links.append((full_url, title))
            except Exception as e:
                logging.debug(f"解析链接失败: {e}")
                continue
                
        logging.info(f"📋 从 {base_url} 提取到 {len(links)} 个链接")
        return links

    def is_download_link(self, url):
        """检查是否为可下载文件链接"""
        try:
            parsed_url = urlparse(url.lower())
            path = parsed_url.path
            return any(path.endswith(suffix) for suffix in DOWNLOAD_SUFFIXES)
        except:
            return False

    def download_and_store(self, url, title):
        """下载并存储文件到MongoDB GridFS"""
        # 检查是否已存在
        if self.files_col.find_one({'url': url}):
            logging.info(f"⏭️ 文件已存在，跳过: {url}")
            return False
            
        try:
            time.sleep(random.uniform(*DELAY_RANGE))
            logging.info(f"⬇️ 开始下载: {url}")
            
            response = requests.get(url, headers=HEADERS, timeout=30, verify=False, stream=True)
            if response.status_code == 200:
                # 尝试从响应头获取文件名
                content_disposition = response.headers.get('Content-Disposition', '')
                if 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[-1].strip('"\'')
                    if filename:
                        title = filename
                
                # 获取文件内容
                content = response.content
                if len(content) == 0:
                    logging.warning(f"⚠️ 文件内容为空: {url}")
                    return False
                
                # 存储到GridFS
                file_id = self.fs.put(
                    content,
                    filename=os.path.basename(title) or f"file_{int(time.time())}",
                    url=url,
                    title=title,
                    upload_date=datetime.utcnow(),
                    file_size=len(content)
                )
                
                # 存储元数据
                file_info = {
                    'url': url,
                    'title': title,
                    'file_name': os.path.basename(title) or f"file_{int(time.time())}",
                    'file_type': os.path.splitext(title)[1].lstrip('.') if '.' in title else 'unknown',
                    'file_size': len(content),
                    'gridfs_id': file_id,
                    'fetched_at': datetime.utcnow()
                }
                
                self.files_col.insert_one(file_info)
                self.downloaded_count += 1
                logging.info(f"✅ 文件保存成功 ({self.downloaded_count}): {title} - {len(content)} bytes")
                return True
            else:
                logging.warning(f"⚠️ 下载失败，状态码 {response.status_code}: {url}")
                
        except requests.exceptions.Timeout:
            logging.warning(f"⏰ 下载超时: {url}")
        except requests.exceptions.ConnectionError:
            logging.warning(f"🔌 下载连接错误: {url}")
        except Exception as e:
            logging.error(f"❌ 下载异常 {url}: {e}")
        
        return False

    def run(self):
        """主运行方法"""
        logging.info(f"🚀 开始爬取，目标文件数: {MAX_FILES}")
        
        # 获取已下载文件数
        existing_files = self.files_col.count_documents({})
        logging.info(f"📊 数据库中已有 {existing_files} 个文件")
        
        processed_pages = 0
        download_futures = []
        
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            while self.to_crawl and (existing_files + self.downloaded_count) < MAX_FILES:
                # 处理当前页面
                current_url = self.to_crawl.pop(0)
                if current_url in self.visited:
                    continue
                    
                self.visited.add(current_url)
                processed_pages += 1
                
                logging.info(f"🔍 正在处理第 {processed_pages} 个页面: {current_url}")
                
                # 获取页面内容
                soup = self.fetch_soup(current_url)
                if not soup:
                    continue
                
                # 提取链接
                links = self.extract_links(soup, current_url)
                
                download_links = []
                page_links = []
                
                for link_url, link_title in links:
                    if self.is_download_link(link_url):
                        download_links.append((link_url, link_title))
                    else:
                        if link_url not in self.visited:
                            page_links.append(link_url)
                
                # 提交下载任务
                for download_url, download_title in download_links:
                    if (existing_files + self.downloaded_count + len(download_futures)) >= MAX_FILES:
                        break
                    future = executor.submit(self.download_and_store, download_url, download_title)
                    download_futures.append(future)
                
                # 添加新的页面链接到待爬取队列
                self.to_crawl.extend(page_links)
                
                logging.info(f"📈 本页找到 {len(download_links)} 个下载链接，{len(page_links)} 个页面链接")
                
                # 检查并收集已完成的下载任务
                completed_futures = []
                for future in download_futures:
                    if future.done():
                        completed_futures.append(future)
                        try:
                            future.result()  # 获取结果，如果有异常会抛出
                        except Exception as e:
                            logging.error(f"❌ 下载任务异常: {e}")
                
                # 移除已完成的future
                for completed_future in completed_futures:
                    download_futures.remove(completed_future)
                
                # 状态报告
                if processed_pages % 10 == 0:
                    total_downloaded = existing_files + self.downloaded_count
                    logging.info(f"📊 进度报告: 已处理 {processed_pages} 页，已下载 {total_downloaded} 文件，队列中还有 {len(self.to_crawl)} 页面，{len(download_futures)} 个下载任务")
            
            # 等待所有下载任务完成
            logging.info("⏳ 等待剩余下载任务完成...")
            for future in as_completed(download_futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"❌ 最终下载任务异常: {e}")

        total_downloaded = existing_files + self.downloaded_count
        logging.info(f"🎉 爬取完成！总共下载了 {total_downloaded} 个文件（本次新增 {self.downloaded_count} 个）")
        self.client.close()

if __name__ == '__main__':
    # 种子URL
    seeds = [
        "https://jwc.nankai.edu.cn",
        "https://zsb.nankai.edu.cn", 
        "https://lib.nankai.edu.cn",
        "https://cc.nankai.edu.cn",
        "https://cs.nankai.edu.cn",
        "https://ai.nankai.edu.cn",
        "https://finance.nankai.edu.cn",
        "https://math.nankai.edu.cn",
        "https://physics.nankai.edu.cn",
        "https://chem.nankai.edu.cn",
        "https://economics.nankai.edu.cn",
        "https://bs.nankai.edu.cn"
    ]
    
    try:
        scraper = MultiDomainFileScraper(seeds)
        scraper.run()
    except KeyboardInterrupt:
        logging.info("🛑 用户中断程序")
    except Exception as e:
        logging.error(f"💥 程序运行失败: {e}")
