import asyncio
from playwright.async_api import async_playwright, Page
 

results ={}
status ={}
context = None
        

async def one_page_handle(context, url):
    # 开启事件监听
    # page.on('response',printdata)
    # 进入子页面
    try:
        global results
        results[url] = [None,None]
        response = await context.request.get(url, timeout=5000)
        # 等待子页面加载完毕
        results[url] = (response.status, await response.text())
    except Exception as e:
        pass
 
async def get_conetent():
    global context
    if not context:
        # print("加载驱动")
        playwright = await async_playwright().start()
        browser = await playwright.firefox.launch()
        # 新建上下文
        context = await browser.new_context()
    return context
    

async def close_browser(browser):
     # 关闭浏览器驱动
    await browser.close()

async def get_raw_pages_(context, urls):
    # 封装异步任务
    tasks = []
    global results
    results = {}
    for url in urls:
        tasks.append(asyncio.create_task(one_page_handle(context, url)))
 
    await asyncio.wait(tasks, timeout=10)

   
async def get_raw_pages(urls, close_browser=False):
    context = await get_conetent()
    await get_raw_pages_(context,urls)
    
