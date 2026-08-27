# 从 Python 标准库中导入 asynccontextmanager。
# 它可以把一个异步函数包装成“上下文管理器”，
# 这里主要用于管理 FastAPI 应用启动和关闭时要执行的操作。
from contextlib import asynccontextmanager

# 导入 FastAPI 核心类，用来创建后端应用。
from fastapi import FastAPI

# 导入 CORS 中间件。
# CORS 用来控制哪些前端地址可以访问这个后端接口。
from fastapi.middleware.cors import CORSMiddleware

# 从 intake 接口文件中导入路由对象。

# 这样可以看出它属于信息采集模块。
from app.api.intake import router as intake_router

# 从 tasks 接口文件中导入任务相关的路由对象。
from app.api.tasks import router as tasks_router
from app.api.admin_models import router as admin_models_router
from app.api.admin_prompts import router as admin_prompts_router

# 导入数据库初始化函数。
# 这个函数通常负责创建数据库表或完成数据库启动检查。
from app.database import init_database


# asynccontextmanager 表示下面这个函数负责管理应用的生命周期。

@asynccontextmanager
async def lifespan(_: FastAPI):
    # FastAPI 应用启动时执行数据库初始化。
    init_database()

    # yield 之前的代码在应用启动时执行，
    # yield 之后的代码会在应用关闭时执行。
    # 当前 yield 后面没有代码，所以关闭时不执行额外操作。
    yield


# 创建 FastAPI 应用对象。
#
# title：接口文档中显示的项目名称。
# version：当前后端版本号。
# lifespan：指定应用启动和关闭时使用上面的 lifespan 函数。
app = FastAPI(title="资源推动 Agent Demo", version="0.1.0", lifespan=lifespan)

# 给 FastAPI 应用添加 CORS 中间件。中间件可以理解为：请求正式进入接口之前，统一经过的一层处理。
app.add_middleware(
    CORSMiddleware,

    # 只允许这个前端地址访问后端。
    # 一般表示本地运行在 3000 端口的前端项目。
    allow_origins=["http://localhost:3000"],

    # 只允许前端使用 GET 和 POST 请求。
    # GET 通常用于查询数据，POST 通常用于提交数据。
    allow_methods=["GET", "POST"],

    # 允许前端请求携带任意请求头。
    # 请求头中可能包含身份信息、数据格式等内容。
    allow_headers=["*"],
)

# 把任务相关接口注册到 FastAPI 应用中。
# 注册后，tasks_router 中定义的接口才能被外部访问。
app.include_router(tasks_router)

# 把信息采集相关接口注册到 FastAPI 应用中。
app.include_router(intake_router)
app.include_router(admin_models_router)
app.include_router(admin_prompts_router)


# 定义一个 GET 请求接口，访问路径是 /health。
# 例如浏览器访问：http://后端地址/health
@app.get("/health")
def health() -> dict[str, str]:
    # 返回一个 JSON 格式的数据。
    # 如果能够正常返回 {"status": "ok"}，
    # 一般说明后端服务已经成功启动，可以正常响应请求。
    return {"status": "ok"}
