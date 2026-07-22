# 下一步工作记录

记录时间：2026-07-22 08:10（Asia/Shanghai）

## 当前已完成

- 人物和企业由大模型直接输出 `CONFIRMED`、`NEEDS_CONFIRMATION` 或 `MISSING`。
- 程序不再使用固定置信度阈值决定是否接受实体，只校验姓名、企业和证据是否真实出现在原文中。
- 规则提取仅作为大模型调用失败时的降级结果，不再与正常模型结果重复竞争。
- 删除实体解析器的 seed 和阈值依赖。
- 删除 MCP 的 `resolve_entities` 工具、客户端及内部实体别名仓库代码。
- 删除新数据库初始化中的 `entity_aliases` 表和数据写入逻辑；现有数据库中的旧表暂未删除。
- 删除人物和企业同时出现时自动建立所属关系的逻辑。只有模型明确给出关系或用户确认补充时才建立关系。
- 保留旧任务数据兼容：旧 `needs_confirmation` 字段会在读取时转换为新的 `resolution`。
- 后端测试 35 项通过，前端 TypeScript 检查和 Docker 镜像构建通过。

## 当前运行状态

- Docker 运行中的容器仍是修改前的版本，尚未切换到刚构建的新镜像。
- 新版 backend、worker、mcp-server、frontend 和 seed 镜像已经构建完成。
- 为避免中断旧任务 `35d065bd-fe13-40b6-996b-99a3c64671cf`，尚未重启容器。

## 恢复后按顺序执行

1. 检查旧任务是否结束：
   `docker compose exec -T postgres psql -U resource_agent -d resource_agent -Atc "SELECT id,status,updated_at FROM research_tasks WHERE id='35d065bd-fe13-40b6-996b-99a3c64671cf';"`
2. 旧任务结束后切换到已构建镜像：
   `docker compose up -d --no-deps backend worker mcp-server frontend`
3. 检查所有服务健康：
   `docker compose ps`
4. 使用固定文本完成实际端到端测试：
   `我晚上去和新城水务的方正赴宴，讨论水务管网监测项目的现场部署问题。`
5. 验收重点：大模型应将 `方正`、`新城水务` 和二者关系输出为 `CONFIRMED`；任务不应停在人物确认页。
6. 再测试 `新城水务和方正赴宴`：人物可以确认，但不得自动写成方正属于新城水务。
7. 再测试 `新城水务的方总`：应进入联网候选确认流程。
8. 验证旧任务 API 仍可读取，确认旧字段迁移有效。
9. 验收通过后删除本文件或更新为最终结果，提交并合并到 `main`，然后推送远程。

## 尚未执行

- 尚未用真实大模型验证新 `resolution` 结构化输出。
- 尚未重启服务加载新代码。
- 尚未把当前 WIP 合并到 `main`。
