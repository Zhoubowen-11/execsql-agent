# FSQ 上海地点数据

## 数据来源与快照

本目录保存基于 **FSQ OS Places** 构建的上海地点业务数据库。地点和分类数据来自
`foursquare/fsq-os-places` 的 **2026-07-09** 快照。FSQ 官方当前通过 Places Portal
提供 Places、Categories 等数据集；源数据采用 Apache License 2.0，使用或再分发时应保留
上游版权、许可证和 notice。参见 [FSQ OS Places Open Source](https://docs.foursquare.com/data-products/docs/fsq-places-open-source)
和 [数据访问说明](https://docs.foursquare.com/data-products/docs/access-fsq-os-places)。

本项目不提交全球原始 FSQ 分片。重建需要自行取得对应快照，并将分类文件放到 README
所示路径。清洗完成的上海地点文件为：

- `processed/shanghai_places_filtered.parquet`
- `raw/release/dt=2026-07-09/categories/parquet/categories_000000.parquet`

## 上海筛选与已验证范围

现有 staging 痕迹表明上游流程逐个扫描了 100 个全球分片并生成上海候选分片，但候选筛选
和精筛脚本未随本阶段输入提供，因此不能可靠重建或声称其精确空间谓词。最终 Parquet 可验证
的精筛后置条件如下：

- 91,770 条唯一 `fsq_place_id`；
- `country = 'CN'`，经纬度均非空；
- 实际坐标范围为经度 120.895815～121.977995、纬度 30.696284～31.891929；
- 日期字段可规范化为 ISO 日期；
- 4,615 个地点无分类，其余分类数组均能关联到分类快照。

这组条件用于构建后质量验证，但不替代缺失的上游候选/行政区或多边形筛选实现。

## 真实 Schema 差异

地点分类不是单个 ID，而是 `fsq_category_ids VARCHAR[]`，并有平行的
`fsq_category_labels VARCHAR[]`；数据库将其拆为 `place_categories`。源数据没有显式
`is_primary`，因此仅将数组第一项标记为推断主分类。`geom` 是 WKB 点，`bbox` 是结构体，
数据库保留可直接查询的 `latitude`/`longitude`，不重复保存等价点几何。
`unresolved_flags VARCHAR[]` 被拆为 `place_unresolved_flags`，没有序列化为文本。

## SQLite 表

- `places`：地点主体和标量业务字段；
- `categories`：1～6 级分类、完整路径和派生父分类；
- `place_categories`：地点与分类多对多关联，数组第一项标记 `is_primary=1`；
- `place_unresolved_flags`：可查询的未解决标记；
- `data_metadata`：输入、快照、构建时间、行数、过滤说明和许可证 notice。

## 重建

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe scripts\build_fsq_shanghai_db.py `
  --places data\fsq\processed\shanghai_places_filtered.parquet `
  --categories data\fsq\raw\release\dt=2026-07-09\categories\parquet\categories_000000.parquet `
  --output data\fsq\shanghai_places.db
```

脚本先构建临时数据库，使用事务完成导入、外键和索引，再运行完整质量检查；只有全部通过后
才原子替换目标。生成的 `shanghai_places.db` 约 47 MiB，已明确加入 `.gitignore`，仓库不
依赖一个未说明的本地数据库文件，始终可由上述命令重建。构建结果见
[`reports/build_report.md`](reports/build_report.md) 和 [`reports/build_report.json`](reports/build_report.json)。
