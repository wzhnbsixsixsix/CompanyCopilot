from __future__ import annotations

from pydantic import BaseModel, Field


class DueDiligenceSummary(BaseModel):
    company_name: str = Field(description="Company name")
    founded_year: str = Field(description="Founded year if available")
    headquarters: str = Field(description="Headquarters location")
    core_business: str = Field(description="Main business summary")
    executives: list[str] = Field(description="Key executives")
    risk_signals: list[str] = Field(
        description="Potential risk signals and negative findings"
    )
    source_urls: list[str] = Field(description="Information source URLs")


class CompanyReport(BaseModel):
    """Comprehensive 8-section company background investigation report."""

    company_name: str = Field(description="公司名称")
    overview: str = Field(
        description="第1章：公司概览与基础信息，包括成立时间、总部、行业、使命愿景"
    )
    products: str = Field(
        description="第2章：产品与服务体系，包括主要产品线、商业模式、定价策略"
    )
    market: str = Field(
        description="第3章：市场表现与受众分析，包括访问量、市场份额、用户画像"
    )
    personnel: str = Field(
        description="第4章：关键人员与组织架构，包括高管团队、创始人背景"
    )
    financials: str = Field(
        description="第5章：融资历史与财务状况，包括融资轮次、投资方、营收"
    )
    tech_stack: str = Field(
        description="第6章：技术栈与数字化水平，包括技术架构、基础设施"
    )
    news: str = Field(
        description="第7章：近期动态与发展趋势，包括最新新闻、产品发布、战略调整"
    )
    competitors: str = Field(
        description="第8章：竞争格局与市场地位，包括主要竞争对手、差异化优势"
    )
    risk_signals: list[str] = Field(
        description="风险信号列表，包括法律、财务、声誉等风险"
    )
    source_urls: list[str] = Field(description="信息来源URL列表")
    report: str = Field(description="完整的Markdown格式背调报告，包含所有8个章节的内容")
