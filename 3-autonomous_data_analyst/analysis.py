"""
Autonomous Data Analyst - Analysis Tools 

Tools for:
- Data profiling
- Statistical analysis
- Time series analysis
- Correlation analysis
- Outlier detection
- Insight generation
"""

from typing import Optional, Any
from dataclasses import dataclass, field
import json


@dataclass
class ColumnProfile:
    """Profile for a single column."""
    name: str
    dtype: str
    
    # Basic stats
    count: int = 0
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    unique_percentage: float = 0.0
    
    # Numeric stats
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    
    # String stats
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    
    # Distribution
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    
    # Quality
    quality_score: float = 1.0
    issues: list[str] = field(default_factory=list)


@dataclass
class DataProfile:
    """Complete data profile."""
    table_name: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    
    # Overall stats
    memory_usage_mb: float = 0
    duplicate_rows: int = 0
    completeness_score: float = 1.0
    
    # Recommendations
    recommendations: list[str] = field(default_factory=list)


@dataclass
class Insight:
    """A data insight."""
    title: str
    description: str
    importance: str  # high, medium, low
    category: str  # trend, anomaly, correlation, distribution, etc.
    metric_value: Optional[Any] = None
    comparison: Optional[str] = None


class DataProfiler:
    """
    Data profiling and quality analysis.
    """
    
    def profile_dataframe(self, df, table_name: str = "data") -> DataProfile:
        """Generate a comprehensive profile of a DataFrame."""
        import pandas as pd
        import numpy as np
        
        columns = []
        
        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            
            profile = ColumnProfile(
                name=col,
                dtype=dtype,
                count=len(series),
                null_count=series.isnull().sum(),
                null_percentage=series.isnull().sum() / len(series) * 100,
                unique_count=series.nunique(),
                unique_percentage=series.nunique() / len(series) * 100,
            )
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                if len(clean) > 0:
                    profile.mean = float(clean.mean())
                    profile.std = float(clean.std())
                    profile.min = float(clean.min())
                    profile.max = float(clean.max())
                    profile.median = float(clean.median())
                    profile.q25 = float(clean.quantile(0.25))
                    profile.q75 = float(clean.quantile(0.75))
            
            # String columns
            elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
                str_lengths = series.dropna().astype(str).str.len()
                if len(str_lengths) > 0:
                    profile.min_length = int(str_lengths.min())
                    profile.max_length = int(str_lengths.max())
                    profile.avg_length = float(str_lengths.mean())
            
            # Top values
            value_counts = series.value_counts().head(5)
            profile.top_values = [(str(k), int(v)) for k, v in value_counts.items()]
            
            # Quality issues
            if profile.null_percentage > 50:
                profile.issues.append(f"High null rate: {profile.null_percentage:.1f}%")
                profile.quality_score -= 0.3
            elif profile.null_percentage > 20:
                profile.issues.append(f"Moderate null rate: {profile.null_percentage:.1f}%")
                profile.quality_score -= 0.1
            
            if profile.unique_percentage > 99 and len(series) > 100:
                profile.issues.append("Possible ID column (very high cardinality)")
            
            columns.append(profile)
        
        # Overall profile
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        duplicates = df.duplicated().sum()
        completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        
        profile = DataProfile(
            table_name=table_name,
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            memory_usage_mb=memory_mb,
            duplicate_rows=duplicates,
            completeness_score=completeness,
        )
        
        # Recommendations
        if duplicates > 0:
            profile.recommendations.append(f"Found {duplicates} duplicate rows - consider deduplication")
        
        high_null_cols = [c.name for c in columns if c.null_percentage > 30]
        if high_null_cols:
            profile.recommendations.append(f"Columns with high null rates: {', '.join(high_null_cols)}")
        
        return profile
    
    def profile_to_text(self, profile: DataProfile) -> str:
        """Convert profile to readable text."""
        text = f"📊 Data Profile: {profile.table_name}\n"
        text += f"{'='*50}\n\n"
        
        text += f"Overview:\n"
        text += f"  • Rows: {profile.row_count:,}\n"
        text += f"  • Columns: {profile.column_count}\n"
        text += f"  • Memory: {profile.memory_usage_mb:.2f} MB\n"
        text += f"  • Completeness: {profile.completeness_score:.1%}\n"
        text += f"  • Duplicate rows: {profile.duplicate_rows:,}\n\n"
        
        text += "Columns:\n"
        for col in profile.columns:
            text += f"\n  📌 {col.name} ({col.dtype})\n"
            text += f"     Nulls: {col.null_count:,} ({col.null_percentage:.1f}%)\n"
            text += f"     Unique: {col.unique_count:,} ({col.unique_percentage:.1f}%)\n"
            
            if col.mean is not None:
                text += f"     Mean: {col.mean:.2f}, Std: {col.std:.2f}\n"
                text += f"     Range: [{col.min:.2f}, {col.max:.2f}]\n"
            
            if col.issues:
                text += f"     ⚠️ Issues: {', '.join(col.issues)}\n"
        
        if profile.recommendations:
            text += "\n💡 Recommendations:\n"
            for rec in profile.recommendations:
                text += f"  • {rec}\n"
        
        return text


class StatisticalAnalyzer:
    """
    Statistical analysis tools.
    """
    
    def compute_statistics(self, df, column: str) -> dict:
        """Compute comprehensive statistics for a column."""
        import pandas as pd
        import numpy as np
        
        series = df[column]
        
        stats = {
            "column": column,
            "count": len(series),
            "null_count": series.isnull().sum(),
        }
        
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            stats.update({
                "mean": float(clean.mean()),
                "std": float(clean.std()),
                "min": float(clean.min()),
                "max": float(clean.max()),
                "median": float(clean.median()),
                "q1": float(clean.quantile(0.25)),
                "q3": float(clean.quantile(0.75)),
                "iqr": float(clean.quantile(0.75) - clean.quantile(0.25)),
                "skewness": float(clean.skew()),
                "kurtosis": float(clean.kurtosis()),
                "variance": float(clean.var()),
            })
        
        return stats
    
    def compute_correlations(self, df, method: str = "pearson") -> dict:
        """Compute correlation matrix for numeric columns."""
        import pandas as pd
        
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.shape[1] < 2:
            return {"error": "Need at least 2 numeric columns"}
        
        corr_matrix = numeric_df.corr(method=method)
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    strong_correlations.append({
                        "column1": corr_matrix.columns[i],
                        "column2": corr_matrix.columns[j],
                        "correlation": float(corr),
                        "strength": "strong" if abs(corr) > 0.9 else "moderate",
                    })
        
        return {
            "matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations,
        }
    
    def detect_outliers(self, df, column: str, method: str = "iqr") -> dict:
        """Detect outliers in a column."""
        import pandas as pd
        import numpy as np
        
        series = df[column].dropna()
        
        if not pd.api.types.is_numeric_dtype(series):
            return {"error": f"Column {column} is not numeric"}
        
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers_mask = (series < lower_bound) | (series > upper_bound)
            
        elif method == "zscore":
            z_scores = (series - series.mean()) / series.std()
            outliers_mask = abs(z_scores) > 3
            lower_bound = series.mean() - 3 * series.std()
            upper_bound = series.mean() + 3 * series.std()
        
        else:
            return {"error": f"Unknown method: {method}"}
        
        outliers = series[outliers_mask]
        
        return {
            "column": column,
            "method": method,
            "outlier_count": len(outliers),
            "outlier_percentage": len(outliers) / len(series) * 100,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_values": outliers.tolist()[:20],  # First 20
        }
    
    def group_statistics(self, df, group_column: str, value_column: str) -> dict:
        """Compute statistics grouped by a column."""
        import pandas as pd
        
        if group_column not in df.columns or value_column not in df.columns:
            return {"error": "Column not found"}
        
        grouped = df.groupby(group_column)[value_column].agg([
            'count', 'mean', 'std', 'min', 'max', 'median', 'sum'
        ]).round(2)
        
        return {
            "group_column": group_column,
            "value_column": value_column,
            "groups": grouped.to_dict('index'),
        }


class TimeSeriesAnalyzer:
    """
    Time series analysis tools.
    """
    
    def analyze_time_series(
        self, 
        df, 
        date_column: str, 
        value_column: str,
        freq: str = "auto"
    ) -> dict:
        """Analyze a time series."""
        import pandas as pd
        import numpy as np
        
        # Convert to datetime
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        
        # Detect frequency
        if freq == "auto":
            diff = df[date_column].diff().median()
            if diff <= pd.Timedelta(days=1):
                freq = "D"
            elif diff <= pd.Timedelta(days=7):
                freq = "W"
            else:
                freq = "M"
        
        # Basic stats
        result = {
            "date_column": date_column,
            "value_column": value_column,
            "frequency": freq,
            "start_date": str(df[date_column].min()),
            "end_date": str(df[date_column].max()),
            "num_periods": len(df),
        }
        
        # Trend analysis
        values = df[value_column].values
        if len(values) > 1:
            # Simple linear trend
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            result["trend"] = {
                "slope": float(slope),
                "direction": "increasing" if slope > 0 else "decreasing",
                "magnitude": "strong" if abs(slope) > values.std() else "weak",
            }
        
        # Seasonality detection (simplified)
        if len(df) >= 12:
            monthly = df.set_index(date_column)[value_column].resample('M').mean()
            if len(monthly) >= 12:
                result["seasonality"] = {
                    "detected": monthly.std() > values.std() * 0.5,
                    "monthly_pattern": monthly.tail(12).to_dict(),
                }
        
        # Growth rates
        df['pct_change'] = df[value_column].pct_change() * 100
        result["growth"] = {
            "mean_growth_rate": float(df['pct_change'].mean()),
            "total_growth": float((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else None,
        }
        
        return result
    
    def compare_periods(
        self, 
        df, 
        date_column: str, 
        value_column: str,
        period1_start: str,
        period1_end: str,
        period2_start: str,
        period2_end: str,
    ) -> dict:
        """Compare two time periods."""
        import pandas as pd
        
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        period1 = df[(df[date_column] >= period1_start) & (df[date_column] <= period1_end)]
        period2 = df[(df[date_column] >= period2_start) & (df[date_column] <= period2_end)]
        
        return {
            "period1": {
                "start": period1_start,
                "end": period1_end,
                "sum": float(period1[value_column].sum()),
                "mean": float(period1[value_column].mean()),
                "count": len(period1),
            },
            "period2": {
                "start": period2_start,
                "end": period2_end,
                "sum": float(period2[value_column].sum()),
                "mean": float(period2[value_column].mean()),
                "count": len(period2),
            },
            "comparison": {
                "sum_change": float(period2[value_column].sum() - period1[value_column].sum()),
                "sum_change_pct": float((period2[value_column].sum() - period1[value_column].sum()) / period1[value_column].sum() * 100) if period1[value_column].sum() != 0 else None,
                "mean_change": float(period2[value_column].mean() - period1[value_column].mean()),
                "mean_change_pct": float((period2[value_column].mean() - period1[value_column].mean()) / period1[value_column].mean() * 100) if period1[value_column].mean() != 0 else None,
            },
        }


class InsightGenerator:
    """
    Automatic insight generation.
    """
    
    def __init__(self):
        self.profiler = DataProfiler()
        self.stats = StatisticalAnalyzer()
        self.timeseries = TimeSeriesAnalyzer()
    
    def generate_insights(self, df, table_name: str = "data") -> list[Insight]:
        """Generate automatic insights from a DataFrame."""
        import pandas as pd
        import numpy as np
        
        insights = []
        
        # Profile the data
        profile = self.profiler.profile_dataframe(df, table_name)
        
        # Data quality insights
        if profile.duplicate_rows > 0:
            insights.append(Insight(
                title="Duplicate Rows Detected",
                description=f"Found {profile.duplicate_rows:,} duplicate rows ({profile.duplicate_rows/profile.row_count*100:.1f}% of data)",
                importance="medium",
                category="quality",
                metric_value=profile.duplicate_rows,
            ))
        
        # High cardinality columns (potential IDs)
        for col in profile.columns:
            if col.unique_percentage > 95 and profile.row_count > 100:
                insights.append(Insight(
                    title=f"High Cardinality: {col.name}",
                    description=f"Column '{col.name}' has {col.unique_percentage:.1f}% unique values - likely an ID column",
                    importance="low",
                    category="distribution",
                ))
        
        # Missing data insights
        high_null_cols = [c for c in profile.columns if c.null_percentage > 20]
        if high_null_cols:
            insights.append(Insight(
                title="Significant Missing Data",
                description=f"{len(high_null_cols)} columns have >20% missing values: {', '.join(c.name for c in high_null_cols)}",
                importance="high",
                category="quality",
            ))
        
        # Numeric column insights
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            # Skewness
            skew = series.skew()
            if abs(skew) > 2:
                direction = "right" if skew > 0 else "left"
                insights.append(Insight(
                    title=f"Skewed Distribution: {col}",
                    description=f"'{col}' is heavily {direction}-skewed (skewness: {skew:.2f}). Consider transformation.",
                    importance="medium",
                    category="distribution",
                    metric_value=skew,
                ))
            
            # Outliers
            outlier_result = self.stats.detect_outliers(df, col)
            if outlier_result.get("outlier_percentage", 0) > 5:
                insights.append(Insight(
                    title=f"Many Outliers: {col}",
                    description=f"'{col}' has {outlier_result['outlier_percentage']:.1f}% outliers ({outlier_result['outlier_count']} values)",
                    importance="medium",
                    category="anomaly",
                    metric_value=outlier_result['outlier_count'],
                ))
        
        # Correlation insights
        if len(numeric_cols) >= 2:
            corr_result = self.stats.compute_correlations(df)
            for corr in corr_result.get("strong_correlations", []):
                insights.append(Insight(
                    title=f"Strong Correlation Found",
                    description=f"'{corr['column1']}' and '{corr['column2']}' are {corr['strength']}ly correlated (r={corr['correlation']:.2f})",
                    importance="high" if abs(corr['correlation']) > 0.9 else "medium",
                    category="correlation",
                    metric_value=corr['correlation'],
                ))
        
        # Date column detection and time series insights
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Also check object columns that might be dates
        for col in df.select_dtypes(include=['object']).columns:
            try:
                pd.to_datetime(df[col].dropna().head(100))
                date_cols.append(col)
            except:
                pass
        
        if date_cols and numeric_cols:
            date_col = date_cols[0]
            value_col = numeric_cols[0]
            
            try:
                ts_result = self.timeseries.analyze_time_series(df, date_col, value_col)
                
                if ts_result.get("trend"):
                    trend = ts_result["trend"]
                    insights.append(Insight(
                        title=f"Time Series Trend",
                        description=f"'{value_col}' shows a {trend['magnitude']} {trend['direction']} trend over time",
                        importance="high",
                        category="trend",
                        metric_value=trend['slope'],
                    ))
            except Exception:
                pass
        
        # Sort by importance
        importance_order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda x: importance_order.get(x.importance, 3))
        
        return insights
    
    def insights_to_text(self, insights: list[Insight]) -> str:
        """Convert insights to readable text."""
        if not insights:
            return "No significant insights found."
        
        text = "💡 Key Insights:\n\n"
        
        for i, insight in enumerate(insights, 1):
            emoji = {
                "quality": "🔍",
                "distribution": "📊",
                "correlation": "🔗",
                "anomaly": "⚠️",
                "trend": "📈",
            }.get(insight.category, "•")
            
            importance_badge = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢",
            }.get(insight.importance, "")
            
            text += f"{i}. {emoji} {importance_badge} {insight.title}\n"
            text += f"   {insight.description}\n\n"
        
        return text
