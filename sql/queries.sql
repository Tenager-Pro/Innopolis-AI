-- Data quality checks
SELECT 
    'raw_layer' as layer,
    COUNT(*) as record_count,
    COUNT(DISTINCT CustomerID) as unique_customers,
    SUM(CASE WHEN Churn IS NULL THEN 1 ELSE 0 END) as null_churn
FROM raw_layer

UNION ALL

SELECT 
    'cleaned_layer' as layer,
    COUNT(*) as record_count,
    COUNT(DISTINCT CustomerID) as unique_customers,
    SUM(CASE WHEN Churn IS NULL THEN 1 ELSE 0 END) as null_churn
FROM cleaned_layer

UNION ALL

SELECT 
    'features_layer' as layer,
    COUNT(*) as record_count,
    COUNT(DISTINCT CustomerID) as unique_customers,
    SUM(CASE WHEN Churn IS NULL THEN 1 ELSE 0 END) as null_churn
FROM features_layer;

-- Churn distribution analysis
SELECT 
    layer,
    Churn,
    count,
    ROUND(count * 100.0 / SUM(count) OVER(PARTITION BY layer), 2) as percentage
FROM (
    SELECT 
        'raw_layer' as layer,
        Churn,
        COUNT(*) as count
    FROM raw_layer 
    GROUP BY Churn
    
    UNION ALL
    
    SELECT 
        'cleaned_layer' as layer,
        Churn,
        COUNT(*) as count
    FROM cleaned_layer 
    GROUP BY Churn
    
    UNION ALL
    
    SELECT 
        'features_layer' as layer,
        Churn,
        COUNT(*) as count
    FROM features_layer 
    GROUP BY Churn
) AS churn_stats
ORDER BY layer, Churn;

-- Feature statistics
SELECT 
    'Tenure' as feature,
    AVG(Tenure) as avg_value,
    STDDEV(Tenure) as std_value,
    MIN(Tenure) as min_value,
    MAX(Tenure) as max_value
FROM features_layer

UNION ALL

SELECT 
    'OrderCount',
    AVG(OrderCount),
    STDDEV(OrderCount),
    MIN(OrderCount),
    MAX(OrderCount)
FROM features_layer

UNION ALL

SELECT 
    'CashbackAmount',
    AVG(CashbackAmount),
    STDDEV(CashbackAmount),
    MIN(CashbackAmount),
    MAX(CashbackAmount)
FROM features_layer;
