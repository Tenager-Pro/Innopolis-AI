-- Create database
SELECT 'CREATE DATABASE customer_analysis_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'customer_analysis_db');

-- Create tables for data layers
CREATE TABLE IF NOT EXISTS raw_layer (
    CustomerID INT,
    Churn INT,
    Tenure INT,
    PreferredLoginDevice VARCHAR(50),
    CityTier INT,
    WarehouseToHome NUMERIC,
    PreferredPaymentMode VARCHAR(50),
    Gender VARCHAR(10),
    HourSpendOnApp NUMERIC,
    NumberOfDeviceRegistered INT,
    PreferedOrderCat VARCHAR(50),
    SatisfactionScore INT,
    MaritalStatus VARCHAR(20),
    NumberOfAddress INT,
    Complain INT,
    OrderAmountHikeFromlastYear NUMERIC,
    CouponUsed INT,
    OrderCount INT,
    DaySinceLastOrder NUMERIC,
    CashbackAmount NUMERIC
);

CREATE TABLE IF NOT EXISTS cleaned_layer (
    CustomerID INT,
    Churn INT,
    Tenure INT,
    PreferredLoginDevice VARCHAR(50),
    CityTier INT,
    WarehouseToHome NUMERIC,
    PreferredPaymentMode VARCHAR(50),
    Gender VARCHAR(10),
    HourSpendOnApp NUMERIC,
    NumberOfDeviceRegistered INT,
    PreferedOrderCat VARCHAR(50),
    SatisfactionScore INT,
    MaritalStatus VARCHAR(20),
    NumberOfAddress INT,
    Complain INT,
    OrderAmountHikeFromlastYear NUMERIC,
    CouponUsed INT,
    OrderCount INT,
    DaySinceLastOrder NUMERIC,
    CashbackAmount NUMERIC,
    AvgOrdersPerMonth NUMERIC,
    AvgCashbackPerOrder NUMERIC
);

CREATE TABLE IF NOT EXISTS features_layer (
    CustomerID INT,
    Churn INT,
    Tenure NUMERIC,
    CityTier NUMERIC,
    WarehouseToHome NUMERIC,
    HourSpendOnApp NUMERIC,
    NumberOfDeviceRegistered NUMERIC,
    SatisfactionScore NUMERIC,
    NumberOfAddress NUMERIC,
    Complain NUMERIC,
    OrderAmountHikeFromlastYear NUMERIC,
    CouponUsed NUMERIC,
    OrderCount NUMERIC,
    DaySinceLastOrder NUMERIC,
    CashbackAmount NUMERIC,
    AvgOrdersPerMonth NUMERIC,
    AvgCashbackPerOrder NUMERIC,
    EngagementScore NUMERIC,
    SatisfactionComplainRatio NUMERIC
);
