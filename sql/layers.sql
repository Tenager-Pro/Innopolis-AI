-- Create views for each data layer
CREATE OR REPLACE VIEW raw_data_view AS
SELECT * FROM raw_layer;

CREATE OR REPLACE VIEW cleaned_data_view AS
SELECT * FROM cleaned_layer;

CREATE OR REPLACE VIEW features_data_view AS
SELECT * FROM features_layer;

-- Create ML ready view
CREATE OR REPLACE VIEW ml_ready_features AS
SELECT * FROM features_layer WHERE public.features_layer."CustomerID" IS NOT NULL;

-- Add comments
COMMENT ON TABLE raw_layer IS 'Raw customer data without any processing';
COMMENT ON TABLE cleaned_layer IS 'Cleaned data after handling missing values, duplicates and outliers';
COMMENT ON TABLE features_layer IS 'Final feature set ready for machine learning';
COMMENT ON VIEW ml_ready_features IS 'Preprocessed features for ML models';
