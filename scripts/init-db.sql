-- NitroAGI Database Initialization Script
-- PostgreSQL database schema for structured data storage

-- Create database if not exists (run as superuser)
-- CREATE DATABASE nitroagi;

-- Use the nitroagi database
\c nitroagi;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS modules;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Set search path
SET search_path TO core, modules, analytics, public;

-- =====================================================
-- Core Tables
-- =====================================================

-- Users table
CREATE TABLE IF NOT EXISTS core.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true
);

CREATE INDEX idx_users_username ON core.users(username);
CREATE INDEX idx_users_email ON core.users(email);
CREATE INDEX idx_users_metadata ON core.users USING gin(metadata);

-- Sessions table
CREATE TABLE IF NOT EXISTS core.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES core.users(id) ON DELETE CASCADE,
    session_token VARCHAR(512) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    ip_address INET,
    user_agent TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_sessions_user_id ON core.sessions(user_id);
CREATE INDEX idx_sessions_token ON core.sessions(session_token);
CREATE INDEX idx_sessions_expires ON core.sessions(expires_at);

-- Conversations table
CREATE TABLE IF NOT EXISTS core.conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES core.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES core.sessions(id) ON DELETE SET NULL,
    title VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    is_archived BOOLEAN DEFAULT false
);

CREATE INDEX idx_conversations_user_id ON core.conversations(user_id);
CREATE INDEX idx_conversations_session_id ON core.conversations(session_id);
CREATE INDEX idx_conversations_created ON core.conversations(created_at DESC);

-- Messages table
CREATE TABLE IF NOT EXISTS core.messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES core.conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tokens_used INTEGER,
    model_used VARCHAR(100),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_messages_conversation ON core.messages(conversation_id);
CREATE INDEX idx_messages_created ON core.messages(created_at);
CREATE INDEX idx_messages_metadata ON core.messages USING gin(metadata);

-- =====================================================
-- Memory System Tables
-- =====================================================

-- Memory items table
CREATE TABLE IF NOT EXISTS core.memory_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key VARCHAR(500) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    memory_type VARCHAR(50) NOT NULL CHECK (memory_type IN ('working', 'episodic', 'semantic')),
    tier VARCHAR(50) DEFAULT 'hot' CHECK (tier IN ('hot', 'warm', 'cold', 'archive')),
    user_id UUID REFERENCES core.users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1,
    importance FLOAT DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    ttl_seconds INTEGER,
    expires_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_memory_key ON core.memory_items(key);
CREATE INDEX idx_memory_type ON core.memory_items(memory_type);
CREATE INDEX idx_memory_tier ON core.memory_items(tier);
CREATE INDEX idx_memory_user ON core.memory_items(user_id);
CREATE INDEX idx_memory_importance ON core.memory_items(importance DESC);
CREATE INDEX idx_memory_expires ON core.memory_items(expires_at);
CREATE INDEX idx_memory_value ON core.memory_items USING gin(value);

-- Memory associations table (for semantic links)
CREATE TABLE IF NOT EXISTS core.memory_associations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_memory_id UUID REFERENCES core.memory_items(id) ON DELETE CASCADE,
    target_memory_id UUID REFERENCES core.memory_items(id) ON DELETE CASCADE,
    association_type VARCHAR(100),
    strength FLOAT DEFAULT 0.5 CHECK (strength >= 0 AND strength <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    UNIQUE(source_memory_id, target_memory_id)
);

CREATE INDEX idx_associations_source ON core.memory_associations(source_memory_id);
CREATE INDEX idx_associations_target ON core.memory_associations(target_memory_id);
CREATE INDEX idx_associations_strength ON core.memory_associations(strength DESC);

-- =====================================================
-- Module Tables
-- =====================================================

-- Module registry table
CREATE TABLE IF NOT EXISTS modules.registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    capabilities TEXT[],
    status VARCHAR(50) DEFAULT 'inactive',
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_modules_name ON modules.registry(name);
CREATE INDEX idx_modules_status ON modules.registry(status);
CREATE INDEX idx_modules_capabilities ON modules.registry USING gin(capabilities);

-- Module execution logs
CREATE TABLE IF NOT EXISTS modules.execution_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    module_id UUID REFERENCES modules.registry(id) ON DELETE CASCADE,
    request_id UUID NOT NULL,
    user_id UUID REFERENCES core.users(id) ON DELETE SET NULL,
    input_data JSONB,
    output_data JSONB,
    status VARCHAR(50),
    processing_time_ms FLOAT,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_execution_module ON modules.execution_logs(module_id);
CREATE INDEX idx_execution_request ON modules.execution_logs(request_id);
CREATE INDEX idx_execution_user ON modules.execution_logs(user_id);
CREATE INDEX idx_execution_created ON modules.execution_logs(created_at DESC);
CREATE INDEX idx_execution_status ON modules.execution_logs(status);

-- =====================================================
-- Analytics Tables
-- =====================================================

-- System metrics table
CREATE TABLE IF NOT EXISTS analytics.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(50),
    tags JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_metrics_name ON analytics.system_metrics(metric_name);
CREATE INDEX idx_metrics_recorded ON analytics.system_metrics(recorded_at DESC);
CREATE INDEX idx_metrics_tags ON analytics.system_metrics USING gin(tags);

-- API request logs
CREATE TABLE IF NOT EXISTS analytics.api_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID UNIQUE NOT NULL,
    user_id UUID REFERENCES core.users(id) ON DELETE SET NULL,
    endpoint VARCHAR(500) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms FLOAT,
    ip_address INET,
    user_agent TEXT,
    request_body JSONB,
    response_body JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_api_request_id ON analytics.api_requests(request_id);
CREATE INDEX idx_api_user ON analytics.api_requests(user_id);
CREATE INDEX idx_api_endpoint ON analytics.api_requests(endpoint);
CREATE INDEX idx_api_created ON analytics.api_requests(created_at DESC);
CREATE INDEX idx_api_status ON analytics.api_requests(status_code);

-- Network performance metrics
CREATE TABLE IF NOT EXISTS analytics.network_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    connection_type VARCHAR(50),
    latency_ms FLOAT,
    bandwidth_mbps FLOAT,
    packet_loss FLOAT,
    jitter_ms FLOAT,
    network_profile VARCHAR(100),
    is_6g_enabled BOOLEAN DEFAULT false,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_network_recorded ON analytics.network_metrics(recorded_at DESC);
CREATE INDEX idx_network_6g ON analytics.network_metrics(is_6g_enabled);

-- =====================================================
-- Functions and Triggers
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON core.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON core.conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_modules_updated_at BEFORE UPDATE ON modules.registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to update memory access statistics
CREATE OR REPLACE FUNCTION update_memory_access()
RETURNS TRIGGER AS $$
BEGIN
    NEW.accessed_at = CURRENT_TIMESTAMP;
    NEW.access_count = OLD.access_count + 1;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Function to clean expired memories
CREATE OR REPLACE FUNCTION clean_expired_memories()
RETURNS void AS $$
BEGIN
    DELETE FROM core.memory_items
    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP;
END;
$$ language 'plpgsql';

-- Function to archive old conversations
CREATE OR REPLACE FUNCTION archive_old_conversations()
RETURNS void AS $$
BEGIN
    UPDATE core.conversations
    SET is_archived = true
    WHERE updated_at < CURRENT_TIMESTAMP - INTERVAL '30 days'
    AND is_archived = false;
END;
$$ language 'plpgsql';

-- =====================================================
-- Views
-- =====================================================

-- Active sessions view
CREATE OR REPLACE VIEW core.active_sessions AS
SELECT 
    s.id,
    s.user_id,
    u.username,
    s.created_at,
    s.expires_at,
    s.ip_address
FROM core.sessions s
JOIN core.users u ON s.user_id = u.id
WHERE s.expires_at > CURRENT_TIMESTAMP OR s.expires_at IS NULL;

-- Memory statistics view
CREATE OR REPLACE VIEW core.memory_statistics AS
SELECT 
    memory_type,
    tier,
    COUNT(*) as item_count,
    AVG(importance) as avg_importance,
    SUM(access_count) as total_accesses,
    MAX(accessed_at) as last_accessed
FROM core.memory_items
GROUP BY memory_type, tier;

-- Module performance view
CREATE OR REPLACE VIEW modules.performance_stats AS
SELECT 
    m.name as module_name,
    COUNT(e.id) as execution_count,
    AVG(e.processing_time_ms) as avg_processing_time,
    MIN(e.processing_time_ms) as min_processing_time,
    MAX(e.processing_time_ms) as max_processing_time,
    SUM(CASE WHEN e.status = 'success' THEN 1 ELSE 0 END)::FLOAT / COUNT(e.id) as success_rate
FROM modules.registry m
LEFT JOIN modules.execution_logs e ON m.id = e.module_id
GROUP BY m.id, m.name;

-- =====================================================
-- Initial Data
-- =====================================================

-- Insert default modules
INSERT INTO modules.registry (name, version, description, capabilities, status, config)
VALUES 
    ('language', '1.0.0', 'Language processing module with multi-provider LLM support', 
     ARRAY['text_generation', 'text_understanding'], 'active', 
     '{"providers": ["openai", "anthropic", "huggingface"]}'),
    ('vision', '1.0.0', 'Computer vision and image processing module', 
     ARRAY['image_recognition', 'image_generation'], 'inactive', 
     '{}'),
    ('audio', '1.0.0', 'Audio processing and speech recognition module', 
     ARRAY['speech_to_text', 'text_to_speech'], 'inactive', 
     '{}')
ON CONFLICT (name) DO NOTHING;

-- Insert system user
INSERT INTO core.users (username, email, metadata)
VALUES ('system', 'system@nitroagi.local', '{"role": "system", "internal": true}')
ON CONFLICT (username) DO NOTHING;

-- =====================================================
-- Permissions
-- =====================================================

-- Grant permissions to nitroagi user
GRANT ALL PRIVILEGES ON SCHEMA core TO nitroagi;
GRANT ALL PRIVILEGES ON SCHEMA modules TO nitroagi;
GRANT ALL PRIVILEGES ON SCHEMA analytics TO nitroagi;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA core TO nitroagi;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA modules TO nitroagi;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO nitroagi;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA core TO nitroagi;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA modules TO nitroagi;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO nitroagi;

-- =====================================================
-- Maintenance Jobs (to be scheduled externally)
-- =====================================================

-- Job: Clean expired memories (run hourly)
-- SELECT clean_expired_memories();

-- Job: Archive old conversations (run daily)
-- SELECT archive_old_conversations();

-- Job: Vacuum and analyze tables (run weekly)
-- VACUUM ANALYZE;