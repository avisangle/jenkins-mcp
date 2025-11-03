# Jenkins MCP Server

A comprehensive Model Context Protocol (MCP) server for Jenkins CI/CD automation. Provides 20+ tools for complete Jenkins management including job control, build monitoring, pipeline orchestration, artifact handling, and batch operations.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-1.11+-green.svg)](https://github.com/modelcontextprotocol)

## Features

### Core Capabilities

- **Job Management** (8 tools)
  - Trigger jobs with parameters (supports multiselect)
  - List jobs with recursive folder traversal
  - Search jobs with wildcard/regex patterns
  - Get detailed job information with auto-search fallback
  - Support for nested job paths (`folder1/subfolder/jobname`)
  - Advanced filtering by status, build result, and activity

- **Build Operations** (4 tools)
  - Monitor build status in real-time
  - Retrieve console logs with progressive fetching
  - Get build queue information
  - LLM-ready log summarization support

- **Folder & Search** (4 tools)
  - Explore folder hierarchies
  - Pattern-based job discovery
  - Search and trigger workflows
  - Multi-criteria filtering

- **Pipeline Operations** (2 tools)
  - Detailed pipeline stage tracking
  - Stage-by-stage status and timing
  - Stage log retrieval
  - Support for Declarative and Scripted pipelines

- **Artifact Management** (3 tools)
  - List build artifacts with metadata
  - Download text-based artifacts
  - Search artifacts across builds with patterns

- **Batch Operations** (3 tools)
  - Parallel job triggering with priority queue
  - Batch monitoring and progress tracking
  - Batch cancellation support

- **Cache Management** (3 tools)
  - Multi-tier caching statistics
  - Fine-grained cache control
  - Cache warming utilities

- **Server Operations** (3 tools)
  - Health checks
  - Server information
  - Configuration management

### Advanced Features

#### Multi-Tier Caching System
- **Static Cache** (1 hour): Server info, job configurations
- **Semi-Static Cache** (5 minutes): Job lists, queue data
- **Dynamic Cache** (30 seconds): Running build statuses
- **Permanent Cache**: Completed builds (LRU-based)
- **Short-lived Cache** (10 seconds): Console logs, active pipelines

#### Resilient Operations
- Exponential backoff retry logic with jitter
- Configurable retry attempts (default: 3)
- Smart retry for transient failures (429, 500-504)
- CSRF crumb token management with caching

#### Security & Reliability
- Jenkins API token authentication
- CSRF protection for POST operations
- Comprehensive error handling with actionable suggestions
- Request timeout controls
- Content size limits for safety

## Prerequisites

- **Python**: 3.12 or higher
- **Jenkins**: Any modern version with API access
- **Network**: Access to Jenkins server URL
- **Credentials**: Jenkins username and API token

## Installation

### Option 1: Docker (Recommended)

```bash
# Pull from Docker Hub
docker pull mcp/jenkins

# Run with environment variables
docker run -d \
  -e JENKINS_URL=http://your-jenkins:8080 \
  -e JENKINS_USER=your-username \
  -e JENKINS_API_TOKEN=your-token \
  -p 8010:8010 \
  mcp/jenkins
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/avisangle/jenkins-mcp.git
cd jenkins-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Create .env file
cp .env.example .env
# Edit .env with your Jenkins credentials

# Run the server
python jenkins_mcp_server_enhanced.py
```

### Option 3: Using uv

```bash
# Install with uv
uv pip install -e .

# Run
python jenkins_mcp_server_enhanced.py
```

## Configuration

### Required Environment Variables

```bash
JENKINS_URL=http://localhost:8080        # Your Jenkins server URL
JENKINS_USER=admin                       # Jenkins username
JENKINS_API_TOKEN=your-api-token-here    # Jenkins API token
```

**Getting Jenkins API Token:**
1. Log into Jenkins
2. Click your name (top right) → Configure
3. API Token section → Add new Token
4. Copy the generated token

### Optional Environment Variables

#### MCP Server Settings
```bash
MCP_PORT=8010                  # Server port (default: 8010)
MCP_HOST=0.0.0.0              # Server host (default: 0.0.0.0)
```

#### Retry Configuration
```bash
JENKINS_MAX_RETRIES=3                    # Max retry attempts
JENKINS_RETRY_BASE_DELAY=1.0            # Initial delay (seconds)
JENKINS_RETRY_MAX_DELAY=60.0            # Max delay (seconds)
JENKINS_RETRY_BACKOFF_MULTIPLIER=2.0    # Backoff multiplier
```

#### Request Timeouts
```bash
JENKINS_DEFAULT_TIMEOUT=10     # Default timeout (seconds)
JENKINS_HEALTH_TIMEOUT=5       # Health check timeout (seconds)
```

#### Cache Configuration
```bash
JENKINS_CRUMB_CACHE_MINUTES=30          # CSRF crumb cache duration
JENKINS_CACHE_STATIC_TTL=3600           # 1 hour for static data
JENKINS_CACHE_SEMI_STATIC_TTL=300       # 5 minutes for semi-static
JENKINS_CACHE_DYNAMIC_TTL=30            # 30 seconds for dynamic
JENKINS_CACHE_SHORT_TTL=10              # 10 seconds for short-lived
```

#### Cache Size Limits
```bash
JENKINS_CACHE_STATIC_SIZE=1000          # Max static cache items
JENKINS_CACHE_SEMI_STATIC_SIZE=500      # Max semi-static items
JENKINS_CACHE_DYNAMIC_SIZE=200          # Max dynamic items
JENKINS_CACHE_PERMANENT_SIZE=2000       # Max permanent items
```

#### Content Limits
```bash
JENKINS_MAX_LOG_SIZE=1000              # Max log content (chars)
JENKINS_MAX_CONTENT_SIZE=10000         # Max general content (chars)
JENKINS_MAX_ARTIFACT_SIZE_MB=50        # Max artifact download (MB)
```

#### Search Defaults
```bash
JENKINS_DEFAULT_MAX_DEPTH=10           # Max folder traversal depth
JENKINS_DEFAULT_MAX_BUILDS=10          # Max builds to search
```

## Usage

### Running the Server

#### STDIO Mode (Direct Interaction)
```bash
python jenkins_mcp_server_enhanced.py
```

#### HTTP Mode (MCP Gateway Compatible)
```bash
python jenkins_mcp_server_enhanced.py --transport streamable-http --port 8010
```

#### Custom Configuration
```bash
python jenkins_mcp_server_enhanced.py \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 8080
```

### Tool Examples

#### Trigger a Simple Job
```python
trigger_job(job_name="my-build-job")
```

#### Trigger Job with Parameters
```python
trigger_job(
    job_name="deploy-app",
    params={
        "environment": "staging",
        "version": "1.2.3",
        "notify": true
    }
)
```

#### Trigger Nested Job
```python
trigger_job(
    job_name="team-a/backend/build-service",
    params={"branch": "main"}
)
```

#### List Jobs with Filtering
```python
# List all jobs recursively
list_jobs(recursive=True)

# List only building jobs
list_jobs(status_filter="building")

# List failed jobs from last 7 days
list_jobs(
    last_build_result="FAILURE",
    days_since_last_build=7
)

# List only enabled jobs
list_jobs(enabled_only=True)
```

#### Search Jobs
```python
# Wildcard search
search_jobs(pattern="*backend*")

# Regex search
search_jobs(pattern="^deploy-.*-prod$", use_regex=True)

# Search with filters
search_jobs(
    pattern="*test*",
    status_filter="idle",
    last_build_result="SUCCESS"
)
```

#### Monitor Build Status
```python
# Get build status
get_build_status(job_name="my-job", build_number=42)

# Get console log
get_console_log(job_name="my-job", build_number=42)

# Get console log from offset
get_console_log(job_name="my-job", build_number=42, start=5000)
```

#### Pipeline Operations
```python
# Get pipeline stage details
get_pipeline_status(job_name="my-pipeline", build_number=10)
```

#### Artifact Management
```python
# List artifacts
list_build_artifacts(job_name="build-job", build_number=15)

# Download artifact
download_build_artifact(
    job_name="build-job",
    build_number=15,
    artifact_path="dist/app.jar"
)

# Search artifacts across builds
search_build_artifacts(
    job_name="build-job",
    pattern="*.log",
    max_builds=5
)
```

#### Batch Operations
```python
# Trigger multiple jobs
batch_trigger_jobs(
    operations=[
        {"job_name": "unit-tests", "priority": 1},
        {"job_name": "integration-tests", "priority": 2},
        {"job_name": "deploy-staging", "params": {"version": "1.0"}, "priority": 3}
    ],
    max_concurrent=3,
    fail_fast=False
)

# Monitor batch operation
batch_monitor_jobs(operation_id="abc123")

# Cancel batch operation
batch_cancel_jobs(operation_id="abc123", cancel_running_builds=True)
```

#### Cache Management
```python
# Get cache statistics
get_cache_statistics()

# Clear all caches
clear_cache(cache_type="all")

# Clear cache for specific job
clear_cache(job_name="my-job")

# Clear specific cache type
clear_cache(cache_type="dynamic")

# Warm cache
warm_cache(operations=["server_info", "job_list"])
```

### MCP Client Integration

#### With Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "jenkins": {
      "command": "python",
      "args": ["/path/to/jenkins_mcp_server_enhanced.py"],
      "env": {
        "JENKINS_URL": "http://your-jenkins:8080",
        "JENKINS_USER": "your-username",
        "JENKINS_API_TOKEN": "your-token"
      }
    }
  }
}
```

#### With MCP CLI

```bash
# List available tools
mcp-cli tools --server jenkins

# Trigger a job
mcp-cli cmd --server jenkins --tool trigger_job --args '{"job_name": "my-job"}'

# Get build status
mcp-cli cmd --server jenkins --tool get_build_status --args '{"job_name": "my-job", "build_number": 10}'
```

## Architecture

### Caching Strategy

The server implements a sophisticated multi-tier caching system:

```
┌─────────────────┬─────────────┬──────────────────┐
│ Cache Type      │ TTL         │ Use Case         │
├─────────────────┼─────────────┼──────────────────┤
│ Static          │ 1 hour      │ Server info      │
│ Semi-Static     │ 5 minutes   │ Job lists        │
│ Dynamic         │ 30 seconds  │ Running builds   │
│ Permanent (LRU) │ Never       │ Completed builds │
│ Short-lived     │ 10 seconds  │ Console logs     │
└─────────────────┴─────────────┴──────────────────┘
```

**Smart Build Caching**: Automatically migrates completed builds from dynamic to permanent cache.

### Retry Logic

```python
Retry Sequence:
Attempt 1: Immediate
Attempt 2: ~1-2 seconds delay
Attempt 3: ~2-4 seconds delay
Attempt 4: ~4-8 seconds delay

With jitter (0.1-0.9x) to prevent thundering herd
```

**Retryable Conditions**:
- HTTP 429 (Too Many Requests)
- HTTP 500-504 (Server Errors)
- Connection timeouts
- Network errors

### Error Handling

All errors include:
- Clear error message
- Actionable suggestions
- Context information
- HTTP status (when applicable)

Example:
```json
{
  "error": "Job 'my-job' not found",
  "suggestion": "Use search_jobs('*my-job*') to find available jobs",
  "operation": "trigger job"
}
```

## Troubleshooting

### Connection Issues

**Problem**: `Connection failed during operation`

**Solutions**:
1. Verify JENKINS_URL is correct and accessible
2. Check network connectivity to Jenkins server
3. Ensure firewall allows access to Jenkins port
4. Try with increased timeout: `JENKINS_DEFAULT_TIMEOUT=30`

### Authentication Errors

**Problem**: `Authentication failed during operation`

**Solutions**:
1. Verify JENKINS_USER is correct
2. Generate new API token in Jenkins
3. Ensure token has proper permissions
4. Check for special characters in credentials (escape if needed)

### Job Not Found

**Problem**: `Job 'xyz' not found`

**Solutions**:
1. Use `list_jobs(recursive=True)` to see all jobs
2. Use `search_jobs('*xyz*')` for pattern matching
3. Check if job is in a folder (use full path: `folder/job`)
4. Try `get_job_info('xyz', auto_search=True)` for auto-search

### Slow Performance

**Solutions**:
1. Check cache statistics: `get_cache_statistics()`
2. Warm cache on startup: `warm_cache()`
3. Increase cache sizes via environment variables
4. Reduce `max_depth` for recursive operations

## API Reference

See [tools.json](tools.json) for complete tool schemas.

### Tool Categories

- **Job Management**: `trigger_job`, `get_job_info`, `list_jobs`, `get_folder_info`, `search_jobs`, `search_and_trigger`
- **Build Monitoring**: `get_build_status`, `get_console_log`, `get_queue_info`, `summarize_build_log`
- **Pipeline**: `get_pipeline_status`
- **Artifacts**: `list_build_artifacts`, `download_build_artifact`, `search_build_artifacts`
- **Batch**: `batch_trigger_jobs`, `batch_monitor_jobs`, `batch_cancel_jobs`
- **Cache**: `get_cache_statistics`, `clear_cache`, `warm_cache`
- **Server**: `server_info`, health check resource

## Performance Considerations

- **Cache Hit Rate**: Monitor with `get_cache_statistics()` (aim for >70%)
- **Batch Operations**: Use for triggering multiple jobs (up to 50% faster)
- **Recursive Depth**: Limit `max_depth` for large folder structures
- **Log Fetching**: Use `start` parameter for progressive log retrieval
- **Concurrent Requests**: Server handles concurrent operations safely

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/avisangle/jenkins-mcp/issues)
- **Documentation**: [MCP Documentation](https://modelcontextprotocol.io)
- **Jenkins API**: [Jenkins Remote API](https://www.jenkins.io/doc/book/using/remote-access-api/)

## Changelog

### Version 0.1.0 (Initial Release)

- 20+ tools for comprehensive Jenkins management
- Multi-tier caching system
- Exponential backoff retry logic
- Pipeline stage tracking
- Artifact management
- Batch operations support
- Advanced filtering and search
- Nested job path support
- CSRF protection
- Docker support

## Acknowledgments

- Built with [FastMCP](https://github.com/jlowin/fastmcp)
- Follows [Model Context Protocol](https://modelcontextprotocol.io) specification
- Powered by [Jenkins Remote Access API](https://www.jenkins.io/doc/book/using/remote-access-api/)
