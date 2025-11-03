FROM python:3.12-slim

LABEL org.opencontainers.image.title="Jenkins MCP Server"
LABEL org.opencontainers.image.description="Comprehensive Jenkins automation with 30+ tools for CI/CD management"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir -e .

# Copy application code
COPY jenkins_mcp_server_enhanced.py ./

# Environment variables (will be overridden at runtime)
ENV JENKINS_URL=""
ENV JENKINS_USER=""
ENV JENKINS_API_TOKEN=""
ENV MCP_PORT="8010"
ENV MCP_HOST="0.0.0.0"

# Run the MCP server
CMD ["python", "jenkins_mcp_server_enhanced.py"]
