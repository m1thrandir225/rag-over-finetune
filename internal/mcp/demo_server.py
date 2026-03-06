from datetime import datetime

try:
    from fastmcp import FastMCP
except ImportError:
    print("ERROR: fastmcp not installed. Run: pip install fastmcp")
    exit(1)

mcp = FastMCP("DemoServer", version="1.0.0")


@mcp.tool()
def get_current_time(timezone: str = "UTC") -> str:
    try:
        try:
            import pytz

            tz = pytz.timezone(timezone)
            current_time = datetime.now(tz)
            return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        except ImportError:
            if timezone.upper() != "UTC":
                return f"Note: pytz not installed, showing system time instead of {timezone}. " + datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            current_time = datetime.utcnow()
            return current_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")