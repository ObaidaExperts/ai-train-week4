import logging
import traceback
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("middleware")

class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Centralized exception handling middleware."""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal Server Error",
                    "detail": str(e) if request.app.debug else "An unexpected error occurred."
                }
            )
