import logging
import traceback
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import openai
import anthropic

logger = logging.getLogger("middleware")

class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Centralized exception handling middleware."""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except openai.OpenAIError as e:
            logger.warning(f"AI API Error: {e}")
            
            # Map OpenAI errors to appropriate status codes
            status_code = status.HTTP_400_BAD_REQUEST
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                status_code = status.HTTP_429_TOO_MANY_REQUESTS
            
            # Extract readable message if possible
            error_details = str(e)
            if hasattr(e, 'body') and isinstance(e.body, dict):
                error_details = e.body.get('error', {}).get('message', str(e))
            elif hasattr(e, 'message'):
                error_details = e.message
                
            return JSONResponse(
                status_code=status_code,
                content={
                    "error": e.__class__.__name__,
                    "detail": error_details
                }
            )
        except anthropic.AnthropicError as e:
            logger.warning(f"Anthropic API Error: {e}")
            status_code = status.HTTP_400_BAD_REQUEST
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                status_code = status.HTTP_429_TOO_MANY_REQUESTS
                
            return JSONResponse(
                status_code=status_code,
                content={
                    "error": e.__class__.__name__,
                    "detail": str(e)
                }
            )
        except Exception as e:
            logger.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal Server Error",
                    "detail": str(e) if request.app.debug else "An unexpected error occurred."
                }
            )
