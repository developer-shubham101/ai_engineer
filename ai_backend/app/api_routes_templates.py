"""Template management API routes."""

import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, require_roles
from app.modules.config.constants import EMPLOYEE_PLUS_ROLES
from app.modules.integration import get_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/templates", tags=["Templates"])


class MessageModel(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content with variable placeholders")


class CreateTemplateRequest(BaseModel):
    name: str = Field(..., description="Template name")
    messages: List[MessageModel] = Field(..., description="Array of messages")
    prompt_variables: str = Field(default="", description="Pipe-separated variable names")


class UpdateTemplateRequest(BaseModel):
    messages: Optional[List[MessageModel]] = Field(default=None, description="Array of messages")
    prompt_variables: Optional[str] = Field(default=None, description="Pipe-separated variable names")


class TemplateResponse(BaseModel):
    id: int
    name: str
    messages: List[Dict[str, str]]
    prompt_variables: str
    created_at: str
    updated_at: str


@router.post("", response_model=TemplateResponse, dependencies=[Depends(require_roles(EMPLOYEE_PLUS_ROLES))])
async def create_template(
    request: CreateTemplateRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new template with message array."""
    try:
        container = get_container()
        container.initialize()
        template_manager = container.get_template_manager()
        
        # Convert Pydantic models to dict
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        template = template_manager.create_template(
            name=request.name,
            messages=messages,
            prompt_variables=request.prompt_variables
        )
        
        # Ensure messages is a list
        if isinstance(template.get('messages'), str):
            import json
            try:
                template['messages'] = json.loads(template['messages'])
            except (json.JSONDecodeError, TypeError):
                template['messages'] = []
        elif not isinstance(template.get('messages'), list):
            template['messages'] = []
        
        return TemplateResponse(**template)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[TemplateResponse])
async def list_templates():
    """List all templates."""
    try:
        container = get_container()
        container.initialize()
        template_manager = container.get_template_manager()
        
        templates = template_manager.list_templates()
        
        # Ensure messages are properly formatted
        response_templates = []
        for template in templates:
            # Ensure messages is a list
            if isinstance(template.get('messages'), str):
                import json
                try:
                    template['messages'] = json.loads(template['messages'])
                except (json.JSONDecodeError, TypeError):
                    template['messages'] = []
            elif not isinstance(template.get('messages'), list):
                template['messages'] = []
            
            response_templates.append(TemplateResponse(**template))
        
        return response_templates
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{name}", response_model=TemplateResponse)
async def get_template(name: str):
    """Get a specific template."""
    try:
        container = get_container()
        container.initialize()
        template_manager = container.get_template_manager()
        
        template = template_manager.get_template(name)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
        
        # Ensure messages is a list
        if isinstance(template.get('messages'), str):
            import json
            try:
                template['messages'] = json.loads(template['messages'])
            except (json.JSONDecodeError, TypeError):
                template['messages'] = []
        elif not isinstance(template.get('messages'), list):
            template['messages'] = []
        
        return TemplateResponse(**template)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{name}", response_model=TemplateResponse, dependencies=[Depends(require_roles(EMPLOYEE_PLUS_ROLES))])
async def update_template(
    name: str,
    request: UpdateTemplateRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Update an existing template."""
    try:
        container = get_container()
        container.initialize()
        template_manager = container.get_template_manager()
        
        # Convert Pydantic models to dict if provided
        messages = None
        if request.messages:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        template = template_manager.update_template(
            name=name,
            messages=messages,
            prompt_variables=request.prompt_variables
        )
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
        
        # Ensure messages is a list
        if isinstance(template.get('messages'), str):
            import json
            try:
                template['messages'] = json.loads(template['messages'])
            except (json.JSONDecodeError, TypeError):
                template['messages'] = []
        elif not isinstance(template.get('messages'), list):
            template['messages'] = []
        
        return TemplateResponse(**template)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}", dependencies=[Depends(require_roles(EMPLOYEE_PLUS_ROLES))])
async def delete_template(
    name: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a template."""
    try:
        container = get_container()
        container.initialize()
        template_manager = container.get_template_manager()
        
        success = template_manager.delete_template(name)
        if not success:
            raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
        
        return {"message": f"Template '{name}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/{name}")
async def test_template(
    name: str,
    test_data: Dict[str, str] = None
):
    """Test a template with sample data."""
    try:
        container = get_container()
        container.initialize()
        template_manager = container.get_template_manager()
        
        template = template_manager.get_template(name)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
        
        # Use provided test data or defaults
        variables = {
            'user_question': 'What is the capital of France?',
            'source_docs': 'Sample document content...',
            'history': '',
            'user_role': 'Employee',
            'department': 'General',
            'user_profile_summary': 'Test user profile',
            'max_tokens': '256'
        }
        
        if test_data:
            variables.update(test_data)
        
        # Process template messages
        messages = template['messages'].copy()
        for message in messages:
            content = message['content']
            for var_name, var_value in variables.items():
                content = content.replace(f'{{{var_name}}}', var_value)
            message['content'] = content
        
        return {
            "template_name": name,
            "processed_messages": messages,
            "variables_used": variables
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test template: {e}")
        raise HTTPException(status_code=500, detail=str(e))