from typing import List

from fastapi import APIRouter, HTTPException, Path, Body, Depends
from pydantic import BaseModel

from .modules.integration import get_container
from .modules.llm.template_manager import TemplateManager

router = APIRouter(prefix="/api/templates", tags=["templates"])


# Dependencies
def get_template_manager():
    container = get_container()
    return container.get_template_manager()


# Pydantic Models
class TemplateResponse(BaseModel):
    id: int
    name: str
    content: str
    prompt_variables: str
    created_at: str
    updated_at: str


class TemplateCreateRequest(BaseModel):
    name: str
    content: str
    prompt_variables: str = ''


class TemplateUpdateRequest(BaseModel):
    content: str
    prompt_variables: str = None


# Endpoints
@router.get("", response_model=List[TemplateResponse])
async def list_templates(
        manager: TemplateManager = Depends(get_template_manager)
):
    """List all prompt templates."""
    try:
        return manager.list_templates()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=TemplateResponse)
async def create_template(
        request: TemplateCreateRequest,
        manager: TemplateManager = Depends(get_template_manager)
):
    """Create a new prompt template."""
    try:
        if manager.get_template(request.name):
            raise HTTPException(status_code=400, detail=f"Template '{request.name}' already exists")

        return manager.create_template(request.name, request.content, request.prompt_variables)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{name}", response_model=TemplateResponse)
async def get_template(
        name: str = Path(..., description="Name of the template"),
        manager: TemplateManager = Depends(get_template_manager)
):
    """Get a specific prompt template by name."""
    template = manager.get_template(name)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    return template


@router.put("/{name}", response_model=TemplateResponse)
async def update_template(
        name: str = Path(..., description="Name of the template"),
        request: TemplateUpdateRequest = Body(...),
        manager: TemplateManager = Depends(get_template_manager)
):
    """Update a prompt template's content."""
    updated = manager.update_template(name, request.content, request.prompt_variables)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    return updated


@router.delete("/{name}")
async def delete_template(
        name: str = Path(..., description="Name of the template"),
        manager: TemplateManager = Depends(get_template_manager)
):
    """Delete a prompt template."""
    deleted = manager.delete_template(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    return {"message": f"Template '{name}' deleted successfully"}
