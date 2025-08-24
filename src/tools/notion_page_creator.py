import json
import logging
from typing import Any, Dict, List

import requests
from pydantic import BaseModel, Field

from portia import Tool, ToolRunContext
from src.config import settings

logger = logging.getLogger(__name__)


class CreateNotionPageArgs(BaseModel):
    pages: List[Dict[str, Any]] = Field(..., description="List of pages to create")


class CreateNotionPageOutput(BaseModel):
    status: str
    pages_created: int
    page_ids: List[str]


class CustomNotionPageCreatorTool(Tool[CreateNotionPageOutput]):
    """Custom tool to create Notion pages directly via API, bypassing MCP issues."""

    id: str = "custom_notion_page_creator"
    name: str = "Custom Notion Page Creator"
    description: str = "Create Notion pages directly via API"
    args_schema: type[BaseModel] = CreateNotionPageArgs
    output_schema: tuple[str, str] = (
        "CreateNotionPageOutput",
        "Result of page creation with status and page IDs",
    )

    def _convert_markdown_to_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Convert markdown content to Notion blocks."""
        blocks = []
        lines = content.split("\n")

        current_paragraph = []
        in_code_block = False
        code_block_content = []
        code_language = "text"

        for line in lines:
            # Handle code blocks
            if line.strip().startswith("```"):
                if not in_code_block:
                    # Starting a code block
                    # Finalize current paragraph first
                    if current_paragraph:
                        paragraph_text = " ".join(current_paragraph).strip()
                        if paragraph_text:
                            blocks.append(
                                {
                                    "object": "block",
                                    "type": "paragraph",
                                    "paragraph": {
                                        "rich_text": [
                                            {
                                                "type": "text",
                                                "text": {"content": paragraph_text},
                                            }
                                        ]
                                    },
                                }
                            )
                        current_paragraph = []

                    in_code_block = True
                    # Extract language if specified
                    lang_part = line.strip()[3:].strip()
                    code_language = lang_part if lang_part else "text"
                    continue
                else:
                    # Ending a code block
                    in_code_block = False
                    if code_block_content:
                        code_text = "\n".join(code_block_content)
                        blocks.append(
                            {
                                "object": "block",
                                "type": "code",
                                "code": {
                                    "rich_text": [
                                        {"type": "text", "text": {"content": code_text}}
                                    ],
                                    "language": code_language,
                                },
                            }
                        )
                    code_block_content = []
                    continue

            if in_code_block:
                code_block_content.append(line)
                continue

            line = line.strip()

            # Empty line - finalize current paragraph if exists
            if not line:
                if current_paragraph:
                    paragraph_text = " ".join(current_paragraph).strip()
                    if paragraph_text:
                        blocks.append(
                            {
                                "object": "block",
                                "type": "paragraph",
                                "paragraph": {
                                    "rich_text": [
                                        {
                                            "type": "text",
                                            "text": {"content": paragraph_text},
                                        }
                                    ]
                                },
                            }
                        )
                    current_paragraph = []
                continue

            # Headers
            if line.startswith("# "):
                # Finalize current paragraph first
                if current_paragraph:
                    paragraph_text = " ".join(current_paragraph).strip()
                    if paragraph_text:
                        blocks.append(
                            {
                                "object": "block",
                                "type": "paragraph",
                                "paragraph": {
                                    "rich_text": [
                                        {
                                            "type": "text",
                                            "text": {"content": paragraph_text},
                                        }
                                    ]
                                },
                            }
                        )
                    current_paragraph = []

                # Add heading
                heading_text = line[2:].strip()
                blocks.append(
                    {
                        "object": "block",
                        "type": "heading_1",
                        "heading_1": {
                            "rich_text": [
                                {"type": "text", "text": {"content": heading_text}}
                            ]
                        },
                    }
                )
            elif line.startswith("## "):
                # Finalize current paragraph first
                if current_paragraph:
                    paragraph_text = " ".join(current_paragraph).strip()
                    if paragraph_text:
                        blocks.append(
                            {
                                "object": "block",
                                "type": "paragraph",
                                "paragraph": {
                                    "rich_text": [
                                        {
                                            "type": "text",
                                            "text": {"content": paragraph_text},
                                        }
                                    ]
                                },
                            }
                        )
                    current_paragraph = []

                # Add heading 2
                heading_text = line[3:].strip()
                blocks.append(
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {
                            "rich_text": [
                                {"type": "text", "text": {"content": heading_text}}
                            ]
                        },
                    }
                )
            elif line.startswith("### "):
                # Finalize current paragraph first
                if current_paragraph:
                    paragraph_text = " ".join(current_paragraph).strip()
                    if paragraph_text:
                        blocks.append(
                            {
                                "object": "block",
                                "type": "paragraph",
                                "paragraph": {
                                    "rich_text": [
                                        {
                                            "type": "text",
                                            "text": {"content": paragraph_text},
                                        }
                                    ]
                                },
                            }
                        )
                    current_paragraph = []

                # Add heading 3
                heading_text = line[4:].strip()
                blocks.append(
                    {
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {
                            "rich_text": [
                                {"type": "text", "text": {"content": heading_text}}
                            ]
                        },
                    }
                )
            elif line.startswith("> "):
                # Finalize current paragraph first
                if current_paragraph:
                    paragraph_text = " ".join(current_paragraph).strip()
                    if paragraph_text:
                        blocks.append(
                            {
                                "object": "block",
                                "type": "paragraph",
                                "paragraph": {
                                    "rich_text": [
                                        {
                                            "type": "text",
                                            "text": {"content": paragraph_text},
                                        }
                                    ]
                                },
                            }
                        )
                    current_paragraph = []

                # Quote block
                quote_text = line[2:].strip()
                blocks.append(
                    {
                        "object": "block",
                        "type": "quote",
                        "quote": {
                            "rich_text": [
                                {"type": "text", "text": {"content": quote_text}}
                            ]
                        },
                    }
                )
            elif line.startswith("- ") or line.startswith("* "):
                # Finalize current paragraph first
                if current_paragraph:
                    paragraph_text = " ".join(current_paragraph).strip()
                    if paragraph_text:
                        blocks.append(
                            {
                                "object": "block",
                                "type": "paragraph",
                                "paragraph": {
                                    "rich_text": [
                                        {
                                            "type": "text",
                                            "text": {"content": paragraph_text},
                                        }
                                    ]
                                },
                            }
                        )
                    current_paragraph = []

                # Bulleted list item
                list_text = line[2:].strip()
                blocks.append(
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [
                                {"type": "text", "text": {"content": list_text}}
                            ]
                        },
                    }
                )
            elif line.startswith("---"):
                # Finalize current paragraph first
                if current_paragraph:
                    paragraph_text = " ".join(current_paragraph).strip()
                    if paragraph_text:
                        blocks.append(
                            {
                                "object": "block",
                                "type": "paragraph",
                                "paragraph": {
                                    "rich_text": [
                                        {
                                            "type": "text",
                                            "text": {"content": paragraph_text},
                                        }
                                    ]
                                },
                            }
                        )
                    current_paragraph = []

                # Divider
                blocks.append({"object": "block", "type": "divider", "divider": {}})
            else:
                # Regular text - add to current paragraph
                current_paragraph.append(line)

        # Finalize any remaining paragraph
        if current_paragraph:
            paragraph_text = " ".join(current_paragraph).strip()
            if paragraph_text:
                blocks.append(
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"type": "text", "text": {"content": paragraph_text}}
                            ]
                        },
                    }
                )

        # Finalize any remaining code block
        if in_code_block and code_block_content:
            code_text = "\n".join(code_block_content)
            blocks.append(
                {
                    "object": "block",
                    "type": "code",
                    "code": {
                        "rich_text": [{"type": "text", "text": {"content": code_text}}],
                        "language": code_language,
                    },
                }
            )

        return blocks

    def _create_page_properties(self, page_title: str) -> Dict[str, Any]:
        """Create page properties based on the database schema."""
        # Basic properties that should work with most Notion databases
        properties = {
            "title": {"title": [{"type": "text", "text": {"content": page_title}}]}
        }

        return properties

    def run(
        self, ctx: ToolRunContext, pages: List[Dict[str, Any]]
    ) -> CreateNotionPageOutput:
        """Create Notion pages directly via API."""
        logger.info(f"CustomNotionPageCreatorTool - Received pages type: {type(pages)}")
        logger.info(
            f"CustomNotionPageCreatorTool - Number of pages: {len(pages) if isinstance(pages, list) else 'Not a list'}"
        )

        # Handle the case where pages is passed as a JSON string
        if isinstance(pages, str):
            logger.info("Pages received as JSON string, attempting to parse...")
            try:
                pages = json.loads(pages)
                logger.info(
                    f"Successfully parsed pages JSON. New type: {type(pages)}, length: {len(pages)}"
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse pages JSON: {e}")
                return CreateNotionPageOutput(
                    status=f"error: Invalid pages JSON: {str(e)}",
                    pages_created=0,
                    page_ids=[],
                )

        # Ensure pages is a list
        if not isinstance(pages, list):
            logger.error(f"Pages must be a list, got: {type(pages)}")
            return CreateNotionPageOutput(
                status=f"error: Pages must be a list, got {type(pages)}",
                pages_created=0,
                page_ids=[],
            )

        # Ensure we have the Notion token
        notion_token = settings.notion_token
        notion_database_id = settings.notion_database_id

        if not notion_token:
            logger.error("No Notion token available")
            return CreateNotionPageOutput(
                status="error: No Notion token configured",
                pages_created=0,
                page_ids=[],
            )

        try:
            logger.info(f"Creating {len(pages)} Notion pages via API...")
            page_ids = []

            headers = {
                "Authorization": f"Bearer {notion_token}",
                "Content-Type": "application/json",
                "Notion-Version": "2022-06-28",
            }

            for i, page in enumerate(pages):
                logger.info(f"Processing page {i+1}/{len(pages)}...")

                # Extract page data
                page_title = page.get("properties", {}).get(
                    "title", f"Compliance Analysis {i+1}"
                )
                page_content = page.get("content", "")

                logger.info(f"Page title: {page_title}")
                logger.info(f"Content length: {len(page_content)} characters")

                # Convert content to Notion blocks
                content_blocks = self._convert_markdown_to_blocks(page_content)
                logger.info(f"Generated {len(content_blocks)} content blocks")

                # Determine how to create the page based on available configuration
                if notion_database_id:
                    # Create page in database if database ID is available
                    page_properties = self._create_page_properties(page_title)
                    page_payload = {
                        "parent": {"database_id": notion_database_id},
                        "properties": page_properties,
                        "children": content_blocks[
                            :100
                        ],  # Notion has limits on blocks per request
                    }
                    logger.info(f"Creating page in database: {notion_database_id}")
                else:
                    # Create standalone page without database
                    # Since we don't have a database ID, we'll need to find a suitable parent
                    # Let's first try to get the user's pages to find a suitable parent
                    logger.info(
                        "No database ID provided, attempting to find a parent page"
                    )

                    # Try to get a list of pages to find a suitable parent
                    search_response = requests.post(
                        "https://api.notion.com/v1/search",
                        headers=headers,
                        json={
                            "query": "",
                            "filter": {"value": "page", "property": "object"},
                            "page_size": 10,
                        },
                        timeout=10,
                    )

                    parent_id = None
                    if search_response.status_code == 200:
                        search_data = search_response.json()
                        results = search_data.get("results", [])
                        if results:
                            # Use the first page found as parent
                            parent_id = results[0].get("id")
                            logger.info(f"Found parent page: {parent_id}")
                        else:
                            logger.warning("No existing pages found for parent")
                    else:
                        logger.warning(
                            f"Failed to search for pages: {search_response.text}"
                        )

                    if not parent_id:
                        # If we can't find a parent, return an error
                        logger.error(
                            "Cannot create page: no database ID and no suitable parent page found"
                        )
                        return CreateNotionPageOutput(
                            status="error: No database ID provided and no suitable parent page found. Please provide NOTION_DATABASE_ID or create some pages in your Notion workspace first.",
                            pages_created=0,
                            page_ids=[],
                        )

                    # Create page as child of found parent
                    page_payload = {
                        "parent": {"page_id": parent_id},
                        "properties": {
                            "title": {
                                "title": [
                                    {"type": "text", "text": {"content": page_title}}
                                ]
                            }
                        },
                        "children": content_blocks[
                            :100
                        ],  # Notion has limits on blocks per request
                    }

                logger.info("Making API request to create Notion page...")
                logger.debug(
                    f"Page payload: {json.dumps(page_payload, indent=2)[:500]}..."
                )

                # Make the API request
                response = requests.post(
                    "https://api.notion.com/v1/pages",
                    headers=headers,
                    json=page_payload,
                    timeout=30,
                )

                logger.info(f"Notion API response status: {response.status_code}")

                if response.status_code == 200:
                    page_data = response.json()
                    page_id = page_data.get("id")
                    page_ids.append(page_id)
                    logger.info(f"Successfully created Notion page: {page_id}")

                    # If we have more than 100 blocks, add them in additional requests
                    if len(content_blocks) > 100:
                        logger.info(
                            f"Adding remaining {len(content_blocks) - 100} blocks..."
                        )
                        remaining_blocks = content_blocks[100:]

                        # Split into chunks of 100
                        for chunk_start in range(0, len(remaining_blocks), 100):
                            chunk_end = min(chunk_start + 100, len(remaining_blocks))
                            chunk_blocks = remaining_blocks[chunk_start:chunk_end]

                            append_response = requests.patch(
                                f"https://api.notion.com/v1/blocks/{page_id}/children",
                                headers=headers,
                                json={"children": chunk_blocks},
                                timeout=30,
                            )

                            if append_response.status_code != 200:
                                logger.warning(
                                    f"Failed to append blocks chunk {chunk_start}-{chunk_end}: {append_response.text}"
                                )
                            else:
                                logger.info(
                                    f"Successfully appended blocks {chunk_start}-{chunk_end}"
                                )

                else:
                    error_text = response.text
                    logger.error(
                        f"Failed to create Notion page {i+1}: HTTP {response.status_code}"
                    )
                    logger.error(f"Error response: {error_text}")

                    # Try to parse error for more specific feedback
                    try:
                        error_data = response.json()
                        error_message = error_data.get("message", "Unknown error")
                        logger.error(f"Notion API error: {error_message}")
                    except json.JSONDecodeError:
                        pass

                    # Continue with other pages even if one fails
                    continue

            if page_ids:
                logger.info(f"Successfully created {len(page_ids)} Notion pages")
                return CreateNotionPageOutput(
                    status="success",
                    pages_created=len(page_ids),
                    page_ids=page_ids,
                )
            else:
                logger.error("No pages were created successfully")
                return CreateNotionPageOutput(
                    status="error: No pages created",
                    pages_created=0,
                    page_ids=[],
                )

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error creating Notion pages: {e}")
            return CreateNotionPageOutput(
                status=f"network_error: {str(e)}",
                pages_created=0,
                page_ids=[],
            )
        except Exception as e:
            logger.error(f"Error in CustomNotionPageCreatorTool: {e}", exc_info=True)
            return CreateNotionPageOutput(
                status=f"error: {str(e)}",
                pages_created=0,
                page_ids=[],
            )
